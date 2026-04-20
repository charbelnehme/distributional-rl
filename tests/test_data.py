from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import AlpacaMarketDataStore, FeatureDataset, build_feature_dataset
from tests.fixtures import make_multi_symbol_bars, make_sample_bars


class FakeBarSet:
    def __init__(self, frame: pd.DataFrame):
        self.df = frame.set_index(["symbol", "timestamp"])


class FakeHistoricalClient:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.calls = []

    def get_stock_bars(self, request_params):
        self.calls.append(request_params)
        return FakeBarSet(self.frame)


class TestData(unittest.TestCase):
    def test_download_store_and_load_bars(self):
        bars = make_sample_bars(periods=20)
        client = FakeHistoricalClient(bars)

        with tempfile.TemporaryDirectory() as tmpdir:
            store = AlpacaMarketDataStore(client=client, storage_root=tmpdir)
            downloaded = store.download_stock_bars(
                ["AAPL"],
                start=datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, 14, 50, tzinfo=timezone.utc),
                timeframe="1Min",
                persist=True,
            )

            self.assertEqual(len(downloaded), 20)
            self.assertTrue(list(Path(tmpdir).rglob("*.parquet")))

            loaded = store.load_stock_bars(symbols=["AAPL"], timeframe="1Min")
            self.assertEqual(len(loaded), 20)
            self.assertIn("close", loaded.columns)

    def test_build_feature_dataset_preserves_metadata(self):
        bars = make_sample_bars(periods=40)
        dataset = build_feature_dataset(
            bars,
            return_horizon=1,
            volatility_window=5,
            atr_window=5,
            volume_window=5,
        )

        self.assertIsInstance(dataset, FeatureDataset)
        self.assertEqual(len(dataset.features), len(dataset.target))
        self.assertEqual(len(dataset.metadata), len(dataset.target))
        self.assertEqual(dataset.target.name, "future_log_return")
        self.assertEqual(
            list(dataset.features.columns),
            ["bar_portion", "log_return", "realised_vol", "atr_pct", "vol_percentile"],
        )
        self.assertEqual(list(dataset.metadata.columns), ["timestamp", "symbol", "timeframe"])

        X, y = dataset
        self.assertTrue(X.equals(dataset.features))
        self.assertTrue(y.equals(dataset.target))

    def test_build_feature_dataset_validates_positive_prices(self):
        bars = make_sample_bars(periods=10)
        bars.loc[0, "close"] = 0.0

        with self.assertRaises(ValueError):
            build_feature_dataset(
                bars,
                return_horizon=1,
                volatility_window=2,
                atr_window=2,
                volume_window=2,
            )

    def test_build_feature_dataset_aligns_return_horizon(self):
        bars = make_sample_bars(periods=20)
        dataset = build_feature_dataset(
            bars,
            return_horizon=2,
            volatility_window=3,
            atr_window=3,
            volume_window=3,
        )

        lookup = bars.copy()
        lookup["timestamp"] = pd.to_datetime(lookup["timestamp"], utc=True)
        lookup = lookup.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        first_metadata = dataset.metadata.iloc[0]
        row = lookup[
            (lookup["symbol"] == first_metadata["symbol"])
            & (lookup["timestamp"] == first_metadata["timestamp"])
        ].iloc[0]
        row_index = int(row.name)
        expected = np.log(lookup.loc[row_index + 2, "close"] / lookup.loc[row_index, "close"])
        self.assertAlmostEqual(dataset.target.iloc[0], expected, places=10)

    def test_timezone_consistency_for_naive_and_aware_timestamps(self):
        aware = make_sample_bars(periods=20)
        naive = make_sample_bars(periods=20, naive_timestamps=True)

        aware_dataset = build_feature_dataset(
            aware,
            return_horizon=1,
            volatility_window=3,
            atr_window=3,
            volume_window=3,
        )
        naive_dataset = build_feature_dataset(
            naive,
            return_horizon=1,
            volatility_window=3,
            atr_window=3,
            volume_window=3,
        )

        self.assertTrue(
            all(ts.tzinfo is not None and str(ts.tzinfo) == "UTC" for ts in aware_dataset.metadata["timestamp"])
        )
        self.assertTrue(
            all(ts.tzinfo is not None and str(ts.tzinfo) == "UTC" for ts in naive_dataset.metadata["timestamp"])
        )
        self.assertTrue(aware_dataset.metadata["timestamp"].equals(naive_dataset.metadata["timestamp"]))

    def test_cross_symbol_feature_scale_sanity(self):
        bars = make_multi_symbol_bars(periods=40)
        dataset = build_feature_dataset(
            bars,
            return_horizon=1,
            volatility_window=5,
            atr_window=5,
            volume_window=5,
        )

        scale_by_symbol = (
            dataset.features.join(dataset.metadata[["symbol"]]).groupby("symbol")["atr_pct"].mean()
        )
        ratio = float(scale_by_symbol.max() / scale_by_symbol.min())
        self.assertLess(ratio, 3.0)

    def test_duplicate_parquet_windows_are_deduplicated(self):
        bars = make_sample_bars(periods=20)
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AlpacaMarketDataStore(storage_root=tmpdir)
            store.persist_bars(bars.iloc[:15])
            store.persist_bars(bars.iloc[5:])

            loaded = store.load_stock_bars(symbols=["AAPL"], timeframe="1Min")
            self.assertEqual(len(loaded), len(bars))
            self.assertEqual(
                len(loaded.drop_duplicates(["symbol", "timestamp", "timeframe"])),
                len(loaded),
            )


if __name__ == "__main__":
    unittest.main()
