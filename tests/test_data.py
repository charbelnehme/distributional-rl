import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data import AlpacaMarketDataStore, build_feature_dataset
from tests.fixtures import make_sample_bars


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
            request = client.calls[0]
            symbols = getattr(request, "symbol_or_symbols", None)
            if symbols is None and isinstance(request, dict):
                symbols = request["symbol_or_symbols"]
            self.assertEqual(list(symbols), ["AAPL"])

    def test_build_feature_dataset(self):
        bars = make_sample_bars(periods=40)
        X, y = build_feature_dataset(
            bars,
            return_horizon=1,
            volatility_window=5,
            atr_window=5,
            volume_window=5,
        )

        self.assertFalse(X.empty)
        self.assertEqual(len(X), len(y))
        self.assertEqual(
            list(X.columns),
            ["bar_portion", "log_return", "realised_vol", "atr", "vol_percentile"],
        )
        self.assertEqual(y.name, "excess_return")

    def test_build_feature_dataset_rejects_non_positive_windows(self):
        bars = make_sample_bars(periods=40)

        with self.assertRaises(ValueError):
            build_feature_dataset(bars, return_horizon=0)

        with self.assertRaises(ValueError):
            build_feature_dataset(bars, volatility_window=0)

    def test_persist_bars_validates_required_columns(self):
        bars = make_sample_bars(periods=5).drop(columns=["timeframe"])

        with tempfile.TemporaryDirectory() as tmpdir:
            store = AlpacaMarketDataStore(storage_root=tmpdir)

            with self.assertRaises(ValueError):
                store.persist_bars(bars)


if __name__ == "__main__":
    unittest.main()
