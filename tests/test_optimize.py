from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import optuna

from optimize import (
    ValidationConfig,
    _walk_forward_splits,
    load_training_data,
    make_objective,
)
from src.data import AlpacaMarketDataStore, build_feature_dataset
from tests.fixtures import make_multi_symbol_bars, make_sample_bars


class TrialStub:
    def __init__(self, number: int = 0):
        self.number = number
        self.params = {}
        self.user_attrs = {}

    def suggest_categorical(self, name, choices):
        value = choices[0]
        self.params[name] = value
        return value

    def suggest_int(self, name, low, high):
        self.params[name] = low
        return low

    def suggest_float(self, name, low, high, log=False):
        self.params[name] = low
        return low

    def set_user_attr(self, name, value):
        self.user_attrs[name] = value


class TestOptimize(unittest.TestCase):
    def test_load_training_data_from_cached_parquet(self):
        bars = make_sample_bars(periods=80)
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AlpacaMarketDataStore(storage_root=tmpdir)
            store.persist_bars(bars)

            dataset = load_training_data(
                symbols=["AAPL"],
                timeframe="1Min",
                start=datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, 15, 49, tzinfo=timezone.utc),
                storage_root=Path(tmpdir),
                download=False,
            )

            self.assertFalse(dataset.features.empty)
            self.assertEqual(len(dataset.features), len(dataset.target))
            self.assertEqual(len(dataset.metadata), len(dataset.target))

    def test_temporal_split_correctness(self):
        bars = make_multi_symbol_bars(periods=60)
        dataset = build_feature_dataset(
            bars,
            return_horizon=1,
            volatility_window=5,
            atr_window=5,
            volume_window=5,
        )
        splits = _walk_forward_splits(
            dataset,
            ValidationConfig(
                min_train_timestamps=10,
                validation_timestamps=5,
                step_timestamps=5,
                max_folds=3,
            ),
        )

        self.assertGreaterEqual(len(splits), 1)
        metadata = dataset.metadata.reset_index(drop=True)
        for train_mask, validation_mask in splits:
            train_timestamps = set(metadata.loc[train_mask, "timestamp"])
            validation_timestamps = set(metadata.loc[validation_mask, "timestamp"])
            self.assertTrue(train_timestamps.isdisjoint(validation_timestamps))
            self.assertEqual(
                int(validation_mask.sum()),
                len(validation_timestamps) * metadata["symbol"].nunique(),
            )

    def test_reproducibility(self):
        bars = make_sample_bars(periods=80)
        dataset = build_feature_dataset(
            bars,
            return_horizon=1,
            volatility_window=5,
            atr_window=5,
            volume_window=5,
        )
        objective = make_objective(
            dataset,
            ValidationConfig(
                min_train_timestamps=10,
                validation_timestamps=5,
                step_timestamps=5,
                max_folds=2,
            ),
            seed=123,
        )
        score_first = objective(TrialStub())
        score_second = objective(TrialStub())
        self.assertAlmostEqual(score_first, score_second, places=12)

    def test_failure_handling_prunes_invalid_trials(self):
        bars = make_sample_bars(periods=80)
        dataset = build_feature_dataset(
            bars,
            return_horizon=1,
            volatility_window=5,
            atr_window=5,
            volume_window=5,
        )
        objective = make_objective(
            dataset,
            ValidationConfig(
                min_train_timestamps=10,
                validation_timestamps=5,
                step_timestamps=5,
                max_folds=2,
            ),
            seed=123,
        )

        with patch("optimize.DistributionalStrategy.fit", side_effect=ValueError("boom")):
            with self.assertRaises(optuna.TrialPruned):
                objective(TrialStub())

    def test_naive_timestamp_behavior(self):
        bars = make_sample_bars(periods=80, naive_timestamps=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AlpacaMarketDataStore(storage_root=tmpdir)
            store.persist_bars(bars)

            dataset = load_training_data(
                symbols=["AAPL"],
                timeframe="1Min",
                start=datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, 15, 49, tzinfo=timezone.utc),
                storage_root=Path(tmpdir),
                download=False,
            )

        self.assertTrue(
            all(ts.tzinfo is not None and str(ts.tzinfo) == "UTC" for ts in dataset.metadata["timestamp"])
        )

    def test_multi_symbol_split_correctness(self):
        bars = make_multi_symbol_bars(periods=40)
        dataset = build_feature_dataset(
            bars,
            return_horizon=1,
            volatility_window=5,
            atr_window=5,
            volume_window=5,
        )
        splits = _walk_forward_splits(
            dataset,
            ValidationConfig(
                min_train_timestamps=8,
                validation_timestamps=4,
                step_timestamps=4,
                max_folds=2,
            ),
        )

        metadata = dataset.metadata.reset_index(drop=True)
        for train_mask, validation_mask in splits:
            train_rows = metadata.loc[train_mask]
            validation_rows = metadata.loc[validation_mask]
            self.assertTrue(
                set(train_rows["timestamp"]).isdisjoint(set(validation_rows["timestamp"]))
            )
            self.assertEqual(
                validation_rows.groupby("timestamp")["symbol"].nunique().min(),
                metadata["symbol"].nunique(),
            )


if __name__ == "__main__":
    unittest.main()
