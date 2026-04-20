import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import optimize
from optimize import (
    OptimizationConfig,
    TrainingDataset,
    ValidationConfig,
    _build_walk_forward_folds,
    load_training_data,
    make_objective,
    parse_utc_datetime,
    run_optimization,
)
from src.data import AlpacaMarketDataStore
from tests.fixtures import make_sample_bars


class StubStrategy:
    def __init__(self, model_params=None, simulation_params=None):
        self.model_params = model_params or {}
        self.simulation_params = simulation_params or {}

    def fit(self, X_train, y_train):
        self.train_rows = len(X_train)

    def predict_positions(self, X):
        base = 0.75 if self.model_params.get("dist_name") == "StudentT" else 1.0
        return np.full(len(X), base, dtype=float)


class RaisingStrategy(StubStrategy):
    def fit(self, X_train, y_train):
        raise RuntimeError("boom")


class TrialStub:
    def __init__(self):
        self.user_attrs = {}

    def suggest_categorical(self, name, choices):
        return choices[0]

    def suggest_int(self, name, low, high):
        return low

    def suggest_float(self, name, low, high, log=False):
        return low

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value


class TestOptimize(unittest.TestCase):
    def setUp(self):
        self._original_strategy = optimize.DistributionalStrategy

    def tearDown(self):
        optimize.DistributionalStrategy = self._original_strategy

    def _make_dataset(self, tmpdir: str, *, multi_symbol: bool = False) -> TrainingDataset:
        store = AlpacaMarketDataStore(storage_root=tmpdir)
        bars = make_sample_bars(periods=80)
        if multi_symbol:
            second = make_sample_bars(symbol="MSFT", periods=80)
            bars = pd.concat([bars, second], ignore_index=True)
        store.persist_bars(bars)
        return load_training_data(
            symbols=["AAPL"] if not multi_symbol else ["AAPL", "MSFT"],
            timeframe="1Min",
            start=datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc),
            end=datetime(2024, 1, 2, 15, 49, tzinfo=timezone.utc),
            storage_root=Path(tmpdir),
            download=False,
            return_metadata=True,
        )

    def test_load_training_data_returns_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = self._make_dataset(tmpdir)

            self.assertIsInstance(dataset, TrainingDataset)
            self.assertFalse(dataset.features.empty)
            self.assertEqual(len(dataset.features), len(dataset.target))
            self.assertEqual(len(dataset.features), len(dataset.metadata))
            self.assertIn("timestamp", dataset.metadata.columns)
            self.assertIn("symbol", dataset.metadata.columns)

    def test_walk_forward_folds_use_metadata_across_symbols(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = self._make_dataset(tmpdir, multi_symbol=True)
            folds = _build_walk_forward_folds(
                dataset.metadata,
                ValidationConfig(n_folds=2, min_train_rows=20, min_validation_rows=10),
            )

            self.assertGreaterEqual(len(folds), 2)
            for train_rows, validation_rows in folds:
                self.assertGreater(len(train_rows), 0)
                self.assertGreater(len(validation_rows), 0)
                symbols = set(dataset.metadata.iloc[validation_rows]["symbol"].unique())
                self.assertEqual(symbols, {"AAPL", "MSFT"})

    def test_naive_timestamp_parsing_assumes_utc(self):
        naive = parse_utc_datetime("2024-01-01T00:00:00")
        aware = parse_utc_datetime("2024-01-01T00:00:00Z")

        self.assertEqual(naive.tzinfo, timezone.utc)
        self.assertEqual(aware.tzinfo, timezone.utc)
        self.assertEqual(naive, aware)

    def test_objective_runs_with_stub_strategy_and_records_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = self._make_dataset(tmpdir)
            optimize.DistributionalStrategy = StubStrategy
            config = OptimizationConfig(
                symbols=("AAPL",),
                timeframe="1Min",
                start=datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, 15, 49, tzinfo=timezone.utc),
                storage_root=Path(tmpdir),
                download=False,
                trials=1,
                seed=7,
                validation=ValidationConfig(n_folds=2, min_train_rows=20, min_validation_rows=10),
                artifacts_dir=Path(tmpdir) / "artifacts",
            )

            trial = TrialStub()
            score = make_objective(dataset, config)(trial)

            self.assertIsInstance(score, float)
            self.assertIn("validation_folds", trial.user_attrs)
            self.assertIn("data_sizes", trial.user_attrs)
            self.assertEqual(trial.user_attrs["fold_count"], len(trial.user_attrs["validation_folds"]))
            self.assertGreater(trial.user_attrs["valid_fold_count"], 0)

    def test_objective_failure_records_structured_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = self._make_dataset(tmpdir)
            optimize.DistributionalStrategy = RaisingStrategy
            config = OptimizationConfig(
                symbols=("AAPL",),
                timeframe="1Min",
                start=datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, 15, 49, tzinfo=timezone.utc),
                storage_root=Path(tmpdir),
                download=False,
                trials=1,
                seed=7,
                validation=ValidationConfig(n_folds=2, min_train_rows=20, min_validation_rows=10),
                artifacts_dir=Path(tmpdir) / "artifacts",
            )

            trial = TrialStub()
            score = make_objective(dataset, config)(trial)

            self.assertEqual(score, -1e9)
            self.assertGreater(trial.user_attrs["failures"], 0)
            self.assertTrue(any(fold["failed"] for fold in trial.user_attrs["validation_folds"]))
            self.assertIn("boom", trial.user_attrs["validation_folds"][0]["error"])

    def test_seeded_optimization_is_reproducible_with_stub_strategy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = self._make_dataset(tmpdir)
            optimize.DistributionalStrategy = StubStrategy
            config = OptimizationConfig(
                symbols=("AAPL",),
                timeframe="1Min",
                start=datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, 15, 49, tzinfo=timezone.utc),
                storage_root=Path(tmpdir),
                download=False,
                trials=1,
                seed=99,
                validation=ValidationConfig(n_folds=2, min_train_rows=20, min_validation_rows=10),
                artifacts_dir=Path(tmpdir) / "artifacts",
            )

            first = run_optimization(dataset, config).best_trial.value
            second = run_optimization(dataset, config).best_trial.value

            self.assertAlmostEqual(first, second, places=12)


if __name__ == "__main__":
    unittest.main()
