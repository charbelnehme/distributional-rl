import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from distributional_rl.data import AlpacaMarketDataStore
from optimize import load_training_data, make_objective
from tests.fixtures import make_sample_bars


class TestOptimize(unittest.TestCase):
    def test_load_training_data_from_cached_parquet(self):
        bars = make_sample_bars(periods=80)
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AlpacaMarketDataStore(storage_root=tmpdir)
            store.persist_bars(bars)

            X, y = load_training_data(
                symbols=["AAPL"],
                timeframe="1Min",
                start=datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, 15, 49, tzinfo=timezone.utc),
                storage_root=Path(tmpdir),
                download=False,
            )

            self.assertFalse(X.empty)
            self.assertEqual(len(X), len(y))

    def test_objective_runs_with_real_feature_frame(self):
        bars = make_sample_bars(periods=80)
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AlpacaMarketDataStore(storage_root=tmpdir)
            store.persist_bars(bars)
            X, y = load_training_data(
                symbols=["AAPL"],
                timeframe="1Min",
                start=datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, 15, 49, tzinfo=timezone.utc),
                storage_root=Path(tmpdir),
                download=False,
            )

            class TrialStub:
                def suggest_categorical(self, name, choices):
                    return choices[0]

                def suggest_int(self, name, low, high):
                    return low

                def suggest_float(self, name, low, high, log=False):
                    return low

            score = make_objective(X, y)(TrialStub())
            self.assertIsInstance(score, float)


if __name__ == "__main__":
    unittest.main()
