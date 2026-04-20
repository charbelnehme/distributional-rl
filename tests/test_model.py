from __future__ import annotations

import unittest

import numpy as np

from src.model import DistributionalModel, DistributionalModelConfig
from tests.fixtures import make_training_frame


class TestModel(unittest.TestCase):
    def test_predict_before_fit_raises(self):
        model = DistributionalModel()
        X, _ = make_training_frame()
        with self.assertRaises(RuntimeError):
            model.predict_mean(X)

    def test_invalid_distribution_raises(self):
        X, y = make_training_frame(periods=80)
        model = DistributionalModel(dist_name="Bogus", n_estimators=5)
        with self.assertRaises(ValueError):
            model.fit(X, y)

    def test_fit_predict(self):
        X, y = make_training_frame(periods=80)
        model = DistributionalModel(n_estimators=10, dist_name="Normal", random_state=7)
        model.fit(X, y)

        dists = model.predict_dist(X)
        mean = model.predict_mean(X)
        scale = model.predict_scale(X)
        samples = model.sample(X.iloc[:5], n_samples=3, random_state=99)

        self.assertEqual(mean.shape[0], len(X))
        self.assertEqual(scale.shape[0], len(X))
        self.assertEqual(samples.shape[0], 3)
        self.assertEqual(samples.shape[1], 5)
        self.assertTrue(np.all(np.isfinite(mean)))
        self.assertTrue(np.all(scale >= 0.0))
        self.assertTrue(hasattr(dists, "loc"))
        self.assertTrue(hasattr(dists, "scale"))

    def test_deterministic_predictions_with_random_state(self):
        X, y = make_training_frame(periods=80)
        config = DistributionalModelConfig(
            dist_name="Normal",
            n_estimators=5,
            learning_rate=0.05,
            random_state=123,
        )
        first = DistributionalModel.from_config(config).fit(X, y).predict_mean(X.iloc[:10])
        second = DistributionalModel.from_config(config).fit(X, y).predict_mean(X.iloc[:10])

        np.testing.assert_allclose(first, second, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
