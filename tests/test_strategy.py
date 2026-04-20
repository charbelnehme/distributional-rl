from __future__ import annotations

import unittest

import numpy as np

from src.strategy import DistributionalStrategy, find_optimal_position
from tests.fixtures import make_training_frame


class MockDist:
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def sample(self, n):
        return np.random.normal(self.loc, self.scale, n)


class TestStrategy(unittest.TestCase):
    def test_find_optimal_position(self):
        positive_dist = MockDist(0.05, 0.01)
        positive_position = find_optimal_position(
            positive_dist,
            grid_points=[0, 1, 2],
            n_samples=2000,
            leverage_penalty=0.01,
            downside_penalty=0.0,
        )
        self.assertEqual(positive_position, 2.0)

    def test_shorting_behavior_when_enabled(self):
        negative_dist = MockDist(-0.05, 0.01)
        negative_position = find_optimal_position(
            negative_dist,
            grid_points=[-1, 0, 1],
            n_samples=2000,
            allow_shorting=True,
            downside_penalty=0.0,
        )
        self.assertEqual(negative_position, -1.0)

    def test_turnover_cost_prefers_staying_put(self):
        flat_dist = MockDist(0.0, 0.0001)
        position = find_optimal_position(
            flat_dist,
            grid_points=[0, 1],
            n_samples=2000,
            previous_position=1.0,
            transaction_cost=0.01,
            leverage_penalty=0.0,
            downside_penalty=0.0,
        )
        self.assertEqual(position, 1.0)

    def test_deterministic_sampling_behavior(self):
        dist = MockDist(0.02, 0.01)
        first = find_optimal_position(
            dist,
            grid_points=[0, 1, 2],
            n_samples=1000,
            rng=np.random.default_rng(1234),
        )
        second = find_optimal_position(
            dist,
            grid_points=[0, 1, 2],
            n_samples=1000,
            rng=np.random.default_rng(1234),
        )
        self.assertEqual(first, second)

    def test_invalid_grid_handling(self):
        dist = MockDist(0.01, 0.01)
        with self.assertRaises(ValueError):
            find_optimal_position(dist, grid_points=[], n_samples=1000)
        with self.assertRaises(ValueError):
            find_optimal_position(dist, grid_points=[-1, 0, 1], n_samples=1000)

    def test_strategy_end_to_end(self):
        X, y = make_training_frame(periods=80)
        strategy = DistributionalStrategy(
            model_params={"n_estimators": 5, "random_state": 42},
            simulation_params={
                "n_samples": 100,
                "grid_points": [0, 0.5, 1.0],
                "leverage_penalty": 0.01,
                "downside_penalty": 0.0,
                "random_state": 123,
            },
        )
        strategy.fit(X, y)
        positions = strategy.predict_positions(X.iloc[:5])

        self.assertEqual(len(positions), 5)
        self.assertTrue(all(position in [0.0, 0.5, 1.0] for position in positions))


if __name__ == "__main__":
    unittest.main()
