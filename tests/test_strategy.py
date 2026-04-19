import unittest

import numpy as np

from src.strategy import DistributionalStrategy, find_optimal_position
from tests.fixtures import make_training_frame


class MockDist:
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def sample(self, n):
        rng = np.random.default_rng(42)
        return rng.normal(self.loc, self.scale, n)


class TestStrategy(unittest.TestCase):
    def test_find_optimal_position(self):
        positive_dist = MockDist(0.05, 0.01)
        positive_position = find_optimal_position(
            positive_dist,
            grid_points=[0, 1, 2],
            n_samples=2000,
            leverage_penalty=0.01,
            drawdown_penalty=0.0,
        )
        self.assertEqual(positive_position, 2.0)

        negative_dist = MockDist(-0.05, 0.01)
        negative_position = find_optimal_position(
            negative_dist,
            grid_points=[0, 1, 2],
            n_samples=2000,
            leverage_penalty=0.01,
            drawdown_penalty=0.0,
        )
        self.assertEqual(negative_position, 0.0)

    def test_strategy_end_to_end(self):
        X, y = make_training_frame(periods=80)
        strategy = DistributionalStrategy(
            model_params={"n_estimators": 5},
            simulation_params={
                "n_samples": 100,
                "grid_points": [0, 0.5, 1.0],
                "leverage_penalty": 0.01,
            },
        )
        strategy.fit(X, y)
        positions = strategy.predict_positions(X.iloc[:5])

        self.assertEqual(len(positions), 5)
        self.assertTrue(all(position in [0.0, 0.5, 1.0] for position in positions))


if __name__ == "__main__":
    unittest.main()
