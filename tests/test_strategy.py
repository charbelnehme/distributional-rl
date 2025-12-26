import unittest
import numpy as np
from distributional_rl.strategy import DistributionalStrategy, find_optimal_position
from distributional_rl.data import generate_synthetic_data

class MockDist:
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        
    def sample(self, n):
        return np.random.normal(self.loc, self.scale, n)

class TestStrategy(unittest.TestCase):
    def test_find_optimal_position(self):
        # Case 1: Positive return, low vol -> should take max position
        dist = MockDist(0.05, 0.01)
        p = find_optimal_position(dist, grid_points=[0, 1, 2], n_samples=1000)
        self.assertEqual(p, 2)
        
        # Case 2: Negative return -> should take 0 position
        dist = MockDist(-0.05, 0.01)
        p = find_optimal_position(dist, grid_points=[0, 1, 2], n_samples=1000)
        self.assertEqual(p, 0)

    def test_strategy_end_to_end(self):
        X, y = generate_synthetic_data(50)
        strategy = DistributionalStrategy(
            model_params={'n_estimators': 5},
            simulation_params={'n_samples': 100, 'grid_points': [0, 0.5, 1.0]}
        )
        strategy.fit(X, y)
        positions = strategy.predict_positions(X.iloc[:5])
        
        self.assertEqual(len(positions), 5)
        self.assertTrue(all(p in [0, 0.5, 1.0] for p in positions))

if __name__ == '__main__':
    unittest.main()
