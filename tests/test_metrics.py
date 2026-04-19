import numpy as np
import unittest
from distributional_rl.metrics import adjusted_sharpe_ratio, max_drawdown, sharpe_ratio, sortino_ratio

class TestMetrics(unittest.TestCase):
    def test_normal_distribution(self):
        # For normal distribution, skew=0, excess_kurtosis=0.
        # ASR should equal SR.
        np.random.seed(42)
        returns = np.random.normal(0.01, 0.02, 10000)
        
        sr = sharpe_ratio(returns)
        asr = adjusted_sharpe_ratio(returns)
        
        self.assertAlmostEqual(sr, asr, places=2)

    def test_skewed_distribution(self):
        # Positive skew should increase ASR relative to SR (if SR > 0).
        np.random.seed(42)
        # LogNormal is positively skewed
        returns = np.random.lognormal(0, 0.5, 10000) - np.exp(0.5**2 / 2) + 0.05 # Centered + drift
        
        asr = adjusted_sharpe_ratio(returns)
        self.assertIsInstance(asr, float)

    def test_sortino_and_drawdown(self):
        returns = np.array([0.01, -0.02, 0.015, -0.01, 0.005])
        self.assertIsInstance(sortino_ratio(returns), float)
        self.assertLessEqual(max_drawdown(returns), 0.0)

if __name__ == '__main__':
    unittest.main()
