import unittest

import numpy as np

from src.evaluation import backtest_portfolio, calibration_summary, summarize_backtest


class MockDistribution:
    def __init__(self, loc: float, scale: float):
        self.loc = loc
        self.scale = scale

    def cdf(self, value):
        from math import erf, sqrt

        return 0.5 * (1.0 + erf((value - self.loc) / (self.scale * sqrt(2.0))))

    def ppf(self, quantile):
        from statistics import NormalDist

        return NormalDist(mu=self.loc, sigma=self.scale).inv_cdf(quantile)


class TestEvaluation(unittest.TestCase):
    def test_backtest_portfolio_summary(self):
        returns = np.array([0.02, -0.01, 0.03])
        positions = np.array([0.0, 1.0, 0.5])

        frame = backtest_portfolio(returns, positions, transaction_cost=0.001)
        self.assertEqual(list(frame.columns), ["base_return", "position", "transaction_cost", "portfolio_return", "equity_curve"])
        self.assertEqual(len(frame), 3)

        summary = summarize_backtest(returns, positions, transaction_cost=0.001)
        self.assertIn("adjusted_sharpe_ratio", summary)
        self.assertIn("turnover", summary)
        self.assertGreaterEqual(summary["average_leverage"], 0.0)

    def test_calibration_summary(self):
        distributions = [MockDistribution(0.0, 1.0), MockDistribution(0.5, 1.0)]
        observed = np.array([0.0, 0.25])

        summary = calibration_summary(distributions, observed)
        self.assertIn("pit_mean", summary)
        self.assertIn("coverage_0.05_0.95", summary)


if __name__ == "__main__":
    unittest.main()
