from __future__ import annotations

import unittest

import numpy as np

from src.metrics import (
    ReturnKind,
    adjusted_sharpe_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)


class TestMetrics(unittest.TestCase):
    def test_log_return_drawdown(self):
        returns = np.log(np.array([1.10, 0.90, 1.20]) / np.array([1.0, 1.0, 1.0]))
        self.assertAlmostEqual(
            max_drawdown(returns, return_kind=ReturnKind.LOG),
            -0.1,
            places=12,
        )

    def test_arithmetic_return_drawdown(self):
        returns = np.array([0.10, -0.20, 0.05])
        self.assertAlmostEqual(max_drawdown(returns, return_kind="simple"), -0.2, places=12)

    def test_empty_input(self):
        self.assertEqual(sharpe_ratio([]), 0.0)
        self.assertEqual(sortino_ratio([]), 0.0)
        self.assertEqual(max_drawdown([], return_kind="simple"), 0.0)
        self.assertEqual(adjusted_sharpe_ratio([]), 0.0)

    def test_one_point_input(self):
        returns = np.array([0.02])
        self.assertEqual(sharpe_ratio(returns), 0.0)
        self.assertEqual(sortino_ratio(returns), 0.0)
        self.assertEqual(adjusted_sharpe_ratio(returns), 0.0)

    def test_near_zero_variance_input(self):
        returns = np.array([0.01, 0.0100000001, 0.0099999999, 0.01000000005])
        self.assertEqual(sharpe_ratio(returns), 0.0)
        self.assertEqual(adjusted_sharpe_ratio(returns), 0.0)

    def test_non_finite_input(self):
        with self.assertRaises(ValueError):
            sharpe_ratio([0.01, np.nan])
        with self.assertRaises(ValueError):
            max_drawdown([0.01, np.inf], return_kind="simple")

    def test_small_sample_adjusted_sharpe_fallback(self):
        returns = np.array([0.02, 0.03, 0.01])
        self.assertAlmostEqual(
            adjusted_sharpe_ratio(returns),
            sharpe_ratio(returns),
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
