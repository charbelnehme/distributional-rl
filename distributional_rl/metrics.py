from __future__ import annotations

import numpy as np
import scipy.stats as stats


def _coerce_returns(returns: np.ndarray | list[float]) -> np.ndarray:
    values = np.asarray(returns, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    return values


def sharpe_ratio(returns: np.ndarray | list[float], risk_free_rate: float = 0.0) -> float:
    values = _coerce_returns(returns)
    if values.size < 2:
        return 0.0

    excess_returns = values - risk_free_rate
    volatility = np.std(excess_returns, ddof=0)
    if np.isclose(volatility, 0.0):
        return 0.0
    return float(np.mean(excess_returns) / volatility)


def downside_deviation(
    returns: np.ndarray | list[float],
    target_return: float = 0.0,
) -> float:
    values = _coerce_returns(returns)
    if values.size == 0:
        return 0.0

    downside = np.minimum(values - target_return, 0.0)
    return float(np.sqrt(np.mean(np.square(downside))))


def sortino_ratio(
    returns: np.ndarray | list[float],
    target_return: float = 0.0,
) -> float:
    values = _coerce_returns(returns)
    if values.size < 2:
        return 0.0

    dd = downside_deviation(values, target_return=target_return)
    if np.isclose(dd, 0.0):
        return 0.0
    return float((np.mean(values) - target_return) / dd)


def max_drawdown(returns: np.ndarray | list[float]) -> float:
    values = _coerce_returns(returns)
    if values.size == 0:
        return 0.0

    equity_curve = np.cumprod(1.0 + values)
    running_peak = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve / running_peak) - 1.0
    return float(drawdowns.min())


def adjusted_sharpe_ratio(returns: np.ndarray | list[float]) -> float:
    """
    Pezier-White adjusted Sharpe ratio:

        ASR = SR * (1 + (S / 6) * SR - (K_excess / 24) * SR^2)
    """

    values = _coerce_returns(returns)
    if values.size < 2:
        return 0.0

    sr = sharpe_ratio(values)
    if np.isclose(sr, 0.0):
        return 0.0

    skew = float(stats.skew(values, bias=False))
    excess_kurtosis = float(stats.kurtosis(values, fisher=True, bias=False))
    if not np.isfinite(skew):
        skew = 0.0
    if not np.isfinite(excess_kurtosis):
        excess_kurtosis = 0.0

    adjustment = 1.0 + (skew / 6.0) * sr - (excess_kurtosis / 24.0) * (sr**2)
    return float(sr * adjustment)
