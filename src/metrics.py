from __future__ import annotations

from enum import Enum
from typing import Sequence

import numpy as np
import scipy.stats as stats


class ReturnKind(str, Enum):
    SIMPLE = "simple"
    LOG = "log"


def _coerce_return_kind(value: ReturnKind | str) -> ReturnKind:
    if isinstance(value, ReturnKind):
        return value
    try:
        return ReturnKind(value.lower())
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported return kind '{value}'.") from exc


def _coerce_returns(returns: np.ndarray | Sequence[float]) -> np.ndarray:
    values = np.asarray(returns, dtype=float).reshape(-1)
    if values.size == 0:
        return values
    if not np.isfinite(values).all():
        raise ValueError("Returns must be finite.")
    return values


def _annualization_multiplier(periods_per_year: float | None) -> float:
    if periods_per_year is None:
        return 1.0
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")
    return float(np.sqrt(periods_per_year))


def _excess_growth_rates(values: np.ndarray, return_kind: ReturnKind) -> np.ndarray:
    if return_kind is ReturnKind.SIMPLE:
        if np.any(values <= -1.0):
            raise ValueError("Simple returns must be greater than -1.0.")
        return np.log1p(values)
    return values


def sharpe_ratio(
    returns: np.ndarray | Sequence[float],
    risk_free_rate: float = 0.0,
    *,
    periods_per_year: float | None = None,
) -> float:
    values = _coerce_returns(returns)
    if values.size < 2:
        return 0.0

    excess_returns = values - risk_free_rate
    volatility = float(np.std(excess_returns, ddof=0))
    if np.isclose(volatility, 0.0):
        return 0.0

    ratio = float(np.mean(excess_returns) / volatility)
    return ratio * _annualization_multiplier(periods_per_year)


def downside_deviation(
    returns: np.ndarray | Sequence[float],
    target_return: float = 0.0,
) -> float:
    values = _coerce_returns(returns)
    if values.size == 0:
        return 0.0

    downside = np.minimum(values - target_return, 0.0)
    return float(np.sqrt(np.mean(np.square(downside))))


def sortino_ratio(
    returns: np.ndarray | Sequence[float],
    target_return: float = 0.0,
    *,
    periods_per_year: float | None = None,
) -> float:
    values = _coerce_returns(returns)
    if values.size < 2:
        return 0.0

    dd = downside_deviation(values, target_return=target_return)
    if np.isclose(dd, 0.0):
        return 0.0

    ratio = float((np.mean(values) - target_return) / dd)
    return ratio * _annualization_multiplier(periods_per_year)


def max_drawdown(
    returns: np.ndarray | Sequence[float],
    *,
    return_kind: ReturnKind | str = ReturnKind.SIMPLE,
) -> float:
    values = _coerce_returns(returns)
    if values.size == 0:
        return 0.0

    kind = _coerce_return_kind(return_kind)
    growth_rates = _excess_growth_rates(values, kind)
    log_equity = np.cumsum(growth_rates)
    running_peak = np.maximum.accumulate(log_equity)
    drawdowns = np.expm1(log_equity - running_peak)
    return float(drawdowns.min(initial=0.0))


def adjusted_sharpe_ratio(
    returns: np.ndarray | Sequence[float],
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: float | None = None,
) -> float:
    """
    Pezier-White adjusted Sharpe ratio.

    The adjustment is only applied when the sample is large enough for stable
    moment estimates. For small or near-degenerate samples, the function falls
    back to the plain Sharpe ratio rather than amplifying noise with an unstable
    skew/kurtosis correction.
    """

    values = _coerce_returns(returns)
    if values.size < 2:
        return 0.0

    sr = sharpe_ratio(
        values,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )
    if np.isclose(sr, 0.0):
        return 0.0
    if values.size < 4:
        return float(sr)

    skew = float(stats.skew(values, bias=False))
    excess_kurtosis = float(stats.kurtosis(values, fisher=True, bias=False))
    if not np.isfinite(skew) or not np.isfinite(excess_kurtosis):
        return float(sr)

    adjustment = 1.0 + (skew / 6.0) * sr - (excess_kurtosis / 24.0) * (sr**2)
    if not np.isfinite(adjustment):
        return float(sr)
    return float(sr * adjustment)
