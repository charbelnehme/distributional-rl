from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from .metrics import adjusted_sharpe_ratio, max_drawdown, sharpe_ratio, sortino_ratio


def _coerce_1d(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def backtest_portfolio(
    returns: Sequence[float] | np.ndarray,
    positions: Sequence[float] | np.ndarray,
    *,
    transaction_cost: float = 0.0,
) -> pd.DataFrame:
    if transaction_cost < 0:
        raise ValueError("transaction_cost must be >= 0.")

    base_returns = _coerce_1d(returns, "returns")
    position_vector = _coerce_1d(positions, "positions")
    if base_returns.size != position_vector.size:
        raise ValueError("returns and positions must have the same length.")

    position_changes = np.abs(np.diff(np.concatenate(([0.0], position_vector))))
    transaction_costs = position_changes * transaction_cost
    portfolio_returns = (position_vector * base_returns) - transaction_costs
    equity_curve = np.cumprod(1.0 + portfolio_returns)

    return pd.DataFrame(
        {
            "base_return": base_returns,
            "position": position_vector,
            "transaction_cost": transaction_costs,
            "portfolio_return": portfolio_returns,
            "equity_curve": equity_curve,
        }
    )


def summarize_backtest(
    returns: Sequence[float] | np.ndarray,
    positions: Sequence[float] | np.ndarray,
    *,
    transaction_cost: float = 0.0,
) -> dict[str, float]:
    frame = backtest_portfolio(returns, positions, transaction_cost=transaction_cost)
    portfolio_returns = frame["portfolio_return"].to_numpy(dtype=float)
    position_vector = frame["position"].to_numpy(dtype=float)

    return {
        "mean_return": float(np.mean(portfolio_returns)),
        "sharpe_ratio": float(sharpe_ratio(portfolio_returns)),
        "sortino_ratio": float(sortino_ratio(portfolio_returns)),
        "adjusted_sharpe_ratio": float(adjusted_sharpe_ratio(portfolio_returns)),
        "max_drawdown": float(max_drawdown(portfolio_returns)),
        "turnover": float(np.mean(np.abs(np.diff(np.concatenate(([0.0], position_vector)))))),
        "average_leverage": float(np.mean(np.abs(position_vector))),
    }


def pit_values(distributions: Iterable[object], observed_returns: Sequence[float] | np.ndarray) -> np.ndarray:
    observations = _coerce_1d(observed_returns, "observed_returns")
    distributions = list(distributions)
    if len(distributions) != observations.size:
        raise ValueError("distributions and observed_returns must have the same length.")

    values: list[float] = []
    for dist, observed in zip(distributions, observations, strict=True):
        if not hasattr(dist, "cdf"):
            raise TypeError("Each distribution must expose a cdf() method.")
        values.append(float(dist.cdf(observed)))
    return np.asarray(values, dtype=float)


def calibration_summary(
    distributions: Iterable[object],
    observed_returns: Sequence[float] | np.ndarray,
    *,
    quantile_pairs: Sequence[tuple[float, float]] = ((0.05, 0.95), (0.10, 0.90)),
) -> dict[str, float]:
    observations = _coerce_1d(observed_returns, "observed_returns")
    distributions = list(distributions)
    if len(distributions) != observations.size:
        raise ValueError("distributions and observed_returns must have the same length.")

    pit = pit_values(distributions, observations)
    summary: dict[str, float] = {
        "pit_mean": float(np.mean(pit)),
        "pit_std": float(np.std(pit, ddof=0)),
    }

    for lower_q, upper_q in quantile_pairs:
        if not (0.0 < lower_q < upper_q < 1.0):
            raise ValueError("quantile_pairs must contain 0 < lower < upper < 1.")
        lower_bounds = np.asarray([dist.ppf(lower_q) for dist in distributions], dtype=float)
        upper_bounds = np.asarray([dist.ppf(upper_q) for dist in distributions], dtype=float)
        coverage = np.mean((observations >= lower_bounds) & (observations <= upper_bounds))
        width = np.mean(upper_bounds - lower_bounds)
        summary[f"coverage_{lower_q:.2f}_{upper_q:.2f}"] = float(coverage)
        summary[f"interval_width_{lower_q:.2f}_{upper_q:.2f}"] = float(width)

    return summary
