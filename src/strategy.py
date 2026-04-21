from __future__ import annotations

import numpy as np

from .metrics import adjusted_sharpe_ratio, max_drawdown
from .model import DistributionalModel


def _sample_distribution(dist_obj: object, n_samples: int) -> np.ndarray:
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2.")

    samples = np.asarray(dist_obj.sample(n_samples), dtype=float).reshape(-1)
    if samples.size == 0:
        raise ValueError("Distribution returned no samples.")
    return samples


def _coerce_grid_points(grid_points: np.ndarray | list[float]) -> np.ndarray:
    points = np.asarray(grid_points, dtype=float).reshape(-1)
    if points.size == 0:
        raise ValueError("grid_points must not be empty.")
    if not np.all(np.isfinite(points)):
        raise ValueError("grid_points must contain only finite values.")
    return points


def position_score(
    base_returns: np.ndarray,
    *,
    position: float,
    transaction_cost: float = 0.0,
    leverage_penalty: float = 0.01,
    drawdown_penalty: float = 0.05,
    expected_return_weight: float = 1.0,
) -> float:
    if position == 0.0:
        return 0.0

    portfolio_returns = (position * base_returns) - (abs(position) * transaction_cost)
    if np.any(portfolio_returns <= -1.0):
        return float("-inf")

    score = adjusted_sharpe_ratio(portfolio_returns)
    score += expected_return_weight * float(np.mean(portfolio_returns))
    score -= leverage_penalty * (position**2)
    score += drawdown_penalty * max_drawdown(portfolio_returns)
    return float(score)


def find_optimal_position(
    dist_obj: object,
    grid_points: np.ndarray | list[float] = np.linspace(0, 2, 21),
    n_samples: int = 1000,
    *,
    transaction_cost: float = 0.0,
    leverage_penalty: float = 0.01,
    drawdown_penalty: float = 0.05,
    expected_return_weight: float = 1.0,
) -> float:
    samples = _sample_distribution(dist_obj, n_samples=n_samples)
    grid = _coerce_grid_points(grid_points)

    best_position = 0.0
    best_score = float("-inf")
    for position in grid:
        score = position_score(
            samples,
            position=position,
            transaction_cost=transaction_cost,
            leverage_penalty=leverage_penalty,
            drawdown_penalty=drawdown_penalty,
            expected_return_weight=expected_return_weight,
        )
        if score > best_score + 1e-12:
            best_score = score
            best_position = float(position)
        elif np.isclose(score, best_score, atol=1e-12) and position < best_position:
            best_position = float(position)

    return best_position


class DistributionalStrategy:
    def __init__(self, model_params: dict | None = None, simulation_params: dict | None = None):
        self.model_params = model_params or {}
        self.simulation_params = simulation_params or {}
        self.model = DistributionalModel(**self.model_params)

    def fit(self, X_train, y_train) -> None:
        self.model.fit(X_train, y_train)

    def predict_positions(self, X) -> np.ndarray:
        if len(X) == 0:
            return np.asarray([], dtype=float)

        distributions = self.model.predict_dist(X)
        grid_points = self.simulation_params.get("grid_points", np.linspace(0, 2, 21))
        n_samples = int(self.simulation_params.get("n_samples", 1000))
        transaction_cost = float(self.simulation_params.get("transaction_cost", 0.0))
        leverage_penalty = float(self.simulation_params.get("leverage_penalty", 0.01))
        drawdown_penalty = float(self.simulation_params.get("drawdown_penalty", 0.05))
        expected_return_weight = float(self.simulation_params.get("expected_return_weight", 1.0))

        positions = [
            find_optimal_position(
                distributions[index],
                grid_points=grid_points,
                n_samples=n_samples,
                transaction_cost=transaction_cost,
                leverage_penalty=leverage_penalty,
                drawdown_penalty=drawdown_penalty,
                expected_return_weight=expected_return_weight,
            )
            for index in range(len(X))
        ]
        return np.asarray(positions, dtype=float)
