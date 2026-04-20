from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import numpy as np

from .metrics import ReturnKind, downside_deviation
from .model import DistributionalModel


@dataclass(frozen=True)
class PositionSelection:
    position: float
    score: float
    mean_return: float
    downside_risk: float
    turnover_cost: float
    candidate_scores: dict[float, float]


def _coerce_return_kind(value: ReturnKind | str) -> ReturnKind:
    if isinstance(value, ReturnKind):
        return value
    return ReturnKind(value.lower())


def _to_simple_returns(samples: np.ndarray, return_kind: ReturnKind) -> np.ndarray:
    if return_kind is ReturnKind.LOG:
        return np.expm1(samples)
    return samples


def _sample_distribution(dist_obj: object, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2.")

    sample_method = getattr(dist_obj, "sample", None)
    if sample_method is None:
        raise AttributeError("Distribution object must expose a sample method.")

    seed = int(rng.integers(0, np.iinfo(np.uint32).max))
    try:
        signature = inspect.signature(sample_method)
        if "random_state" in signature.parameters:
            samples = sample_method(n_samples, random_state=seed)
            samples = np.asarray(samples, dtype=float).reshape(-1)
            if samples.size == 0:
                raise ValueError("Distribution returned no samples.")
            return samples
    except (TypeError, ValueError):
        pass

    legacy_state = np.random.get_state()
    np.random.seed(seed)
    try:
        samples = np.asarray(sample_method(n_samples), dtype=float).reshape(-1)
    finally:
        np.random.set_state(legacy_state)

    if samples.size == 0:
        raise ValueError("Distribution returned no samples.")
    return samples


def _normalize_grid(
    grid_points: np.ndarray | list[float],
    *,
    allow_shorting: bool,
) -> np.ndarray:
    grid = np.asarray(grid_points, dtype=float).reshape(-1)
    if grid.size == 0:
        raise ValueError("grid_points must contain at least one value.")
    if not np.isfinite(grid).all():
        raise ValueError("grid_points must be finite.")
    if not allow_shorting and np.any(grid < 0.0):
        raise ValueError("Negative grid points require allow_shorting=True.")
    return np.unique(np.sort(grid))


def position_score(
    base_returns: np.ndarray,
    *,
    position: float,
    previous_position: float = 0.0,
    transaction_cost: float = 0.0,
    leverage_penalty: float = 0.01,
    downside_penalty: float = 0.05,
    expected_return_weight: float = 1.0,
    return_kind: ReturnKind | str = ReturnKind.LOG,
) -> float:
    return_kind_enum = _coerce_return_kind(return_kind)
    simple_returns = _to_simple_returns(np.asarray(base_returns, dtype=float).reshape(-1), return_kind_enum)
    if simple_returns.size == 0:
        return float("-inf")

    turnover = abs(position - previous_position)
    portfolio_returns = (position * simple_returns) - (turnover * transaction_cost)
    if np.any(portfolio_returns <= -1.0):
        return float("-inf")

    mean_return = float(np.mean(portfolio_returns))
    downside_risk = downside_deviation(portfolio_returns)
    score = (
        expected_return_weight * mean_return
        - downside_penalty * downside_risk
        - leverage_penalty * abs(position)
    )
    return float(score)


def find_optimal_position(
    dist_obj: object,
    grid_points: np.ndarray | list[float] = np.linspace(0, 2, 21),
    n_samples: int = 1000,
    *,
    transaction_cost: float = 0.0,
    leverage_penalty: float = 0.01,
    downside_penalty: float = 0.05,
    expected_return_weight: float = 1.0,
    previous_position: float = 0.0,
    rng: np.random.Generator | None = None,
    return_kind: ReturnKind | str = ReturnKind.LOG,
    allow_shorting: bool = False,
    return_details: bool = False,
) -> float | PositionSelection:
    rng = rng or np.random.default_rng()
    grid = _normalize_grid(grid_points, allow_shorting=allow_shorting)
    samples = _sample_distribution(dist_obj, n_samples=n_samples, rng=rng)
    return_kind_enum = _coerce_return_kind(return_kind)

    candidate_scores: dict[float, float] = {}
    best_position = float(grid[0])
    best_score = float("-inf")
    for position in grid:
        score = position_score(
            samples,
            position=float(position),
            previous_position=previous_position,
            transaction_cost=transaction_cost,
            leverage_penalty=leverage_penalty,
            downside_penalty=downside_penalty,
            expected_return_weight=expected_return_weight,
            return_kind=return_kind_enum,
        )
        candidate_scores[float(position)] = float(score)
        if score > best_score + 1e-12:
            best_score = float(score)
            best_position = float(position)
            continue

        if not np.isclose(score, best_score, atol=1e-12):
            continue

        current_key = (abs(float(position) - previous_position), abs(float(position)), float(position))
        best_key = (abs(best_position - previous_position), abs(best_position), best_position)
        if current_key < best_key:
            best_position = float(position)
            best_score = float(score)

    if not return_details:
        return best_position

    return PositionSelection(
        position=best_position,
        score=best_score,
        mean_return=float(np.mean(_to_simple_returns(samples, return_kind_enum))),
        downside_risk=float(downside_deviation(_to_simple_returns(samples, return_kind_enum))),
        turnover_cost=float(abs(best_position - previous_position) * transaction_cost),
        candidate_scores=candidate_scores,
    )


class DistributionalStrategy:
    def __init__(
        self,
        model_params: dict | None = None,
        simulation_params: dict | None = None,
    ) -> None:
        self.model_params = model_params or {}
        self.simulation_params = simulation_params or {}
        self.model = DistributionalModel(**self.model_params)

    def fit(self, X_train: Any, y_train: Any) -> "DistributionalStrategy":
        self.model.fit(X_train, y_train)
        return self

    def _simulation_value(self, name: str, default: Any) -> Any:
        return self.simulation_params.get(name, default)

    def predict_positions(
        self,
        X: Any,
        *,
        initial_position: float = 0.0,
    ) -> np.ndarray:
        if len(X) == 0:
            return np.asarray([], dtype=float)

        distributions = self.model.predict_dist(X)
        grid_points = self._simulation_value("grid_points", np.linspace(0, 2, 21))
        n_samples = int(self._simulation_value("n_samples", 1000))
        transaction_cost = float(self._simulation_value("transaction_cost", 0.0))
        leverage_penalty = float(self._simulation_value("leverage_penalty", 0.01))
        downside_penalty = float(
            self._simulation_value(
                "downside_penalty",
                self._simulation_value("drawdown_penalty", 0.05),
            )
        )
        expected_return_weight = float(self._simulation_value("expected_return_weight", 1.0))
        allow_shorting = bool(self._simulation_value("allow_shorting", False))
        return_kind = self._simulation_value("return_kind", ReturnKind.LOG)
        random_state = self._simulation_value("random_state", None)
        rng = np.random.default_rng(random_state)

        positions: list[float] = []
        previous_position = float(initial_position)
        for index in range(len(X)):
            selection = find_optimal_position(
                distributions[index],
                grid_points=grid_points,
                n_samples=n_samples,
                transaction_cost=transaction_cost,
                leverage_penalty=leverage_penalty,
                downside_penalty=downside_penalty,
                expected_return_weight=expected_return_weight,
                previous_position=previous_position,
                rng=rng,
                return_kind=return_kind,
                allow_shorting=allow_shorting,
                return_details=True,
            )
            positions.append(float(selection.position))
            previous_position = float(selection.position)

        return np.asarray(positions, dtype=float)
