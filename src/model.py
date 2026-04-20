from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from sklearn.utils.validation import check_X_y, check_array


@dataclass(frozen=True)
class DistributionalModelConfig:
    dist_name: str = "Normal"
    n_estimators: int = 100
    learning_rate: float = 0.01
    random_state: int | None = None
    verbose: bool = False


@lru_cache(maxsize=1)
def _import_ngboost() -> tuple[Any, Any, Any]:
    try:
        from ngboost import NGBRegressor
        from ngboost.distns import Normal, T
        from ngboost.scores import LogScore
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "src.model requires the optional dependency set for model training. "
            "Install the project dependencies with `pip install -e .` or "
            "`pip install -e .[test]`."
        ) from exc

    return NGBRegressor, Normal, T, LogScore


class DistributionalModel:
    def __init__(
        self,
        dist_name: str = "Normal",
        n_estimators: int = 100,
        learning_rate: float = 0.01,
        *,
        random_state: int | None = None,
        verbose: bool = False,
        config: DistributionalModelConfig | None = None,
    ) -> None:
        self.config = config or DistributionalModelConfig(
            dist_name=dist_name,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=verbose,
        )
        self._model: Any | None = None
        self._is_fitted = False
        self._n_features_in_: int | None = None

    @staticmethod
    def _distribution_registry() -> dict[str, str]:
        return {
            "Normal": "Normal",
            "StudentT": "T",
            "T": "T",
        }

    @classmethod
    def from_config(cls, config: DistributionalModelConfig) -> "DistributionalModel":
        return cls(config=config)

    def _resolve_distribution(self) -> Any:
        _, Normal, T, _ = _import_ngboost()
        registry = {
            "Normal": Normal,
            "StudentT": T,
            "T": T,
        }
        try:
            return registry[self.config.dist_name]
        except KeyError as exc:
            supported = ", ".join(sorted(registry))
            raise ValueError(
                f"Unsupported distribution '{self.config.dist_name}'. Supported values: {supported}."
            ) from exc

    def _ensure_fitted(self) -> Any:
        if not self._is_fitted or self._model is None:
            raise RuntimeError("DistributionalModel must be fitted before prediction.")
        return self._model

    def _validate_features(self, X: Any, *, allow_empty: bool = False) -> np.ndarray:
        array = check_array(X, ensure_2d=True, ensure_all_finite=True, dtype=float)
        if array.ndim != 2:
            raise ValueError("X must be two-dimensional.")
        if not allow_empty and array.shape[0] == 0:
            raise ValueError("X must contain at least one row.")
        if self._is_fitted and self._n_features_in_ is not None and array.shape[1] != self._n_features_in_:
            raise ValueError(
                f"X has {array.shape[1]} features, but the model was fitted with "
                f"{self._n_features_in_} features."
            )
        return array

    def fit(self, X: Any, y: Any) -> "DistributionalModel":
        X_arr, y_arr = check_X_y(
            X,
            y,
            ensure_2d=True,
            ensure_all_finite=True,
            dtype=float,
            y_numeric=True,
        )
        if X_arr.shape[0] < 2:
            raise ValueError("At least two training rows are required.")

        NGBRegressor, _, _, LogScore = _import_ngboost()
        dist = self._resolve_distribution()
        kwargs: dict[str, Any] = {
            "Dist": dist,
            "Score": LogScore,
            "n_estimators": self.config.n_estimators,
            "learning_rate": self.config.learning_rate,
            "verbose": self.config.verbose,
        }
        try:
            signature = inspect.signature(NGBRegressor)
            if "random_state" in signature.parameters:
                kwargs["random_state"] = self.config.random_state
        except (TypeError, ValueError):  # pragma: no cover - defensive
            kwargs["random_state"] = self.config.random_state

        self._model = NGBRegressor(**kwargs)
        self._model.fit(X_arr, y_arr)
        self._is_fitted = True
        self._n_features_in_ = X_arr.shape[1]
        return self

    def predict_dist(self, X: Any) -> Any:
        model = self._ensure_fitted()
        X_arr = self._validate_features(X)
        return model.pred_dist(X_arr)

    def predict_mean(self, X: Any) -> np.ndarray:
        model = self._ensure_fitted()
        X_arr = self._validate_features(X)
        if hasattr(model, "predict"):
            return np.asarray(model.predict(X_arr), dtype=float)
        dist = model.pred_dist(X_arr)
        return np.asarray(getattr(dist, "loc"), dtype=float)

    def predict_scale(self, X: Any) -> np.ndarray:
        dist = self.predict_dist(X)
        scale = getattr(dist, "scale", None)
        if scale is None:
            raise AttributeError("Predicted distribution does not expose a scale parameter.")
        return np.asarray(scale, dtype=float)

    def predict_std(self, X: Any) -> np.ndarray:
        return self.predict_scale(X)

    def sample(
        self,
        X: Any,
        n_samples: int = 1,
        *,
        random_state: int | None = None,
    ) -> np.ndarray:
        if n_samples < 1:
            raise ValueError("n_samples must be positive.")
        dist = self.predict_dist(X)
        if random_state is not None:
            rng = np.random.default_rng(random_state)
            sample_method = getattr(dist, "sample")
            try:
                signature = inspect.signature(sample_method)
                if "random_state" in signature.parameters:
                    return np.asarray(sample_method(n_samples, random_state=int(random_state)), dtype=float)
            except (TypeError, ValueError):
                pass

            legacy_state = np.random.get_state()
            np.random.seed(int(rng.integers(0, np.iinfo(np.uint32).max)))
            try:
                return np.asarray(sample_method(n_samples), dtype=float)
            finally:
                np.random.set_state(legacy_state)

        return np.asarray(dist.sample(n_samples), dtype=float)
