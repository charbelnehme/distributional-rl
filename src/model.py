from __future__ import annotations

from typing import Any

try:
    from ngboost import NGBRegressor
    from ngboost.distns import Normal, T
    from ngboost.scores import LogScore
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "src.model requires the optional dependency set for model training. Install "
        "the project dependencies with `pip install -e .` or `pip install -e .[test]`."
    ) from exc


_DISTRIBUTION_ALIASES: dict[str, Any] = {
    "normal": Normal,
    "studentt": T,
    "student_t": T,
    "t": T,
}


class DistributionalModel:
    """Small NGBoost wrapper used by the strategy layer."""

    def __init__(
        self,
        dist_name: str = "Normal",
        n_estimators: int = 100,
        learning_rate: float = 0.01,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")

        self.dist_name = dist_name
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.dist = self._resolve_distribution(dist_name)
        self.model = NGBRegressor(
            Dist=self.dist,
            Score=LogScore,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            verbose=False,
        )
        self._fitted = False

    @staticmethod
    def _resolve_distribution(dist_name: str) -> Any:
        normalized = dist_name.strip().replace(" ", "").lower()
        try:
            return _DISTRIBUTION_ALIASES[normalized]
        except KeyError as exc:
            supported = ", ".join(sorted({"Normal", "StudentT"}))
            raise ValueError(
                f"Unsupported distribution: {dist_name}. Supported values: {supported}."
            ) from exc

    def fit(self, X: Any, y: Any) -> "DistributionalModel":
        self.model.fit(X, y)
        self._fitted = True
        return self

    def predict_dist(self, X: Any) -> Any:
        """Return the NGBoost predictive distribution for each input row."""

        if not self._fitted:
            raise RuntimeError("DistributionalModel must be fit before calling predict_dist().")
        return self.model.pred_dist(X)
