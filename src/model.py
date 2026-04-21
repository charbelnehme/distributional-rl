from __future__ import annotations

import numpy as np

try:
    from ngboost import NGBRegressor
    from ngboost.distns import Normal, T
    from ngboost.scores import LogScore
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "src.model requires the optional dependency set for model "
        "training. Install the project dependencies with `pip install -e .` "
        "or `pip install -e .[test]`."
    ) from exc


class _ScaledDistribution:
    def __init__(self, dist: object, mean: float, scale: float) -> None:
        self._dist = dist
        self._mean = mean
        self._scale = scale

    def sample(self, n_samples: int):
        samples = np.asarray(self._dist.sample(n_samples), dtype=float)
        return (samples * self._scale) + self._mean

    def __getitem__(self, index: int):
        return _ScaledDistribution(self._dist[index], self._mean, self._scale)

    def __len__(self) -> int:
        return len(self._dist)

    def __getattr__(self, name: str):
        return getattr(self._dist, name)

class DistributionalModel:
    def __init__(self, dist_name='Normal', n_estimators=100, learning_rate=0.01):
        self.dist_name = dist_name
        self._target_mean = 0.0
        self._target_scale = 1.0
        if dist_name == 'Normal':
            self.dist = Normal
        elif dist_name == 'StudentT':
            self.dist = T
        else:
            raise ValueError(f"Unsupported distribution: {dist_name}")
            
        self.model = NGBRegressor(
            Dist=self.dist,
            Score=LogScore,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            verbose=False
        )

    def _scale_target(self, y):
        target = np.asarray(y, dtype=float).reshape(-1)
        if not np.all(np.isfinite(target)):
            raise ValueError("Target values must be finite.")

        self._target_mean = float(np.mean(target))
        self._target_scale = float(np.std(target, ddof=0))
        if not np.isfinite(self._target_scale) or self._target_scale == 0.0:
            self._target_scale = 1.0

        return (target - self._target_mean) / self._target_scale
        
    def fit(self, X, y):
        scaled_y = self._scale_target(y)
        self.model.fit(X, scaled_y)
        
    def predict_dist(self, X):
        """
        Returns the predicted distribution object(s) for the input X.
        """
        return _ScaledDistribution(self.model.pred_dist(X), self._target_mean, self._target_scale)
