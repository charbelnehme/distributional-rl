try:
    from ngboost import NGBRegressor
    from ngboost.distns import Normal, T
    from ngboost.scores import LogScore
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "distributional_rl.model requires the optional dependency set for model "
        "training. Install the project dependencies with `pip install -e .` "
        "or `pip install -e .[test]`."
    ) from exc

class DistributionalModel:
    def __init__(self, dist_name='Normal', n_estimators=100, learning_rate=0.01):
        self.dist_name = dist_name
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
        
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict_dist(self, X):
        """
        Returns the predicted distribution object(s) for the input X.
        """
        return self.model.pred_dist(X)
