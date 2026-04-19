import unittest

from distributional_rl.model import DistributionalModel
from tests.fixtures import make_training_frame


class TestModel(unittest.TestCase):
    def test_fit_predict(self):
        X, y = make_training_frame(periods=80)
        model = DistributionalModel(n_estimators=10, dist_name="Normal")
        model.fit(X, y)

        dists = model.predict_dist(X)
        sample = dists.sample(1)
        self.assertEqual(sample.shape, (1, len(X)))
        self.assertTrue(hasattr(dists, "loc"))
        self.assertTrue(hasattr(dists, "scale"))


if __name__ == "__main__":
    unittest.main()
