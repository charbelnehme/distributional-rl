import unittest

from src.model import DistributionalModel
from tests.fixtures import make_training_frame


class TestModel(unittest.TestCase):
    def test_rejects_invalid_hyperparameters(self):
        with self.assertRaises(ValueError):
            DistributionalModel(n_estimators=0)

        with self.assertRaises(ValueError):
            DistributionalModel(learning_rate=0)

    def test_predict_requires_fit(self):
        model = DistributionalModel(n_estimators=10)

        with self.assertRaises(RuntimeError):
            model.predict_dist(make_training_frame(periods=10)[0])

    def test_fit_predict(self):
        X, y = make_training_frame(periods=80)
        model = DistributionalModel(n_estimators=10, dist_name="Normal")
        returned = model.fit(X, y)

        self.assertIs(returned, model)

        dists = model.predict_dist(X)
        sample = dists.sample(1)
        self.assertEqual(sample.shape, (1, len(X)))
        self.assertTrue(hasattr(dists, "loc"))
        self.assertTrue(hasattr(dists, "scale"))

    def test_distribution_aliases_are_case_insensitive(self):
        model = DistributionalModel(dist_name="student_t", n_estimators=10)
        self.assertEqual(model.dist_name, "student_t")


if __name__ == "__main__":
    unittest.main()
