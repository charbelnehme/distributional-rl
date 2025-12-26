import unittest
import numpy as np
from distributional_rl.model import DistributionalModel
from distributional_rl.data import generate_synthetic_data

class TestModel(unittest.TestCase):
    def test_fit_predict(self):
        X, y = generate_synthetic_data(100)
        model = DistributionalModel(n_estimators=10, dist_name='Normal')
        model.fit(X, y)
        
        dists = model.predict_dist(X)
        # NGBoost pred_dist returns a Distribution object
        # We can sample from it
        sample = dists.sample(1)
        # Sample shape is (1, 100) based on quick check
        self.assertEqual(sample.shape, (1, 100))
        
        # Check params
        # Normal distribution has loc and scale
        self.assertTrue(hasattr(dists, 'loc'))
        self.assertTrue(hasattr(dists, 'scale'))

if __name__ == '__main__':
    unittest.main()
