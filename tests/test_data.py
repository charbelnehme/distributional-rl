import unittest
import pandas as pd
from distributional_rl.data import generate_synthetic_data

class TestData(unittest.TestCase):
    def test_generation(self):
        X, y = generate_synthetic_data(100)
        self.assertEqual(len(X), 100)
        self.assertEqual(len(y), 100)
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertIn('momentum', X.columns)

if __name__ == '__main__':
    unittest.main()
