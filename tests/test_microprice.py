import unittest

import jax.numpy as jnp
import pandas as pd

from microprice import MicroPriceEstimator


class TestMicroPriceEstimator(unittest.TestCase):
    def set_up(self):
        self.estimator = MicroPriceEstimator("AAPL")

    def test_load_sample_data(self):
        df = self.estimator.load_sample_data()

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1000)

    def test_load(self):
        df = self.estimator.load_sample_data()

        self.estimator.load(df)
        self.assertIsNotNone(self.estimator.data)
        self.assertIsInstance(self.estimator.data, pd.DataFrame)

    def test_clean_dataset(self):
        df = self.estimator.load_sample_data()

        self.estimator.load(df)

        cleaned_df, tick_size = self.estimator._clean_dataset()

        self.assertIsInstance(cleaned_df, pd.DataFrame)
        self.assertIsInstance(tick_size, float)

    def test_compute_G(self):
        Q = jnp.array([[0.1, 0.2], [0.3, 0.4]])

        R1 = jnp.array([[0.5, 0.6], [0.7, 0.8]])
        R2 = jnp.array([[0.9, 1.0], [1.1, 1.2]])

        K = jnp.array([0.01, 0.02])

        n_imbalances = 2
        n_spread = 2
        iterations = 5

        G, B = self.estimator._compute_G(
            Q, R1, R2, K, n_imbalances, n_spread, iterations
        )

        self.assertIsInstance(G, jnp.ndarray)
        self.assertIsInstance(B, jnp.ndarray)
        self.assertEqual(G.shape, (2, 2))
        self.assertEqual(B.shape, (2, 2))

    def test_fit(self):
        df = self.estimator.load_sample_data()

        self.estimator.load(df)

        cleaned_df, _ = self.estimator._clean_dataset()

        G = self.estimator.fit(cleaned_df)

        self.assertIsInstance(G, jnp.ndarray)
        self.assertEqual(G.shape, (20, 20))


if __name__ == "__main__":
    unittest.main()
