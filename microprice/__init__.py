import random
from typing import Tuple

import jax
import jax.numpy as jnp
import pandas as pd
from jax import jit


class MicroPriceEstimator:
    def __init__(
        self, symbol: str, n_imbalances: int = 10, n_spread: int = 2, dt: int = 1
    ) -> None:
        self.symbol = symbol
        self.n_imbalances = n_imbalances
        self.n_spread = n_spread
        self.dt = dt
        self.data = None

    def load_sample_data(self, n_rows: int = 1000) -> pd.DataFrame:
        """
        Load sample data into a pandas DataFrame.

        Parameters:
        - n_rows (int): The number of rows to generate in the DataFrame. Default is 1000.

        Returns:
        - df (pd.DataFrame): The generated DataFrame containing sample data.
        """
        key = random.PRNGKey(0)
        date = jnp.arange(n_rows)
        timestamp = jnp.arange(n_rows) * 1000
        bid_price = random.normal(key, (n_rows,)) * 10 + 100
        ask_price = bid_price + random.uniform(key, (n_rows,)) * 0.5
        bid_size = random.poisson(key, 100, (n_rows,))
        ask_size = random.poisson(key, 100, (n_rows,))

        df = pd.DataFrame(
            {
                "date": date,
                "timestamp": timestamp,
                "bid_price": bid_price,
                "ask_price": ask_price,
                "bid_size": bid_size,
                "ask_size": ask_size,
            }
        )
        return df

    def load(self, df: pd.DataFrame) -> None:
        """
        Load the given DataFrame into the MicroPrice object.

        Args:
            df (pd.DataFrame): The DataFrame containing the data to be loaded.

        Returns:
            None
        """
        self.data = df.dropna()

        float_columns = [
            "date",
            "timestamp",
            "bid_price",
            "ask_price",
            "bid_size",
            "ask_size",
        ]
        self.data[float_columns] = self.data[float_columns].astype(float)

        self.data["mid_price"] = (self.data["bid_price"] + self.data["ask_price"]) / 2
        self.data["imbalance"] = self.data["bid_size"] / (
            self.data["bid_size"] + self.data["ask_size"]
        )
        self.data["weighted_mid"] = self.data["ask_price"] * self.data[
            "imbalance"
        ] + self.data["bid_price"] * (1 - self.data["imbalance"])

        self._clean_dataset()

    def _clean_dataset(self) -> Tuple[pd.DataFrame, float]:
        T = self.data
        spread = T["ask_price"] - T["bid_price"]
        tick_size = jnp.round(jnp.min(spread[spread > 0]) * 100) / 100

        T["spread"] = (
            jnp.round((T["ask_price"] - T["bid_price"]) / tick_size) * tick_size
        )
        T["mid_price"] = (T["bid_price"] + T["ask_price"]) / 2

        T = T.loc[(T.spread <= self.n_spread * tick_size) & (T.spread > 0)]

        T["imbalance"] = T["bid_size"] / (T["bid_size"] + T["ask_size"])
        T["imbalance_bucket"] = pd.qcut(T["imbalance"], self.n_imbalances, labels=False)

        shift_columns = ["mid_price", "spread", "timestamp", "imbalance_bucket"]

        for col in shift_columns:
            T[f"next_{col}"] = T[col].shift(-self.dt)

        T["dM"] = (
            jnp.round((T["next_mid_price"] - T["mid_price"]) / tick_size * 2)
            * tick_size
            / 2
        )
        T = T.loc[(T.dM <= tick_size * 1.1) & (T.dM >= -tick_size * 1.1)]

        T2 = T.copy(deep=True)

        T2["imbalance_bucket"] = self.n_imbalances - 1 - T2["imbalance_bucket"]

        T2["next_imbalance_bucket"] = (
            self.n_imbalances - 1 - T2["next_imbalance_bucket"]
        )

        T2["dM"] = -T2["dM"]
        T2["mid_price"] = -T2["mid_price"]

        T3 = pd.concat([T, T2])
        T3.index = pd.RangeIndex(len(T3.index))

        return T3, tick_size

    @staticmethod
    @jit
    def _compute_G(
        Q: jnp.ndarray,
        R1: jnp.ndarray,
        R2: jnp.ndarray,
        K: jnp.ndarray,
        n_imbalances: int,
        n_spread: int,
        iterations: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute the value of G using the given parameters.

        Args:
            Q (jnp.ndarray): The Q matrix.
            R1 (jnp.ndarray): The R1 matrix.
            R2 (jnp.ndarray): The R2 matrix.
            K (jnp.ndarray): The K matrix.
            n_imbalances (int): The number of imbalances.
            n_spread (int): The number of spreads.
            iterations (int): The number of iterations.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: A tuple containing the computed value of G and the B matrix.
        """
        eye = jnp.eye(n_imbalances * n_spread)
        inv_I_Q = jnp.linalg.inv(eye - Q)

        G1 = jnp.dot(jnp.dot(inv_I_Q, R1), K)

        B = jnp.dot(inv_I_Q, R2)

        def body_fun(G: jnp.ndarray) -> jnp.ndarray:
            return jnp.dot(B, G) + G1

        G = jax.lax.fori_loop(0, iterations, body_fun, G1)

        return G, B

    def fit(self, T: pd.DataFrame, iterations: int = 6) -> jnp.ndarray:
        """
        Fits the model to the given data.

        Args:
            T (pd.DataFrame): The input data frame.
            iterations (int): The number of iterations for the fitting process. Default is 6.

        Returns:
            jnp.ndarray: The fitted model.

        """
        no_move = T[T["dM"] == 0]

        no_move_counts = no_move.pivot_table(
            index=["next_imbalance_bucket"],
            columns=["spread", "imbalance_bucket"],
            values="timestamp",
            fill_value=0,
            aggfunc="count",
        ).unstack()

        Q_counts = jnp.array(no_move_counts.values).reshape(
            self.n_spread, self.n_imbalances, self.n_imbalances
        )
        Q_counts = jax.lax.map(lambda x: jnp.diag(x), Q_counts)

        move_counts = (
            T[T["dM"] != 0]
            .pivot_table(
                index=["dM"],
                columns=["spread", "imbalance_bucket"],
                values="timestamp",
                fill_value=0,
                aggfunc="count",
            )
            .unstack()
        )

        R_counts = jnp.array(move_counts.values).reshape(
            self.n_imbalances * self.n_spread, 4
        )

        T1 = jnp.concatenate((Q_counts, R_counts), axis=1).astype(float)
        T1 = T1 / T1.sum(axis=1, keepdims=True)

        Q = T1[:, : (self.n_imbalances * self.n_spread)]
        R1 = T1[:, (self.n_imbalances * self.n_spread) :]

        K = jnp.array([-0.01, -0.005, 0.005, 0.01])

        move_counts = T[T["dM"] != 0].pivot_table(
            index=["spread", "imbalance_bucket"],
            columns=["next_spread", "next_imbalance_bucket"],
            values="timestamp",
            fill_value=0,
            aggfunc="count",
        )

        R2_counts = jnp.array(move_counts.values).reshape(
            self.n_imbalances * self.n_spread, self.n_imbalances * self.n_spread
        )
        T2 = jnp.concatenate((Q_counts, R2_counts), axis=1).astype(float)
        T2 = T2 / T2.sum(axis=1, keepdims=True)

        R2 = T2[:, (self.n_imbalances * self.n_spread) :]

        G, _ = self._compute_G(
            Q, R1, R2, K, self.n_imbalances, self.n_spread, iterations
        )

        return G
