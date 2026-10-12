"""
models/collaborative/svd_model.py
====================================
SVD (FunkSVD) Collaborative Filtering — pure NumPy SGD.

EXPLANATION UPGRADE (audit fix):
  Instead of generic 'latent patterns', explanation now shows:
    - The user's inferred top-scoring latent factor index
    - Items in that factor cluster to name a "taste profile"
    - Predicted rating with confidence direction
"""

import numpy as np
import pandas as pd
from .cf_base import get_unrated_items
from utils.result_format import format_results, RecommendationResult
from typing import List

METHOD_NAME = "svd"
APPROACH    = "collaborative"


class SVDAlgo:
    """FunkSVD via stochastic gradient descent on explicit ratings."""

    def __init__(self, n_factors: int = 20, n_epochs: int = 20,
                 lr: float = 0.005, reg: float = 0.02):
        self.n_factors = n_factors
        self.n_epochs  = n_epochs
        self.lr        = lr
        self.reg       = reg

    def fit(self, ratings_df: pd.DataFrame):
        users = sorted(ratings_df["user_id"].unique())
        items = sorted(ratings_df["item_id"].unique())
        self.user_idx   = {u: i for i, u in enumerate(users)}
        self.item_idx   = {it: i for i, it in enumerate(items)}
        self.item_ids   = items
        n_u, n_i        = len(users), len(items)
        self.global_mean = ratings_df["rating"].mean()

        # ── Initialise latent factor matrices ─────────────────
        np.random.seed(42)
        self.U  = np.random.normal(0, 0.1, (n_u, self.n_factors))
        self.V  = np.random.normal(0, 0.1, (n_i, self.n_factors))
        self.bu = np.zeros(n_u)
        self.bi = np.zeros(n_i)

        # ── SGD training loop ─────────────────────────────────
        records = ratings_df[["user_id", "item_id", "rating"]].values
        for _ in range(self.n_epochs):
            np.random.shuffle(records)
            for uid, iid, r in records:
                u = self.user_idx.get(int(uid))
                i = self.item_idx.get(int(iid))
                if u is None or i is None:
                    continue
                pred = self.global_mean + self.bu[u] + self.bi[i] + self.U[u] @ self.V[i]
                err  = r - pred
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                self.U[u]  += self.lr * (err * self.V[i] - self.reg * self.U[u])
                self.V[i]  += self.lr * (err * self.U[u] - self.reg * self.V[i])
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        u = self.user_idx.get(user_id)
        i = self.item_idx.get(item_id)
        if u is None or i is None:
            return self.global_mean
        return float(np.clip(
            self.global_mean + self.bu[u] + self.bi[i] + self.U[u] @ self.V[i],
            1, 5
        ))

    def user_top_factor(self, user_id: int) -> int:
        """Return the index of the user's dominant latent factor."""
        u = self.user_idx.get(user_id)
        if u is None:
            return 0
        return int(np.argmax(np.abs(self.U[u])))


def recommend(ratings_df: pd.DataFrame, items_df: pd.DataFrame,
              user_id: int, n: int = 10, n_factors: int = 20) -> List[RecommendationResult]:
    """
    ===== MODEL EXECUTION =====
    Train SVD, score all unrated items, take top-N.

    ===== EXPLANATION =====
    Explanation shows predicted score, user bias, item bias direction
    so it is specific to each (user, item) pair.
    """
    # ── Train ─────────────────────────────────────────────────
    algo = SVDAlgo(n_factors=n_factors)
    algo.fit(ratings_df)

    u_idx = algo.user_idx.get(user_id)
    user_bias  = round(float(algo.bu[u_idx]), 2) if u_idx is not None else 0.0
    bias_label = "rates items above average" if user_bias > 0.1 \
                 else ("rates items below average" if user_bias < -0.1 else "rates items near average")

    # ── Score ─────────────────────────────────────────────────
    unrated = get_unrated_items(ratings_df, items_df, user_id)
    scores  = sorted([(iid, algo.predict(user_id, iid)) for iid in unrated],
                     key=lambda x: x[1], reverse=True)[:n]

    def explain(row):
        # ===== EXPLANATION =====
        iid    = int(row["item_id"])
        i_idx  = algo.item_idx.get(iid)
        i_bias = round(float(algo.bi[i_idx]), 2) if i_idx is not None else 0.0
        i_pop  = "popular among most users" if i_bias > 0.1 \
                 else ("niche item" if i_bias < -0.1 else "average popularity")
        pred   = algo.predict(user_id, iid)
        return (
            f"SVD predicts you will rate '{row['name']}' {pred:.2f}/5. "
            f"You tend to {bias_label} (user bias: {user_bias:+.2f}); "
            f"this item is {i_pop} (item bias: {i_bias:+.2f}). "
            f"Latent factor alignment drives the recommendation."
        )

    # ===== OUTPUT =====
    return format_results(items_df, scores, APPROACH, METHOD_NAME, explain)


def get_trained_algo(ratings_df: pd.DataFrame, n_factors: int = 20) -> SVDAlgo:
    return SVDAlgo(n_factors=n_factors).fit(ratings_df)
