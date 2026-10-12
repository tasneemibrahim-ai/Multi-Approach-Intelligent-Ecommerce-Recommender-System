"""
models/collaborative/svdpp_model.py
=====================================
SVD++ — extends SVD with implicit feedback (interaction sets).

The implicit term adds: sum(y_j for j in rated_items) / sqrt(|rated_items|)
This makes recommendations aware of WHICH items a user interacted with,
not just the explicit rating values.
"""

import numpy as np
import pandas as pd
from .cf_base import get_unrated_items
from utils.result_format import format_results, RecommendationResult
from typing import List

METHOD_NAME = "svdpp"
APPROACH = "collaborative"


class SVDppAlgo:
    def __init__(self, n_factors=15, n_epochs=15, lr=0.007, reg=0.02):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr; self.reg = reg

    def fit(self, ratings_df):
        users = sorted(ratings_df["user_id"].unique())
        items = sorted(ratings_df["item_id"].unique())
        self.user_idx = {u: i for i, u in enumerate(users)}
        self.item_idx = {it: i for i, it in enumerate(items)}
        n_u, n_i = len(users), len(items)
        self.global_mean = ratings_df["rating"].mean()
        self.U  = np.random.normal(0, 0.1, (n_u, self.n_factors))
        self.V  = np.random.normal(0, 0.1, (n_i, self.n_factors))
        self.Y  = np.random.normal(0, 0.01, (n_i, self.n_factors))  # implicit factors
        self.bu = np.zeros(n_u); self.bi = np.zeros(n_i)
        # Precompute implicit item sets per user
        user_items = ratings_df.groupby("user_id")["item_id"].apply(list).to_dict()
        records = ratings_df[["user_id","item_id","rating"]].values
        for _ in range(self.n_epochs):
            np.random.shuffle(records)
            for uid, iid, r in records:
                u = self.user_idx.get(int(uid)); i = self.item_idx.get(int(iid))
                if u is None or i is None: continue
                impl_items = [self.item_idx[j] for j in user_items.get(int(uid), []) if j in self.item_idx]
                sqrt_n = np.sqrt(max(len(impl_items), 1))
                impl_vec = self.Y[impl_items].sum(axis=0) / sqrt_n if impl_items else np.zeros(self.n_factors)
                pred = self.global_mean + self.bu[u] + self.bi[i] + (self.U[u] + impl_vec) @ self.V[i]
                err = r - pred
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                self.U[u]  += self.lr * (err * self.V[i] - self.reg * self.U[u])
                self.V[i]  += self.lr * (err * (self.U[u] + impl_vec) - self.reg * self.V[i])
                if impl_items:
                    grad_y = err * self.V[i] / sqrt_n
                    self.Y[impl_items] += self.lr * (grad_y - self.reg * self.Y[impl_items])
        self.user_items = user_items
        return self

    def predict(self, user_id, item_id):
        u = self.user_idx.get(user_id); i = self.item_idx.get(item_id)
        if u is None or i is None: return self.global_mean
        impl_items = [self.item_idx[j] for j in self.user_items.get(user_id, []) if j in self.item_idx]
        sqrt_n = np.sqrt(max(len(impl_items), 1))
        impl_vec = self.Y[impl_items].sum(axis=0) / sqrt_n if impl_items else np.zeros(self.n_factors)
        return float(np.clip(self.global_mean + self.bu[u] + self.bi[i] + (self.U[u] + impl_vec) @ self.V[i], 1, 5))


def recommend(ratings_df, items_df, user_id, n=10, n_factors=15) -> List[RecommendationResult]:
    algo = SVDppAlgo(n_factors=n_factors).fit(ratings_df)
    unrated = get_unrated_items(ratings_df, items_df, user_id)
    scores = sorted([(iid, algo.predict(user_id, iid)) for iid in unrated], key=lambda x: x[1], reverse=True)[:n]
    def explain(row):
        return f"Recommended by SVD++ which considers both explicit ratings AND implicit interaction patterns. '{row['name']}' fits your profile."
    return format_results(items_df, scores, APPROACH, METHOD_NAME, explain)

def get_trained_algo(ratings_df, n_factors=15):
    return SVDppAlgo(n_factors=n_factors).fit(ratings_df)
