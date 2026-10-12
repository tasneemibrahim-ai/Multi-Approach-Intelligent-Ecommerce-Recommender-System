"""
models/collaborative/nmf_model.py
====================================
NMF (Non-Negative Matrix Factorization) using sklearn.
Factorizes the filled user-item matrix into W @ H where W, H >= 0.
Non-negativity makes factors interpretable as additive preference topics.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from .cf_base import build_user_item_matrix, get_unrated_items
from utils.result_format import format_results, RecommendationResult
from typing import List

METHOD_NAME = "nmf"
APPROACH = "collaborative"


class NMFAlgo:
    def __init__(self, n_factors=15, n_epochs=200):
        self.n_factors = n_factors
        self.n_epochs = n_epochs

    def fit(self, ratings_df):
        mat = build_user_item_matrix(ratings_df)
        self.user_ids = list(mat.index)
        self.item_ids = list(mat.columns)
        self.global_mean = ratings_df["rating"].mean()
        # Fill NaN with global mean for NMF (requires non-negative dense matrix)
        filled = mat.fillna(self.global_mean).values
        model = NMF(n_components=self.n_factors, max_iter=self.n_epochs, random_state=42)
        self.W = model.fit_transform(filled)  # user factors
        self.H = model.components_            # item factors
        return self

    def predict(self, user_id, item_id):
        if user_id not in self.user_ids or item_id not in self.item_ids:
            return self.global_mean
        u = self.user_ids.index(user_id)
        i = self.item_ids.index(item_id)
        return float(np.clip(self.W[u] @ self.H[:, i], 1, 5))


def recommend(ratings_df, items_df, user_id, n=10, n_factors=15) -> List[RecommendationResult]:
    algo = NMFAlgo(n_factors=n_factors).fit(ratings_df)
    unrated = get_unrated_items(ratings_df, items_df, user_id)
    scores = sorted([(iid, algo.predict(user_id, iid)) for iid in unrated], key=lambda x: x[1], reverse=True)[:n]
    def explain(row):
        return f"Recommended by NMF. '{row['name']}' matches non-negative latent preference topics (e.g., category affinity, price sensitivity) learned from your history."
    return format_results(items_df, scores, APPROACH, METHOD_NAME, explain)

def get_trained_algo(ratings_df, n_factors=15):
    return NMFAlgo(n_factors=n_factors).fit(ratings_df)
