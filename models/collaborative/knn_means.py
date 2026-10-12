"""
models/collaborative/knn_means.py
====================================
KNN with Means — bias-corrected User-Based CF.

Before computing similarity, subtract each user's mean rating.
This corrects for generous/stingy raters, making similarity more meaningful.
Prediction = target_mean + weighted_avg(neighbor_deviations)
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .cf_base import build_user_item_matrix, get_unrated_items
from utils.result_format import format_results, RecommendationResult
from typing import List

METHOD_NAME = "knn_means"
APPROACH = "collaborative"


class KNNMeansAlgo:
    def __init__(self, k=20):
        self.k = k

    def fit(self, ratings_df):
        mat = build_user_item_matrix(ratings_df)
        self.user_ids = list(mat.index)
        self.item_ids = list(mat.columns)
        self.matrix = mat
        # Compute per-user mean (ignoring NaN)
        self.user_means = mat.mean(axis=1).values
        # Mean-centered matrix for similarity
        centered = mat.subtract(mat.mean(axis=1), axis=0).fillna(0).values
        self.sim_matrix = cosine_similarity(centered)
        return self

    def predict(self, user_id, item_id):
        if user_id not in self.user_ids or item_id not in self.item_ids:
            return 3.0
        u_idx = self.user_ids.index(user_id)
        i_idx = self.item_ids.index(item_id)
        item_col = self.matrix.iloc[:, i_idx].values
        rated_mask = ~np.isnan(item_col); rated_mask[u_idx] = False
        if not rated_mask.any(): return self.user_means[u_idx]
        sims = self.sim_matrix[u_idx]
        neighbor_sims = sims[rated_mask]
        neighbor_devs = item_col[rated_mask] - self.user_means[rated_mask]
        top_k = np.argsort(neighbor_sims)[-self.k:]
        ts = neighbor_sims[top_k]; td = neighbor_devs[top_k]
        denom = np.sum(np.abs(ts))
        if denom == 0: return self.user_means[u_idx]
        return float(np.clip(self.user_means[u_idx] + np.dot(ts, td) / denom, 1, 5))


def recommend(ratings_df, items_df, user_id, n=10) -> List[RecommendationResult]:
    algo = KNNMeansAlgo().fit(ratings_df)
    unrated = get_unrated_items(ratings_df, items_df, user_id)
    scores = sorted([(iid, algo.predict(user_id, iid)) for iid in unrated], key=lambda x: x[1], reverse=True)[:n]
    def explain(row):
        return f"Recommended by bias-corrected KNN. '{row['name']}' scores above your average rating when compared to similar users' deviations."
    return format_results(items_df, scores, APPROACH, METHOD_NAME, explain)

def get_trained_algo(ratings_df):
    return KNNMeansAlgo().fit(ratings_df)
