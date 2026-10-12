"""
models/collaborative/user_knn.py
=================================
User-Based KNN Collaborative Filtering — pure NumPy implementation.

EXPLANATION UPGRADE (audit fix):
  Explanations are now item-specific. They name:
    - The similarity score of the top neighbor
    - How many neighbors rated this item
    - The top neighbor's actual rating for the item
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .cf_base import build_user_item_matrix, get_unrated_items, cross_validate_cf
from utils.result_format import format_results, RecommendationResult
from typing import List

METHOD_NAME = "user_knn"
APPROACH    = "collaborative"


class UserKNNAlgo:
    """Stateful User-KNN model — cosine similarity on rating vectors."""

    def __init__(self, k: int = 20):
        self.k = k
        self.matrix = self.sim_matrix = None
        self.user_ids = self.item_ids = None

    def fit(self, ratings_df: pd.DataFrame):
        mat = build_user_item_matrix(ratings_df)
        self.user_ids   = list(mat.index)
        self.item_ids   = list(mat.columns)
        self.matrix     = mat
        filled          = mat.fillna(0).values
        self.sim_matrix = cosine_similarity(filled)
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        if user_id not in self.user_ids or item_id not in self.item_ids:
            return 3.0
        u_idx    = self.user_ids.index(user_id)
        i_idx    = self.item_ids.index(item_id)
        sims     = self.sim_matrix[u_idx]
        item_col = self.matrix.iloc[:, i_idx].values
        rated_mask = ~np.isnan(item_col)
        rated_mask[u_idx] = False
        if not rated_mask.any():
            return 3.0
        ns   = sims[rated_mask]
        nr   = item_col[rated_mask]
        top  = np.argsort(ns)[-self.k:]
        ts   = ns[top]; tr = nr[top]
        d    = np.sum(np.abs(ts))
        return float(np.clip(np.dot(ts, tr) / d, 1, 5)) if d else 3.0

    # ── SPECIFIC EXPLANATION HELPER ──────────────────────────
    def explain_item(self, user_id: int, item_id: int) -> dict:
        """
        Return structured explanation data for one (user, item) pair.
        Used to build per-item specific explanations.
        """
        if user_id not in self.user_ids or item_id not in self.item_ids:
            return {"n_neighbors": 0, "top_sim": 0.0, "top_rating": None, "avg_neighbor_rating": None}
        u_idx    = self.user_ids.index(user_id)
        i_idx    = self.item_ids.index(item_id)
        sims     = self.sim_matrix[u_idx]
        item_col = self.matrix.iloc[:, i_idx].values
        rated_mask = ~np.isnan(item_col)
        rated_mask[u_idx] = False
        if not rated_mask.any():
            return {"n_neighbors": 0, "top_sim": 0.0, "top_rating": None, "avg_neighbor_rating": None}
        ns   = sims[rated_mask]; nr = item_col[rated_mask]
        top  = np.argsort(ns)[-self.k:]
        ts   = ns[top]; tr = nr[top]
        return {
            "n_neighbors":         int((ts > 0).sum()),
            "top_sim":             round(float(ts.max()), 3),
            "top_rating":          round(float(tr[ts.argmax()]), 1),
            "avg_neighbor_rating": round(float(tr[ts > 0].mean()), 2) if (ts > 0).any() else None,
        }


def recommend(ratings_df: pd.DataFrame, items_df: pd.DataFrame,
              user_id: int, n: int = 10) -> List[RecommendationResult]:
    """
    ===== MODEL EXECUTION =====
    Train UserKNN, score all unrated items, take top-N.

    ===== EXPLANATION =====
    Explanation is item-specific: names neighbor count, top similarity,
    and average rating those neighbors gave this item.
    """
    # ── Train ─────────────────────────────────────────────────
    algo = UserKNNAlgo()
    algo.fit(ratings_df)

    # ── Score unrated items ───────────────────────────────────
    unrated = get_unrated_items(ratings_df, items_df, user_id)
    scores  = [(iid, algo.predict(user_id, iid)) for iid in unrated]
    scores.sort(key=lambda x: x[1], reverse=True)
    top_scores = scores[:n]

    # ── Build per-item explanations ───────────────────────────
    exp_cache = {iid: algo.explain_item(user_id, iid) for iid, _ in top_scores}

    def explain(row):
        # ===== EXPLANATION =====
        iid  = int(row["item_id"])
        info = exp_cache.get(iid, {})
        nn   = info.get("n_neighbors", 0)
        ts   = info.get("top_sim", 0.0)
        ar   = info.get("avg_neighbor_rating")
        if nn > 0 and ar is not None:
            return (
                f"Recommended because {nn} user(s) with similar taste to you "
                f"(top similarity: {ts:.2f}) rated '{row['name']}' an average of "
                f"{ar:.1f}/5. (User-Based KNN, cosine similarity)"
            )
        return (
            f"Recommended because users with similar taste to you rated "
            f"'{row['name']}' highly. (User-Based KNN)"
        )

    # ===== OUTPUT =====
    return format_results(items_df, top_scores, APPROACH, METHOD_NAME, explain)


def get_trained_algo(ratings_df: pd.DataFrame) -> UserKNNAlgo:
    return UserKNNAlgo().fit(ratings_df)
