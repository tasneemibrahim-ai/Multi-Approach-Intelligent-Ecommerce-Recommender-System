"""
models/collaborative/item_knn.py
=================================
Item-Based KNN Collaborative Filtering — pure NumPy implementation.

EXPLANATION UPGRADE (audit fix):
  Names which highly-rated items the user owns that are most similar
  to the recommended item (the actual "because you liked X" logic).
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .cf_base import build_user_item_matrix, get_unrated_items
from utils.result_format import format_results, RecommendationResult
from typing import List

METHOD_NAME = "item_knn"
APPROACH    = "collaborative"


class ItemKNNAlgo:
    def __init__(self, k: int = 20):
        self.k = k
        self.matrix = self.item_sim = None
        self.user_ids = self.item_ids = None

    def fit(self, ratings_df: pd.DataFrame):
        mat = build_user_item_matrix(ratings_df)
        self.user_ids = list(mat.index)
        self.item_ids = list(mat.columns)
        self.matrix   = mat
        self.item_sim = cosine_similarity(mat.fillna(0).values.T)
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        if user_id not in self.user_ids or item_id not in self.item_ids:
            return 3.0
        u_idx    = self.user_ids.index(user_id)
        i_idx    = self.item_ids.index(item_id)
        user_row = self.matrix.iloc[u_idx].values
        sims     = self.item_sim[i_idx]
        rated_mask = ~np.isnan(user_row); rated_mask[i_idx] = False
        if not rated_mask.any(): return 3.0
        ns = sims[rated_mask]; nr = user_row[rated_mask]
        top = np.argsort(ns)[-self.k:]
        ts  = ns[top]; tr = nr[top]
        d   = np.sum(np.abs(ts))
        return float(np.clip(np.dot(ts, tr) / d, 1, 5)) if d else 3.0

    def explain_item(self, ratings_df: pd.DataFrame, items_df: pd.DataFrame,
                     user_id: int, item_id: int) -> dict:
        """
        ===== EXPLANATION =====
        Find which items the user rated highly are most similar to this item.
        Returns name of the most similar rated item and its similarity score.
        """
        if user_id not in self.user_ids or item_id not in self.item_ids:
            return {}
        u_idx    = self.user_ids.index(user_id)
        i_idx    = self.item_ids.index(item_id)
        user_row = self.matrix.iloc[u_idx].values
        sims     = self.item_sim[i_idx]

        # Only consider items the user rated >= 3.5 (positive signal)
        high_rated_mask = (~np.isnan(user_row)) & (user_row >= 3.5)
        high_rated_mask[i_idx] = False
        if not high_rated_mask.any():
            return {}

        high_sims  = sims[high_rated_mask]
        high_ids   = np.array(self.item_ids)[high_rated_mask]
        high_rates = user_row[high_rated_mask]
        best_idx   = np.argmax(high_sims)
        best_iid   = int(high_ids[best_idx])
        best_sim   = round(float(high_sims[best_idx]), 3)
        best_rate  = round(float(high_rates[best_idx]), 1)
        best_name_row = items_df[items_df["item_id"] == best_iid]
        best_name  = best_name_row["name"].values[0] if len(best_name_row) else f"item {best_iid}"
        return {"similar_to": best_name, "similarity": best_sim, "your_rating": best_rate}


def recommend(ratings_df: pd.DataFrame, items_df: pd.DataFrame,
              user_id: int, n: int = 10) -> List[RecommendationResult]:
    """
    ===== MODEL EXECUTION =====
    Train ItemKNN, score all unrated items, take top-N.

    ===== EXPLANATION =====
    Each explanation names the specific rated item that is most similar
    to the recommendation ("because you rated X highly").
    """
    # ── Train ─────────────────────────────────────────────────
    algo = ItemKNNAlgo()
    algo.fit(ratings_df)

    # ── Score ─────────────────────────────────────────────────
    unrated = get_unrated_items(ratings_df, items_df, user_id)
    scores  = sorted([(iid, algo.predict(user_id, iid)) for iid in unrated],
                     key=lambda x: x[1], reverse=True)[:n]

    # ── Build per-item explanations ───────────────────────────
    exp_cache = {iid: algo.explain_item(ratings_df, items_df, user_id, iid)
                 for iid, _ in scores}

    def explain(row):
        # ===== EXPLANATION =====
        iid  = int(row["item_id"])
        info = exp_cache.get(iid, {})
        if info.get("similar_to"):
            return (
                f"Recommended because '{row['name']}' is similar "
                f"(score: {info['similarity']:.2f}) to '{info['similar_to']}', "
                f"which you rated {info['your_rating']}/5. (Item-Based KNN)"
            )
        return (
            f"Recommended because '{row['name']}' is similar to items "
            f"you have already rated highly. (Item-Based KNN)"
        )

    # ===== OUTPUT =====
    return format_results(items_df, scores, APPROACH, METHOD_NAME, explain)


def get_trained_algo(ratings_df: pd.DataFrame) -> ItemKNNAlgo:
    return ItemKNNAlgo().fit(ratings_df)
