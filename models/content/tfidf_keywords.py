"""
models/content/tfidf_keywords.py
==================================
Content-Based Filtering: TF-IDF on product KEYWORDS/TAGS.

EXPLANATION UPGRADE (audit fix):
  Explanation now names:
    - The actual matching keywords shared with the user's rated items
    - The specific highly-rated item that is most similar to the recommendation

This turns a generic "shares key attributes" into a specific "because
you rated 'Python Programming Book' highly, and this item shares tags:
programming, coding, beginner."
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from .content_base import preprocess_text, compute_cosine_sim, aggregate_scores_for_user
from utils.result_format import format_results, RecommendationResult
from typing import List

METHOD_NAME = "tfidf_keywords"
APPROACH    = "content"


def build_similarity_matrix(items_df: pd.DataFrame):
    """
    ===== MODEL EXECUTION =====
    TF-IDF vectorize keyword field, compute cosine similarity matrix.
    """
    corpus = items_df["keywords"].fillna("").apply(preprocess_text).tolist()
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=500,
        sublinear_tf=True,     # log(1+tf) — dampens very frequent terms
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    sim_matrix   = compute_cosine_sim(tfidf_matrix)
    return sim_matrix, vectorizer


def _find_most_similar_rated(
    items_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    sim_matrix: np.ndarray,
    user_id: int,
    target_item_id: int,
) -> dict:
    """
    ===== EXPLANATION =====
    For a target item, find which of the user's highly-rated items
    is most similar to it (highest cosine similarity).
    Returns name and similarity score of that reference item.
    """
    id_to_idx = {int(row["item_id"]): idx for idx, row in items_df.iterrows()}
    target_idx = id_to_idx.get(target_item_id)
    if target_idx is None:
        return {}

    user_ratings = ratings_df[ratings_df["user_id"] == user_id]
    high_rated   = user_ratings[user_ratings["rating"] >= 3.5]
    if high_rated.empty:
        high_rated = user_ratings  # fall back to all rated

    best_sim   = -1.0
    best_name  = ""
    best_kw    = ""
    best_rating = 0.0

    for _, rrow in high_rated.iterrows():
        rid = int(rrow["item_id"])
        ridx = id_to_idx.get(rid)
        if ridx is None or ridx == target_idx:
            continue
        sim = float(sim_matrix[target_idx][ridx])
        if sim > best_sim:
            best_sim    = sim
            item_meta   = items_df[items_df["item_id"] == rid]
            best_name   = item_meta["name"].values[0]   if len(item_meta) else ""
            best_kw     = item_meta["keywords"].values[0] if len(item_meta) else ""
            best_rating = float(rrow["rating"])

    return {
        "similar_to":   best_name,
        "similarity":   round(best_sim, 3),
        "your_rating":  round(best_rating, 1),
        "ref_keywords": best_kw,
    }


def recommend(
    ratings_df: pd.DataFrame,
    items_df: pd.DataFrame,
    user_id: int,
    n: int = 10,
) -> List[RecommendationResult]:
    """
    ===== MODEL EXECUTION =====
    Build TF-IDF similarity, aggregate scores across user's rated items.

    ===== EXPLANATION =====
    Each recommendation names the specific rated item that drove it and
    the overlapping keywords (feature similarity made concrete).

    ===== OUTPUT =====
    Returns List[RecommendationResult], sorted by aggregated score.
    """
    # ── Build similarity ───────────────────────────────────────
    sim_matrix, _ = build_similarity_matrix(items_df)

    # ── Aggregate scores ───────────────────────────────────────
    scores = aggregate_scores_for_user(ratings_df, items_df, sim_matrix, user_id, n)

    # ── Build per-item explanations ───────────────────────────
    exp_cache = {
        iid: _find_most_similar_rated(items_df, ratings_df, sim_matrix, user_id, iid)
        for iid, _ in scores
    }

    def explain(row):
        # ===== EXPLANATION =====
        iid  = int(row["item_id"])
        info = exp_cache.get(iid, {})
        item_kw = str(row.get("keywords", ""))
        kw_list = ", ".join(item_kw.split()[:4]) if item_kw else "relevant attributes"

        if info.get("similar_to") and info["similarity"] > 0.01:
            # Find keyword overlap for richer explanation
            ref_kws = set(str(info.get("ref_keywords", "")).lower().split())
            tgt_kws = set(item_kw.lower().split())
            overlap = ref_kws & tgt_kws
            overlap_str = (", ".join(sorted(overlap)[:3])
                           if overlap else kw_list)
            return (
                f"Recommended because '{row['name']}' shares attributes "
                f"({overlap_str}) with '{info['similar_to']}', "
                f"which you rated {info['your_rating']}/5. "
                f"(TF-IDF Keywords, cosine sim: {info['similarity']:.3f})"
            )
        return (
            f"Recommended because '{row['name']}' shares key attributes "
            f"({kw_list}) with products you have rated highly. (TF-IDF Keywords)"
        )

    # ===== OUTPUT =====
    return format_results(items_df, scores, APPROACH, METHOD_NAME, explain)
