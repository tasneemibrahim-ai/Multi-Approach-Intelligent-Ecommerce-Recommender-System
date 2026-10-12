"""
models/content/content_base.py
================================
Shared utilities for all Content-Based Filtering models.
Provides text preprocessing, cosine similarity computation,
and a unified interface for retrieving similar items.
"""

import numpy as np
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────────────────────────────────────────
#  TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────────

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "be",
    "this", "that", "it", "as", "your", "you", "our", "its"
}


def preprocess_text(text: str) -> str:
    """
    Lowercase, remove punctuation, strip stopwords.
    Used before TF-IDF vectorization.
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # remove punctuation
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


# ─────────────────────────────────────────────────────────────
#  SIMILARITY HELPERS
# ─────────────────────────────────────────────────────────────

def compute_cosine_sim(matrix) -> np.ndarray:
    """Compute pairwise cosine similarity matrix from a feature matrix."""
    return cosine_similarity(matrix)


def get_similar_items(
    sim_matrix: np.ndarray,
    item_index: int,
    items_df: pd.DataFrame,
    n: int = 10,
    exclude_ids: list = None
) -> list:
    """
    Given a cosine similarity matrix and a reference item index,
    return the top-N most similar items (excluding already-seen ones).

    Returns list of (item_id, similarity_score) tuples.
    """
    exclude_ids = set(exclude_ids or [])
    scores = list(enumerate(sim_matrix[item_index]))
    # Remove the item itself
    scores = [(i, s) for i, s in scores if i != item_index]
    # Sort by similarity descending
    scores.sort(key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in scores:
        item_id = int(items_df.iloc[idx]["item_id"])
        if item_id not in exclude_ids:
            results.append((item_id, score))
        if len(results) >= n:
            break
    return results


def aggregate_scores_for_user(
    ratings_df: pd.DataFrame,
    items_df: pd.DataFrame,
    sim_matrix: np.ndarray,
    user_id: int,
    n: int = 10
) -> list:
    """
    For a user, aggregate content similarity scores across all items they rated.
    Each unrated item's score = weighted average similarity to rated items,
    weighted by the user's rating.

    Returns list of (item_id, aggregated_score) tuples sorted descending.
    """
    # Items the user has rated
    user_ratings = ratings_df[ratings_df["user_id"] == user_id]
    rated_ids = set(user_ratings["item_id"].tolist())
    all_ids = set(items_df["item_id"].tolist())
    unrated_ids = all_ids - rated_ids

    # Build index mapping: item_id → row index in items_df
    id_to_idx = {int(row["item_id"]): idx for idx, row in items_df.iterrows()}

    # For each unrated item, compute weighted similarity score
    scores = {}
    for unrated_id in unrated_ids:
        if unrated_id not in id_to_idx:
            continue
        unrated_idx = id_to_idx[unrated_id]
        total_weight = 0.0
        weighted_sum = 0.0
        for _, r_row in user_ratings.iterrows():
            rated_id = int(r_row["item_id"])
            rating = float(r_row["rating"])
            if rated_id not in id_to_idx:
                continue
            rated_idx = id_to_idx[rated_id]
            sim = sim_matrix[unrated_idx][rated_idx]
            weighted_sum += sim * rating
            total_weight += rating
        if total_weight > 0:
            scores[unrated_id] = weighted_sum / total_weight

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores[:n]
