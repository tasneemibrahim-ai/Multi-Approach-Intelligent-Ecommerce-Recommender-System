"""
utils/result_format.py
======================
Defines a unified recommendation result format used across ALL modules.
This ensures every approach returns data in the same structure,
making comparison and display straightforward.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RecommendationResult:
    """
    Unified format for a single recommendation.
    Every model (CF, Content, Knowledge) must return a list of these.
    """
    item_id: int
    item_name: str
    predicted_score: float          # Normalized 0–5 or 0–1 score
    explanation: str                 # Human-readable reason for recommendation
    category: str = ""
    price: float = 0.0
    brand: str = ""
    approach: str = ""              # "collaborative" | "content" | "knowledge"
    method: str = ""                # e.g. "user_knn", "tfidf_genre", "constraint"
    extra: dict = field(default_factory=dict)  # Any method-specific metadata


def format_results(items_df, scores: list, approach: str, method: str, explanation_fn) -> List[RecommendationResult]:
    """
    Helper to convert (item_id, score) pairs into RecommendationResult objects.

    Args:
        items_df     : pandas DataFrame with item metadata
        scores       : list of (item_id, score) tuples, descending by score
        approach     : 'collaborative' | 'content' | 'knowledge'
        method       : specific method name
        explanation_fn: callable(item_row) -> str
    """
    results = []
    for item_id, score in scores:
        row = items_df[items_df["item_id"] == item_id]
        if row.empty:
            continue
        row = row.iloc[0]
        results.append(RecommendationResult(
            item_id=int(item_id),
            item_name=row.get("name", f"Item {item_id}"),
            predicted_score=round(float(score), 4),
            explanation=explanation_fn(row),
            category=row.get("category", ""),
            price=float(row.get("price", 0)),
            brand=row.get("brand", ""),
            approach=approach,
            method=method,
        ))
    return results
