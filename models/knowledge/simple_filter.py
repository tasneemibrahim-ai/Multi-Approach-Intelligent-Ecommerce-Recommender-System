"""
models/knowledge/simple_filter.py
====================================
Knowledge-Based Recommendation: Simple Filtering.

Logic: Filter products by hard constraints supplied by the user
(category, max price, brand). Return items matching ALL constraints,
sorted by average community rating (or price ascending if no ratings).

This is the most basic knowledge-based approach:
  - No learning required
  - Fully transparent / explainable
  - Works from day 1 with zero user history (no cold-start problem)

Constraints accepted:
  category  : str or None  – exact category match
  max_price : float or None – upper price bound
  brand     : str or None  – exact brand match
"""

import pandas as pd
from utils.result_format import format_results, RecommendationResult
from typing import List, Optional


METHOD_NAME = "simple_filter"
APPROACH = "knowledge"


def recommend(
    items_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    user_id: int,
    category: Optional[str] = None,
    max_price: Optional[float] = None,
    brand: Optional[str] = None,
    min_rating: float = 1.0,   # ===== INPUT HANDLING ===== min community rating filter
    n: int = 10,
) -> List[RecommendationResult]:
    """
    Filter items by hard constraints and rank by average community rating.

    Args:
        items_df   : product catalogue DataFrame
        ratings_df : all user ratings DataFrame
        user_id    : target user (to exclude already-rated items)
        category   : filter by this category (case-insensitive)
        max_price  : upper bound on price
        brand      : filter by this brand (case-insensitive)
        n          : number of recommendations to return
    """
    df = items_df.copy()

    # ── Apply hard filters ─────────────────────────────────────
    active_constraints = []

    if category:
        df = df[df["category"].str.lower() == category.lower()]
        active_constraints.append(f"category = '{category}'")

    if max_price is not None:
        df = df[df["price"] <= max_price]
        active_constraints.append(f"price ≤ ${max_price:.2f}")

    if brand:
        df = df[df["brand"].str.lower() == brand.lower()]
        active_constraints.append(f"brand = '{brand}'")

    # ── Exclude already-rated items ───────────────────────────
    rated_ids = set(ratings_df[ratings_df["user_id"] == user_id]["item_id"].tolist())
    df = df[~df["item_id"].isin(rated_ids)]

    if df.empty:
        return []

    # ── Score by average community rating ─────────────────────
    avg_ratings = ratings_df.groupby("item_id")["rating"].mean().reset_index()
    avg_ratings.columns = ["item_id", "avg_rating"]
    df = df.merge(avg_ratings, on="item_id", how="left")
    df["avg_rating"] = df["avg_rating"].fillna(3.0)  # default neutral score

    # ===== INPUT HANDLING ===== Apply minimum rating threshold
    df = df[df["avg_rating"] >= min_rating]
    df = df.sort_values("avg_rating", ascending=False).head(n)

    if min_rating > 1.0:
        active_constraints.append(f"community rating ≥ {min_rating}")
    constraints_str = " AND ".join(active_constraints) if active_constraints else "no filters applied"

    scores = list(zip(df["item_id"].tolist(), df["avg_rating"].tolist()))

    def explain(row):
        return (
            f"Recommended by simple constraint filtering ({constraints_str}). "
            f"'{row['name']}' (${row['price']:.2f}, {row['brand']}) matches your requirements "
            f"and has a high average community rating."
        )

    return format_results(items_df, scores, APPROACH, METHOD_NAME, explain)
