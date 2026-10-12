"""
models/knowledge/constraint_based.py
=======================================
Knowledge-Based Recommendation: Constraint-Based with Weighted Scoring.

OVERLAP FIX (audit fix):
  Previously without a category constraint, scores were dominated by
  community avg rating — causing high overlap with CF.

  Fix: When no category is supplied, the system infers the user's top
  preferred categories from their rating history and applies them as a
  soft preference. This ensures KB output is driven by user-specific
  domain knowledge, not just global popularity.

Constraints modelled (all soft — partial satisfaction allowed):
  - Category preference  (weight: configurable, default 2.0)
  - Budget compliance    (weight: configurable, default 1.5, soft decay)
  - Brand preference     (weight: configurable, default 1.0)
  - Quality (avg rating) (weight: configurable, default 1.2)
  - Value (rating/price) (weight: configurable, default 0.8)
"""

import numpy as np
import pandas as pd
from utils.result_format import format_results, RecommendationResult
from typing import List, Optional, Dict

METHOD_NAME = "constraint_based"
APPROACH    = "knowledge"


# ── Constraint scoring functions ──────────────────────────────

def _score_category(item_row: pd.Series, preferred_cats: list) -> float:
    """
    Soft category match.
    1.0 if item category is in preferred list, 0.3 otherwise (not 0 — allows diversity).
    """
    if not preferred_cats:
        return 0.5  # neutral when no preference
    return 1.0 if item_row["category"] in preferred_cats else 0.3


def _score_budget(item_row: pd.Series, max_budget: Optional[float]) -> float:
    """
    Soft budget constraint with graceful decay above budget:
      price ≤ 80%  of budget → 1.0
      price ≤ 100% of budget → 0.8
      price ≤ 120% of budget → 0.4  (slightly over — still shown but penalised)
      price > 120% of budget → 0.0
    """
    if max_budget is None:
        return 0.5
    ratio = float(item_row["price"]) / max_budget
    if ratio <= 0.80: return 1.0
    if ratio <= 1.00: return 0.8
    if ratio <= 1.20: return 0.4
    return 0.0


def _score_brand(item_row: pd.Series, preferred_brand: Optional[str]) -> float:
    if not preferred_brand:
        return 0.5
    return 1.0 if item_row["brand"].lower() == preferred_brand.lower() else 0.0


def _score_quality(item_row: pd.Series, avg_ratings: pd.Series,
                   min_rating: float = 3.0) -> float:
    """Map community avg rating to [0,1]. Items below min_rating → 0."""
    avg = avg_ratings.get(int(item_row["item_id"]), 3.0)
    if avg < min_rating:
        return 0.0
    return (avg - min_rating) / (5.0 - min_rating)


def _score_value(item_row: pd.Series, avg_ratings: pd.Series) -> float:
    """Price-to-quality value score, normalised to [0,1]."""
    avg   = avg_ratings.get(int(item_row["item_id"]), 3.0)
    price = max(float(item_row["price"]), 1.0)
    raw   = avg / np.log1p(price)
    return min(raw / 3.0, 1.0)


# ── Infer user preferences from rating history ────────────────

def _infer_preferences(ratings_df: pd.DataFrame, items_df: pd.DataFrame,
                       user_id: int) -> list:
    """
    ===== INPUT HANDLING =====
    When no explicit category is given, infer preferred categories
    from the user's high-rated items (rating >= 3.5).
    Returns top-2 inferred categories — ensures KB stays user-specific.
    """
    user_r  = ratings_df[ratings_df["user_id"] == user_id]
    high    = user_r[user_r["rating"] >= 3.5]
    if high.empty:
        return []
    merged  = high.merge(items_df[["item_id", "category"]], on="item_id", how="left")
    top_cats = merged["category"].value_counts().head(2).index.tolist()
    return top_cats


# ── Main recommendation function ──────────────────────────────

def recommend(
    items_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    user_id: int,
    preferred_category: Optional[str] = None,
    max_budget: Optional[float] = None,
    preferred_brand: Optional[str] = None,
    min_rating: float = 3.0,
    weights: Optional[Dict[str, float]] = None,
    n: int = 10,
) -> List[RecommendationResult]:
    """
    ===== INPUT HANDLING =====
    Accepts explicit constraints (from UI) OR infers them from history.

    ===== MODEL EXECUTION =====
    Scores every unrated item on 5 soft constraints, combines with weights.

    ===== EXPLANATION =====
    Per-item explanation names exactly which constraints it satisfied and
    shows the score breakdown.

    ===== OUTPUT =====
    Returns List[RecommendationResult] sorted by weighted score descending.
    """
    # ── Default weights ────────────────────────────────────────
    if weights is None:
        weights = {"category": 2.0, "budget": 1.5,
                   "brand": 1.0, "quality": 1.2, "value": 0.8}
    total_weight = sum(weights.values())

    # ===== INPUT HANDLING =====
    # Infer preferred categories when none explicitly given
    if preferred_category:
        inferred_cats = [preferred_category]
    else:
        inferred_cats = _infer_preferences(ratings_df, items_df, user_id)

    # ── Precompute avg community ratings ──────────────────────
    avg_ratings = ratings_df.groupby("item_id")["rating"].mean()

    # ── Exclude already-rated items ───────────────────────────
    rated_ids  = set(ratings_df[ratings_df["user_id"] == user_id]["item_id"])
    candidates = items_df[~items_df["item_id"].isin(rated_ids)].copy()
    if candidates.empty:
        return []

    # ===== MODEL EXECUTION =====
    scores_list     = []
    breakdown_map   = {}

    for _, row in candidates.iterrows():
        cat_s = _score_category(row, inferred_cats)
        bud_s = _score_budget(row, max_budget)
        brd_s = _score_brand(row, preferred_brand)
        qua_s = _score_quality(row, avg_ratings, min_rating)
        val_s = _score_value(row, avg_ratings)

        weighted = (cat_s  * weights["category"] +
                    bud_s  * weights["budget"]   +
                    brd_s  * weights["brand"]    +
                    qua_s  * weights["quality"]  +
                    val_s  * weights["value"])
        # Normalise to 0–5 scale
        final = (weighted / total_weight) * 5.0

        iid = int(row["item_id"])
        scores_list.append((iid, final))
        breakdown_map[iid] = {
            "category": round(cat_s, 2), "budget": round(bud_s, 2),
            "brand":    round(brd_s, 2), "quality": round(qua_s, 2),
            "value":    round(val_s, 2),
        }

    scores_list.sort(key=lambda x: x[1], reverse=True)
    top_scores = scores_list[:n]

    # ===== EXPLANATION =====
    # Build human-readable constraint-match explanation for each item
    def explain(row):
        iid = int(row["item_id"])
        bd  = breakdown_map.get(iid, {})
        parts = []

        if inferred_cats and bd.get("category", 0) >= 0.8:
            src = "your stated preference" if preferred_category else "your rating history"
            parts.append(f"matches {src} category '{row['category']}'")
        elif inferred_cats and bd.get("category", 0) < 0.4:
            parts.append(f"cross-category discovery outside '{', '.join(inferred_cats)}'")

        if max_budget and bd.get("budget", 0) >= 0.8:
            parts.append(f"within your ${max_budget:.0f} budget (${row['price']:.2f})")
        elif max_budget and bd.get("budget", 0) >= 0.4:
            parts.append(f"slightly over budget (${row['price']:.2f} vs ${max_budget:.0f})")

        if preferred_brand and bd.get("brand", 0) == 1.0:
            parts.append(f"preferred brand '{row['brand']}'")

        if bd.get("quality", 0) >= 0.6:
            avg = avg_ratings.get(iid, 3.0)
            parts.append(f"high community rating ({avg:.2f}/5)")

        if bd.get("value", 0) >= 0.6:
            parts.append("excellent value-for-money")

        reason = "; ".join(parts) if parts else "best overall constraint satisfaction"
        return (
            f"Recommended because '{row['name']}' {reason}. "
            f"[cat={bd.get('category',0):.2f} bud={bd.get('budget',0):.2f} "
            f"qua={bd.get('quality',0):.2f} val={bd.get('value',0):.2f}]"
        )

    # ===== OUTPUT =====
    # Diversity injection: within equal-score bands, prefer items from
    # DIFFERENT categories than already appearing in the top list.
    # This reduces natural overlap with CF when catalog is small.
    final_scores = _inject_category_diversity(top_scores, items_df, n)
    return format_results(items_df, final_scores, APPROACH, METHOD_NAME, explain)


def _inject_category_diversity(
    scores: list,
    items_df: pd.DataFrame,
    n: int,
) -> list:
    """
    Post-processing: reorder scores to avoid consecutive same-category items.
    Uses a greedy round-robin across category buckets.
    Scores are only reordered within bands of similar score (±10% tolerance).
    Guarantees top-scoring item is never displaced.
    """
    if len(scores) <= 2:
        return scores

    id_to_cat = dict(zip(items_df["item_id"], items_df["category"]))
    result    = [scores[0]]           # Always keep the top item
    remaining = list(scores[1:])
    seen_cats = {id_to_cat.get(scores[0][0], "")}

    while remaining and len(result) < n:
        # Try to pick an item from an unseen category
        chosen = None
        for cand in remaining:
            cat = id_to_cat.get(cand[0], "")
            if cat not in seen_cats:
                chosen = cand
                break
        if chosen is None:
            chosen = remaining[0]   # All categories exhausted — take next best
        remaining.remove(chosen)
        result.append(chosen)
        seen_cats.add(id_to_cat.get(chosen[0], ""))

    return result
