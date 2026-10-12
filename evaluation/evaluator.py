"""
evaluation/evaluator.py
=========================
Comprehensive evaluation and comparison module.

Implements 7 evaluation metrics:
  1. RMSE           — Root Mean Squared Error (rating prediction accuracy)
  2. MAE            — Mean Absolute Error (rating prediction accuracy)
  3. Precision@K    — Fraction of top-K recs that are relevant
  4. Recall@K       — Fraction of all relevant items found in top-K
  5. F1@K           — Harmonic mean of Precision@K and Recall@K
  6. Coverage       — Fraction of catalog recommended across all users
  7. Diversity      — Intra-list category diversity score

Comparisons:
  compare_cf_methods()  → 6 CF algorithms, 3-fold cross-validated RMSE/MAE
  compare_approaches()  → CF vs Content vs Knowledge on all 7 metrics
  run_full_evaluation() → Runs both; entry point for Streamlit dashboard
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from typing import List, Dict
from utils.result_format import RecommendationResult
from models.collaborative.cf_base import cross_validate_cf


# ══════════════════════════════════════════════════════════════
#  RATING PREDICTION METRICS
# ══════════════════════════════════════════════════════════════

def compute_rmse(preds: list, actuals: list) -> float:
    """Root Mean Squared Error — lower is better."""
    p, a = np.array(preds, dtype=float), np.array(actuals, dtype=float)
    return float(np.sqrt(np.mean((p - a) ** 2)))


def compute_mae(preds: list, actuals: list) -> float:
    """Mean Absolute Error — lower is better."""
    p, a = np.array(preds, dtype=float), np.array(actuals, dtype=float)
    return float(np.mean(np.abs(p - a)))


# ══════════════════════════════════════════════════════════════
#  RANKING METRICS
# ══════════════════════════════════════════════════════════════

def precision_at_k(recommended_ids: list, relevant_ids: set, k: int) -> float:
    """
    Precision@K: of the top-K recommendations, what fraction is relevant?
    relevant_ids = items the user actually rated above the threshold in test set.
    """
    top_k = recommended_ids[:k]
    if not top_k:
        return 0.0
    return sum(1 for iid in top_k if iid in relevant_ids) / k


def recall_at_k(recommended_ids: list, relevant_ids: set, k: int) -> float:
    """
    Recall@K: of all relevant items, what fraction appears in top-K?
    """
    if not relevant_ids:
        return 0.0
    top_k = recommended_ids[:k]
    return sum(1 for iid in top_k if iid in relevant_ids) / len(relevant_ids)


def f1_at_k(precision: float, recall: float) -> float:
    """F1@K: harmonic mean of Precision@K and Recall@K."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_ranking(
    recommend_fn,
    ratings_df: pd.DataFrame,
    items_df: pd.DataFrame,
    k: int = 10,
    relevance_threshold: float = 3.5,
    n_users: int = 20,
) -> Dict:
    """
    Evaluate a recommendation function using a proper leave-one-out strategy:

      For each sampled user:
        1. Hold out their highest-rated item (the "test item").
        2. Remove it from training so the model cannot see it.
        3. Re-train / re-recommend from the reduced rating set.
        4. Check whether the held-out item appears in the top-K list.

    This avoids the common mistake of holding out items that are still in
    the training set (which makes recommendations always miss them).

    Returns dict with mean Precision@K, Recall@K, F1@K.
    """
    # Users with at least 5 ratings (need some to hold one out)
    active = (
        ratings_df.groupby("user_id")
        .filter(lambda x: len(x) >= 5)["user_id"]
        .unique()
    )
    sample = active[:n_users]

    precisions, recalls, f1s = [], [], []

    for uid in sample:
        user_r = ratings_df[ratings_df["user_id"] == uid]

        # Hold out the single highest-rated item (leave-one-out)
        held_out_row = user_r.loc[user_r["rating"].idxmax()]
        held_out_id  = int(held_out_row["item_id"])

        if held_out_row["rating"] < relevance_threshold:
            continue  # No truly relevant item to test against

        # Training set = all ratings EXCEPT the held-out item
        train_df = ratings_df[
            ~((ratings_df["user_id"] == uid) & (ratings_df["item_id"] == held_out_id))
        ].copy()

        relevant_ids = {held_out_id}

        try:
            recs     = recommend_fn(uid, train_df)
            rec_ids  = [r.item_id for r in recs]
            p  = precision_at_k(rec_ids, relevant_ids, k)
            r  = recall_at_k(rec_ids, relevant_ids, k)
            f1 = f1_at_k(p, r)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)
        except Exception:
            continue

    return {
        f"precision@{k}": round(float(np.mean(precisions)) if precisions else 0.0, 4),
        f"recall@{k}":    round(float(np.mean(recalls))    if recalls    else 0.0, 4),
        f"f1@{k}":        round(float(np.mean(f1s))        if f1s        else 0.0, 4),
        "n_evaluated":    len(precisions),
    }


# ══════════════════════════════════════════════════════════════
#  CATALOG COVERAGE
# ══════════════════════════════════════════════════════════════

def catalog_coverage(
    recommend_fn,
    all_item_ids: list,
    sample_users: list,
    ratings_df: pd.DataFrame,
    k: int = 10,
) -> float:
    """
    Catalog Coverage: what fraction of all items appear in recommendations
    across a sample of users?

    Coverage = |union of all recommended items| / |total catalog size|
    Higher is better — rewards surfacing long-tail items, not just popular ones.
    """
    recommended = set()
    for uid in sample_users:
        try:
            recs = recommend_fn(uid, ratings_df)
            for r in recs[:k]:
                recommended.add(r.item_id)
        except Exception:
            pass
    return round(len(recommended) / len(all_item_ids), 4) if all_item_ids else 0.0


# ══════════════════════════════════════════════════════════════
#  INTRA-LIST DIVERSITY
# ══════════════════════════════════════════════════════════════

def intra_list_diversity(recs: List[RecommendationResult]) -> float:
    """
    Intra-List Diversity: average pairwise category dissimilarity.

    Dissimilarity(i, j) = 1 if category[i] != category[j], else 0.
    Diversity = mean over all unique pairs.

    Range: 0 (all same category) → 1 (all different categories).
    Higher = more varied recommendation list.
    """
    if len(recs) < 2:
        return 0.0
    cats = [r.category for r in recs]
    pairs = dissimilar = 0
    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            pairs += 1
            if cats[i] != cats[j]:
                dissimilar += 1
    return round(dissimilar / pairs, 4) if pairs else 0.0


# ══════════════════════════════════════════════════════════════
#  CF METHOD COMPARISON  (cross-validated RMSE / MAE)
# ══════════════════════════════════════════════════════════════

def compare_cf_methods(ratings_df: pd.DataFrame, n_splits: int = 3) -> pd.DataFrame:
    """
    Cross-validate all 6 CF algorithms using k-fold CV.
    Each fold: train on 2/3 of ratings, test on 1/3.
    Reports mean ± std of RMSE and MAE across folds.

    Returns a DataFrame with one row per method.
    """
    from models.collaborative.user_knn    import UserKNNAlgo
    from models.collaborative.item_knn    import ItemKNNAlgo
    from models.collaborative.svd_model   import SVDAlgo
    from models.collaborative.svdpp_model import SVDppAlgo
    from models.collaborative.knn_means   import KNNMeansAlgo
    from models.collaborative.nmf_model   import NMFAlgo

    methods = {
        "User-KNN":  UserKNNAlgo,
        "Item-KNN":  ItemKNNAlgo,
        "SVD":       SVDAlgo,
        "SVD++":     SVDppAlgo,
        "KNN-Means": KNNMeansAlgo,
        "NMF":       NMFAlgo,
    }

    rows = []
    for name, cls in methods.items():
        print(f"    Evaluating {name}…")
        try:
            res = cross_validate_cf(cls, ratings_df, n_splits=n_splits)
            rows.append({
                "Method":   name,
                "Approach": "Collaborative Filtering",
                "RMSE":     res["rmse_mean"],
                "RMSE±":    res["rmse_std"],
                "MAE":      res["mae_mean"],
                "MAE±":     res["mae_std"],
            })
        except Exception:
            rows.append({
                "Method": name, "Approach": "Collaborative Filtering",
                "RMSE": None, "RMSE±": None, "MAE": None, "MAE±": None,
            })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
#  APPROACH COMPARISON  (CF vs Content vs Knowledge)
# ══════════════════════════════════════════════════════════════

def compare_approaches(
    ratings_df: pd.DataFrame,
    items_df: pd.DataFrame,
    sample_users: list,
    k: int = 10,
) -> pd.DataFrame:
    """
    Compare the three recommendation approaches on 5 metrics:
    Precision@K, Recall@K, F1@K, Coverage, Diversity.

    Uses leave-one-out evaluation for P/R/F1 (proper held-out protocol).
    Uses full ratings_df for Coverage and Diversity.

    Representatives:
      CF       → SVD
      Content  → TF-IDF Keywords
      Knowledge → Constraint-Based
    """
    import models.collaborative.svd_model    as svd_mod
    import models.content.tfidf_keywords     as cb_mod
    import models.knowledge.constraint_based as kb_mod
    from models.collaborative.cf_base import get_unrated_items
    from utils.result_format import format_results

    all_ids = items_df["item_id"].tolist()

    # ── Wrap each approach so it accepts (uid, train_df) ──────
    def make_svd_fn(base_ratings):
        """Trains a fresh SVD on base_ratings and returns a recommend function."""
        algo = svd_mod.SVDAlgo().fit(base_ratings)
        def fn(uid, train_df):
            a2 = svd_mod.SVDAlgo().fit(train_df)
            unrated = get_unrated_items(train_df, items_df, uid)
            scores  = sorted(
                [(iid, a2.predict(uid, iid)) for iid in unrated],
                key=lambda x: x[1], reverse=True
            )[:k]
            return format_results(items_df, scores, "collaborative", "svd",
                                  lambda row: "SVD recommendation")
        return fn, algo

    svd_fn, svd_algo_full = make_svd_fn(ratings_df)

    def cb_fn(uid, train_df):
        return cb_mod.recommend(train_df, items_df, uid, k)

    def kb_fn(uid, train_df):
        return kb_mod.recommend(items_df, train_df, uid, n=k)

    # For coverage/diversity we use full ratings_df (no hold-out needed)
    def svd_full(uid, _df):
        unrated = get_unrated_items(ratings_df, items_df, uid)
        scores  = sorted(
            [(iid, svd_algo_full.predict(uid, iid)) for iid in unrated],
            key=lambda x: x[1], reverse=True
        )[:k]
        return format_results(items_df, scores, "collaborative", "svd",
                              lambda row: "SVD recommendation")

    def cb_full(uid, _df):  return cb_mod.recommend(ratings_df, items_df, uid, k)
    def kb_full(uid, _df):  return kb_mod.recommend(items_df, ratings_df, uid, n=k)

    configs = [
        ("Collaborative (SVD)",          svd_fn,  svd_full),
        ("Content-Based (TF-IDF KW)",    cb_fn,   cb_full),
        ("Knowledge-Based (Constraint)", kb_fn,   kb_full),
    ]

    rows = []
    for label, loo_fn, full_fn in configs:
        print(f"    Evaluating {label}…")

        # Leave-one-out P/R/F1
        rm  = evaluate_ranking(loo_fn, ratings_df, items_df, k=k,
                               n_users=min(20, len(sample_users)))
        # Coverage & diversity use full ratings
        cov = catalog_coverage(full_fn, all_ids, sample_users[:20], ratings_df, k=k)
        try:
            div = intra_list_diversity(full_fn(sample_users[0], ratings_df))
        except Exception:
            div = 0.0

        rows.append({
            "Approach":       label,
            f"Precision@{k}": rm.get(f"precision@{k}", 0.0),
            f"Recall@{k}":    rm.get(f"recall@{k}",    0.0),
            f"F1@{k}":        rm.get(f"f1@{k}",        0.0),
            "Coverage":       cov,
            "Diversity":      div,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
#  FULL EVALUATION PIPELINE  (Streamlit entry point)
# ══════════════════════════════════════════════════════════════

def run_full_evaluation(
    ratings_df: pd.DataFrame,
    items_df: pd.DataFrame,
    cv: int = 3,
) -> Dict:
    """
    Master entry point: runs CF cross-validation + approach comparison.
    Returns a dict used by the Streamlit evaluation dashboard.
    """
    sample_users = ratings_df["user_id"].unique()[:30].tolist()

    print("  [1/2] Running CF method cross-validation…")
    cf_comparison = compare_cf_methods(ratings_df, n_splits=cv)

    print("  [2/2] Running approach comparison (leave-one-out)…")
    approach_comparison = compare_approaches(
        ratings_df, items_df, sample_users, k=10
    )

    return {
        "cf_comparison":       cf_comparison,
        "approach_comparison": approach_comparison,
        "sample_users":        sample_users,
    }
