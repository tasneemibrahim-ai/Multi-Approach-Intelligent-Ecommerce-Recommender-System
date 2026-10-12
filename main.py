"""
main.py
=========
Intelligent E-Commerce Recommendation System
Course: AIE425 — Alamein International University

Master integration file. Ties together ALL modules:
  - Data loading (utils/data_loader.py)
  - Collaborative Filtering (6 methods)
  - Content-Based Filtering (3 TF-IDF modes)
  - Knowledge-Based Filtering (3 approaches)
  - Evaluation (RMSE, MAE, Precision@K, Recall@K, F1@K, Coverage, Diversity)

Each block is independent. Comment out any block without breaking the system.

Usage:
  python main.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
#  DATA LOADING
#  Change source='movielens' or 'amazon' to swap datasets
# ══════════════════════════════════════════════════════════════
from utils.data_loader import load_dataset

print("=" * 60)
print("  Intelligent E-Commerce Recommendation System")
print("  AIE425 — Alamein International University")
print("=" * 60)
print("\n[DATA] Loading dataset...")

dataset = load_dataset(source="synthetic")   # ← Change source here
ratings_df = dataset["ratings"]
items_df   = dataset["items"]
users_df   = dataset["users"]

print(f"  Users:   {ratings_df['user_id'].nunique()}")
print(f"  Items:   {items_df['item_id'].nunique()}")
print(f"  Ratings: {len(ratings_df)}")

# Demo user for all recommendation examples
DEMO_USER = 0
TOP_N = 5

print(f"\n[DEMO] Generating recommendations for User #{DEMO_USER}")
print(f"[DEMO] Top-{TOP_N} per method\n")


# ══════════════════════════════════════════════════════════════
#  ===== COLLABORATIVE FILTERING =====
#  6 independent CF algorithms. Remove any without side effects.
# ══════════════════════════════════════════════════════════════

print("━" * 60)
print("  BLOCK 1: COLLABORATIVE FILTERING")
print("━" * 60)

# ── CF Method 1: User-Based KNN ───────────────────────────────
print("\n[CF-1] User-Based KNN...")
from models.collaborative import user_knn
cf_user_knn_recs = user_knn.recommend(ratings_df, items_df, DEMO_USER, n=TOP_N)
print(f"  → {len(cf_user_knn_recs)} recommendations")
for r in cf_user_knn_recs:
    print(f"    [{r.predicted_score:.2f}] {r.item_name} | {r.explanation[:70]}...")

# ── CF Method 2: Item-Based KNN ───────────────────────────────
print("\n[CF-2] Item-Based KNN...")
from models.collaborative import item_knn
cf_item_knn_recs = item_knn.recommend(ratings_df, items_df, DEMO_USER, n=TOP_N)
print(f"  → {len(cf_item_knn_recs)} recommendations")
for r in cf_item_knn_recs:
    print(f"    [{r.predicted_score:.2f}] {r.item_name} | {r.explanation[:70]}...")

# ── CF Method 3: SVD ──────────────────────────────────────────
print("\n[CF-3] SVD (Matrix Factorization)...")
from models.collaborative import svd_model
cf_svd_recs = svd_model.recommend(ratings_df, items_df, DEMO_USER, n=TOP_N)
print(f"  → {len(cf_svd_recs)} recommendations")
for r in cf_svd_recs:
    print(f"    [{r.predicted_score:.2f}] {r.item_name} | {r.explanation[:70]}...")

# ── CF Method 4: SVD++ ────────────────────────────────────────
print("\n[CF-4] SVD++ (with Implicit Feedback)...")
from models.collaborative import svdpp_model
cf_svdpp_recs = svdpp_model.recommend(ratings_df, items_df, DEMO_USER, n=TOP_N)
print(f"  → {len(cf_svdpp_recs)} recommendations")
for r in cf_svdpp_recs:
    print(f"    [{r.predicted_score:.2f}] {r.item_name} | {r.explanation[:70]}...")

# ── CF Method 5: KNN with Means ───────────────────────────────
print("\n[CF-5] KNN with Means (Bias-Corrected)...")
from models.collaborative import knn_means
cf_knn_means_recs = knn_means.recommend(ratings_df, items_df, DEMO_USER, n=TOP_N)
print(f"  → {len(cf_knn_means_recs)} recommendations")
for r in cf_knn_means_recs:
    print(f"    [{r.predicted_score:.2f}] {r.item_name} | {r.explanation[:70]}...")

# ── CF Method 6: NMF ──────────────────────────────────────────
print("\n[CF-6] NMF (Non-Negative Matrix Factorization)...")
from models.collaborative import nmf_model
cf_nmf_recs = nmf_model.recommend(ratings_df, items_df, DEMO_USER, n=TOP_N)
print(f"  → {len(cf_nmf_recs)} recommendations")
for r in cf_nmf_recs:
    print(f"    [{r.predicted_score:.2f}] {r.item_name} | {r.explanation[:70]}...")


# ══════════════════════════════════════════════════════════════
#  ===== CONTENT-BASED FILTERING =====
#  3 TF-IDF modes. Remove any without side effects.
# ══════════════════════════════════════════════════════════════

print("\n" + "━" * 60)
print("  BLOCK 2: CONTENT-BASED FILTERING")
print("━" * 60)

# ── Content Method 1: TF-IDF Genre ────────────────────────────
print("\n[CB-1] TF-IDF Genre/Category Similarity...")
from models.content import tfidf_genre
cb_genre_recs = tfidf_genre.recommend(ratings_df, items_df, DEMO_USER, n=TOP_N)
print(f"  → {len(cb_genre_recs)} recommendations")
for r in cb_genre_recs:
    print(f"    [{r.predicted_score:.4f}] {r.item_name} | {r.explanation[:70]}...")

# ── Content Method 2: TF-IDF Keywords ────────────────────────
print("\n[CB-2] TF-IDF Keyword/Tag Similarity...")
from models.content import tfidf_keywords
cb_kw_recs = tfidf_keywords.recommend(ratings_df, items_df, DEMO_USER, n=TOP_N)
print(f"  → {len(cb_kw_recs)} recommendations")
for r in cb_kw_recs:
    print(f"    [{r.predicted_score:.4f}] {r.item_name} | {r.explanation[:70]}...")

# ── Content Method 3: TF-IDF Full Description ────────────────
print("\n[CB-3] TF-IDF Full Description Similarity...")
from models.content import tfidf_description
cb_desc_recs = tfidf_description.recommend(ratings_df, items_df, DEMO_USER, n=TOP_N)
print(f"  → {len(cb_desc_recs)} recommendations")
for r in cb_desc_recs:
    print(f"    [{r.predicted_score:.4f}] {r.item_name} | {r.explanation[:70]}...")


# ══════════════════════════════════════════════════════════════
#  ===== KNOWLEDGE-BASED FILTERING =====
#  3 independent approaches. Remove any without side effects.
# ══════════════════════════════════════════════════════════════

print("\n" + "━" * 60)
print("  BLOCK 3: KNOWLEDGE-BASED FILTERING")
print("━" * 60)

# ── Knowledge Method 1: Simple Filter ────────────────────────
print("\n[KB-1] Simple Constraint Filtering (category=Electronics, max=$150)...")
from models.knowledge import simple_filter
kb_simple_recs = simple_filter.recommend(
    items_df, ratings_df, DEMO_USER,
    category="Electronics",
    max_price=150.0,
    n=TOP_N
)
print(f"  → {len(kb_simple_recs)} recommendations")
for r in kb_simple_recs:
    print(f"    [{r.predicted_score:.2f}] {r.item_name} | {r.explanation[:70]}...")

# ── Knowledge Method 2: Rule-Based ────────────────────────────
print("\n[KB-2] Rule-Based System...")
from models.knowledge import rule_based
kb_rule_recs = rule_based.recommend(
    items_df, ratings_df, DEMO_USER,
    budget=200.0,
    n=TOP_N
)
print(f"  → {len(kb_rule_recs)} recommendations")
for r in kb_rule_recs:
    print(f"    [{r.predicted_score:.2f}] {r.item_name} | {r.explanation[:70]}...")

# ── Knowledge Method 3: Constraint-Based Weighted Scoring ─────
print("\n[KB-3] Constraint-Based Weighted Scoring...")
from models.knowledge import constraint_based
kb_constraint_recs = constraint_based.recommend(
    items_df, ratings_df, DEMO_USER,
    preferred_category="Electronics",
    max_budget=150.0,
    preferred_brand=None,
    min_rating=3.0,
    n=TOP_N
)
print(f"  → {len(kb_constraint_recs)} recommendations")
for r in kb_constraint_recs:
    print(f"    [{r.predicted_score:.2f}] {r.item_name} | {r.explanation[:70]}...")


# ══════════════════════════════════════════════════════════════
#  ===== EVALUATION =====
#  Quick evaluation summary (full eval runs in Streamlit dashboard)
# ══════════════════════════════════════════════════════════════

print("\n" + "━" * 60)
print("  BLOCK 4: EVALUATION SUMMARY")
print("━" * 60)

from evaluation.evaluator import intra_list_diversity, precision_at_k, recall_at_k, f1_at_k

# Quick diversity comparison
print("\n[EVAL] Intra-List Diversity (higher = more diverse recommendations):")
diversity_results = {
    "CF User-KNN":      intra_list_diversity(cf_user_knn_recs),
    "CF SVD":           intra_list_diversity(cf_svd_recs),
    "CF NMF":           intra_list_diversity(cf_nmf_recs),
    "Content Genre":    intra_list_diversity(cb_genre_recs),
    "Content Keywords": intra_list_diversity(cb_kw_recs),
    "Knowledge Filter": intra_list_diversity(kb_simple_recs),
    "Knowledge Rules":  intra_list_diversity(kb_rule_recs),
    "Knowledge Constr.":intra_list_diversity(kb_constraint_recs),
}
for method, score in diversity_results.items():
    bar = "█" * int(score * 20)
    print(f"  {method:<22}: {score:.3f}  {bar}")

print("\n[DONE] Run `streamlit run app/streamlit_app.py` for the full interactive dashboard.")
print("       Or set RUN_FULL_EVAL=1 for complete cross-validation metrics.")

# Full evaluation — set env var to run (slow, takes ~2 min)
if os.environ.get("RUN_FULL_EVAL") == "1":
    from evaluation.evaluator import run_full_evaluation
    print("\n[EVAL] Running full cross-validation (this may take a few minutes)...")
    eval_results = run_full_evaluation(ratings_df, items_df, cv=3)
    print("\n[EVAL] CF Method Comparison:")
    print(eval_results["cf_comparison"].to_string(index=False))
    print("\n[EVAL] Approach Comparison:")
    print(eval_results["approach_comparison"].to_string(index=False))
