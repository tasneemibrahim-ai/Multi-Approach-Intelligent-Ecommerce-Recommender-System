"""
models/content/tfidf_genre.py
================================
Content-Based Filtering using TF-IDF on product CATEGORIES/GENRES.

Logic: Build a TF-IDF matrix from each product's category field.
Compute cosine similarity between all products.
For a given user, aggregate similarities across their rated items
(weighted by their ratings) to score unrated items.

Use case: Works well when category structure is rich and meaningful.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from .content_base import preprocess_text, compute_cosine_sim, aggregate_scores_for_user
from utils.result_format import format_results, RecommendationResult
from typing import List


METHOD_NAME = "tfidf_genre"
APPROACH = "content"


def build_similarity_matrix(items_df: pd.DataFrame):
    """
    Build TF-IDF cosine similarity matrix using the 'category' field.
    """
    # Preprocess genre/category text
    corpus = items_df["category"].fillna("").apply(preprocess_text).tolist()

    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(corpus)
    sim_matrix = compute_cosine_sim(tfidf_matrix)
    return sim_matrix, vectorizer


def recommend(
    ratings_df: pd.DataFrame,
    items_df: pd.DataFrame,
    user_id: int,
    n: int = 10
) -> List[RecommendationResult]:
    """
    Generate top-N content-based recommendations using genre/category TF-IDF.
    """
    sim_matrix, _ = build_similarity_matrix(items_df)
    scores = aggregate_scores_for_user(ratings_df, items_df, sim_matrix, user_id, n)

    def explain(row):
        return (
            f"Recommended because '{row['name']}' belongs to the '{row['category']}' category, "
            f"which matches categories you have shown interest in. (TF-IDF Genre)"
        )

    return format_results(items_df, scores, APPROACH, METHOD_NAME, explain)
