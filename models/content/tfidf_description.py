"""
models/content/tfidf_description.py
=====================================
Content-Based Filtering using TF-IDF on product DESCRIPTIONS.

Logic: Full product descriptions contain rich semantic information.
TF-IDF on descriptions captures nuanced product characteristics that
short keywords or categories might miss.

Text preprocessing pipeline:
  1. Lowercase
  2. Remove punctuation
  3. Remove stopwords
  4. TF-IDF vectorization with sublinear TF scaling

Use case: Best when descriptions are detailed and well-written.
Captures latent topic similarity (e.g., two products both mentioning
"noise cancellation" without sharing the same category).
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from .content_base import preprocess_text, compute_cosine_sim, aggregate_scores_for_user
from utils.result_format import format_results, RecommendationResult
from typing import List


METHOD_NAME = "tfidf_description"
APPROACH = "content"


def build_similarity_matrix(items_df: pd.DataFrame):
    """
    Build TF-IDF cosine similarity matrix using full product descriptions.
    Combines name + keywords + description for maximum coverage.
    """
    def combine_text(row):
        parts = [
            str(row.get("name", "")),
            str(row.get("keywords", "")),
            str(row.get("description", "")),
        ]
        return preprocess_text(" ".join(parts))

    corpus = items_df.apply(combine_text, axis=1).tolist()

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),       # Capture up to trigrams for richer context
        max_features=1000,
        sublinear_tf=True,        # log(1+tf) dampens frequent terms
        min_df=1
    )
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
    Generate top-N content-based recommendations using description TF-IDF.
    """
    sim_matrix, _ = build_similarity_matrix(items_df)
    scores = aggregate_scores_for_user(ratings_df, items_df, sim_matrix, user_id, n)

    def explain(row):
        desc_preview = str(row.get("description", ""))[:60].rstrip()
        return (
            f"Recommended because the description of '{row['name']}' ('{desc_preview}...') "
            f"closely matches the textual profile of products you enjoy. (TF-IDF Description)"
        )

    return format_results(items_df, scores, APPROACH, METHOD_NAME, explain)
