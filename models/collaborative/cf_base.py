"""
models/collaborative/cf_base.py
================================
Pure NumPy/SciPy collaborative filtering utilities.
No external CF library dependency — fully transparent, fully portable.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def build_user_item_matrix(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot ratings DataFrame into user x item matrix. NaN = not rated."""
    return ratings_df.pivot_table(index="user_id", columns="item_id", values="rating")


def get_rated_items(ratings_df: pd.DataFrame, user_id: int) -> set:
    return set(ratings_df[ratings_df["user_id"] == user_id]["item_id"].tolist())


def get_unrated_items(ratings_df: pd.DataFrame, items_df: pd.DataFrame, user_id: int) -> list:
    rated = get_rated_items(ratings_df, user_id)
    return [iid for iid in items_df["item_id"].tolist() if iid not in rated]


def cross_validate_cf(algo_class, ratings_df: pd.DataFrame, n_splits: int = 3) -> dict:
    """K-fold cross-validation for any CF algo."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    indices = np.arange(len(ratings_df))
    rmses, maes = [], []
    for train_idx, test_idx in kf.split(indices):
        train_df = ratings_df.iloc[train_idx].copy()
        test_df  = ratings_df.iloc[test_idx].copy()
        algo = algo_class()
        algo.fit(train_df)
        preds, actuals = [], []
        for _, row in test_df.iterrows():
            try:
                p = algo.predict(int(row["user_id"]), int(row["item_id"]))
                preds.append(p)
                actuals.append(row["rating"])
            except Exception:
                pass
        if preds:
            p_arr = np.array(preds); a_arr = np.array(actuals)
            rmses.append(np.sqrt(np.mean((p_arr - a_arr) ** 2)))
            maes.append(np.mean(np.abs(p_arr - a_arr)))
    return {
        "rmse_mean": round(float(np.mean(rmses)) if rmses else 0.0, 4),
        "rmse_std":  round(float(np.std(rmses))  if rmses else 0.0, 4),
        "mae_mean":  round(float(np.mean(maes))  if maes else 0.0, 4),
        "mae_std":   round(float(np.std(maes))   if maes else 0.0, 4),
    }
