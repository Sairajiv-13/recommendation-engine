"""Evaluate every model on the same per-user holdout protocol.

Each user in the test set is scored against the items they rated 4.0 or
higher in the holdout. Users with no liked holdout items cannot produce a
meaningful ranking score and are left out of the averages; the count of
evaluated users is part of the result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recs.data import relevant_items
from recs.metrics import catalog_coverage, ndcg_at_k, precision_at_k, recall_at_k

K = 10


def evaluate_model(model, test: pd.DataFrame, catalog_size: int, k: int = K) -> dict:
    relevant = relevant_items(test)
    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []
    lists: list[list[int]] = []
    for user_id, liked in relevant.items():
        try:
            recs = model.recommend(user_id, n=k)
        except KeyError:
            continue
        ranked = [movie_id for movie_id, _ in recs]
        lists.append(ranked)
        precisions.append(precision_at_k(ranked, liked, k))
        recalls.append(recall_at_k(ranked, liked, k))
        ndcgs.append(ndcg_at_k(ranked, liked, k))
    return {
        "precision_at_10": float(np.mean(precisions)) if precisions else 0.0,
        "recall_at_10": float(np.mean(recalls)) if recalls else 0.0,
        "ndcg_at_10": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "coverage": catalog_coverage(lists, catalog_size, k),
        "users_evaluated": len(precisions),
    }
