"""Ranking metrics for top-k recommendation lists.

Relevance is binary: an item is either in the user's held-out liked set or
it is not. All functions take the ranked item ids (best first) and the set
of relevant item ids.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def precision_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for item in ranked[:k] if item in relevant)
    return hits / k


def recall_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for item in ranked[:k] if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    """Normalized discounted cumulative gain with binary relevance.

    A ranking that puts every relevant item first, in any order, scores
    exactly 1.0.
    """
    dcg = 0.0
    for position, item in enumerate(ranked[:k]):
        if item in relevant:
            dcg += 1.0 / math.log2(position + 2)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(position + 2) for position in range(ideal_hits))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def catalog_coverage(ranked_lists: Iterable[Sequence[int]], catalog_size: int, k: int) -> float:
    """Fraction of the catalog that appears in at least one top-k list."""
    if catalog_size <= 0:
        return 0.0
    seen: set[int] = set()
    for ranked in ranked_lists:
        seen.update(ranked[:k])
    return len(seen) / catalog_size
