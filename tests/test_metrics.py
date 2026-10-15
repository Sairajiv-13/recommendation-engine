from __future__ import annotations

import math

import pytest

from recs.metrics import catalog_coverage, ndcg_at_k, precision_at_k, recall_at_k


def test_precision_and_recall():
    ranked = [10, 20, 30, 40]
    relevant = {20, 40, 50}
    assert precision_at_k(ranked, relevant, 4) == 0.5
    assert precision_at_k(ranked, relevant, 2) == 0.5
    assert recall_at_k(ranked, relevant, 4) == pytest.approx(2 / 3)


def test_ndcg_perfect_ranking_is_one():
    ranked = [7, 8, 9, 10, 11]
    assert ndcg_at_k(ranked, {7, 8, 9}, 5) == pytest.approx(1.0)
    assert ndcg_at_k(ranked, {7, 8, 9}, 3) == pytest.approx(1.0)


def test_ndcg_hand_computed():
    # Ranked relevance pattern: hit, miss, hit. Two relevant items total.
    # DCG  = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
    # IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.6309...
    ranked = [1, 2, 3]
    relevant = {1, 3}
    expected = 1.5 / (1.0 + 1.0 / math.log2(3))
    assert ndcg_at_k(ranked, relevant, 3) == pytest.approx(expected)


def test_ndcg_no_relevant_is_zero():
    assert ndcg_at_k([1, 2, 3], set(), 3) == 0.0
    assert recall_at_k([1, 2, 3], set(), 3) == 0.0


def test_catalog_coverage():
    lists = [[1, 2, 3], [3, 4, 5]]
    assert catalog_coverage(lists, catalog_size=10, k=3) == pytest.approx(0.5)
    assert catalog_coverage(lists, catalog_size=10, k=1) == pytest.approx(0.2)
