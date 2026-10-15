from __future__ import annotations

import math

import pytest

from recs.models import ItemItemRecommender, PopularityRecommender, SVDRecommender


def test_popularity_order_and_exclusions(ratings):
    model = PopularityRecommender().fit(ratings)
    recs = model.recommend(2, n=10)
    ids = [movie_id for movie_id, _ in recs]
    # Ranking by count, then mean rating, then id is
    # 101, 103, 105, 102, 106, 107, 104, 108. User 2 has seen all but these.
    assert ids == [103, 105, 107, 104]
    assert not set(ids) & model.rm_.seen_movies(2)


def test_popularity_is_same_list_for_everyone(ratings):
    model = PopularityRecommender().fit(ratings)
    first = model.recommend(1, n=2)
    assert [m for m, _ in first] == [106, 107]


def test_item_item_similar_excludes_self(ratings):
    model = ItemItemRecommender().fit(ratings)
    similar = model.similar_movies(101, n=5)
    ids = [movie_id for movie_id, _ in similar]
    assert 101 not in ids
    assert len(ids) == 5
    assert all(score > 0 for _, score in similar)
    assert [s for _, s in similar] == sorted((s for _, s in similar), reverse=True)


def test_item_item_recommend_shape(ratings):
    model = ItemItemRecommender().fit(ratings)
    recs = model.recommend(1, n=3)
    assert len(recs) == 3
    assert not {m for m, _ in recs} & model.rm_.seen_movies(1)


def test_svd_recommend_shape_and_range(ratings):
    model = SVDRecommender(n_components=4).fit(ratings)
    recs = model.recommend(3, n=4)
    assert len(recs) == 4
    assert not {m for m, _ in recs} & model.rm_.seen_movies(3)
    assert all(math.isfinite(score) for _, score in recs)
    assert 0.0 <= model.explained_variance_ <= 1.0


def test_unknown_user_raises(ratings):
    model = PopularityRecommender().fit(ratings)
    with pytest.raises(KeyError):
        model.recommend(999)
