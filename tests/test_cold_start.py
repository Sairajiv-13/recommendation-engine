from __future__ import annotations

import pytest

from recs.models import GenreProfileRecommender


def test_genre_profile_prefers_liked_genre(ratings, movies):
    model = GenreProfileRecommender().fit(ratings, movies)
    # Loved an Action movie, disliked a Comedy: Action affinity is +2
    # around the mean of 3.0, Comedy is -2.
    recs = model.recommend_from_ratings([(101, 5.0), (103, 1.0)], n=6)
    ids = [movie_id for movie_id, _ in recs]
    assert ids[0] == 102  # unseen Action, most popular of the Action pair
    assert ids.index(108) < ids.index(104)  # any Action beats pure Comedy
    assert 101 not in ids and 103 not in ids  # already rated, never repeated


def test_genre_profile_returns_requested_count(ratings, movies):
    model = GenreProfileRecommender().fit(ratings, movies)
    recs = model.recommend_from_ratings([(105, 5.0)], n=3)
    assert len(recs) == 3
    assert recs[0][0] == 107  # the other Drama


def test_genre_profile_needs_at_least_one_rating(ratings, movies):
    model = GenreProfileRecommender().fit(ratings, movies)
    with pytest.raises(ValueError):
        model.recommend_from_ratings([], n=5)
