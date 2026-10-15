from __future__ import annotations

from recs.matrix import RatingMatrix


def test_matrix_shape_and_ids(ratings):
    rm = RatingMatrix(ratings)
    assert rm.matrix.shape == (5, 8)
    assert rm.n_users == 5
    assert rm.n_movies == 8
    assert rm.has_user(1)
    assert not rm.has_user(999)
    assert rm.has_movie(108)
    assert not rm.has_movie(999)


def test_seen_movies(ratings):
    rm = RatingMatrix(ratings)
    assert rm.seen_movies(1) == {101, 102, 103, 105}
    assert rm.seen_movies(4) == {101, 103, 105, 107}


def test_user_ratings(ratings):
    rm = RatingMatrix(ratings)
    assert rm.user_ratings(2) == {101: 5.0, 102: 5.0, 106: 4.0, 108: 4.0}
