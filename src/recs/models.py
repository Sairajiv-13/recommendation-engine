"""Three recommenders plus a cold-start fallback.

Every model shares the same small interface: ``fit`` on a training ratings
frame, then ``recommend(user_id, n)`` returning ``(movie_id, score)`` pairs,
best first, with items the user already rated excluded. Scores are only
comparable within one model's output, not across models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from recs.matrix import RatingMatrix


def _top_n(scores: np.ndarray, exclude: set[int], n: int) -> list[tuple[int, float]]:
    order = np.argsort(-scores)
    picked: list[tuple[int, float]] = []
    for j in order:
        if int(j) in exclude:
            continue
        picked.append((int(j), float(scores[j])))
        if len(picked) == n:
            break
    return picked


class PopularityRecommender:
    """Ranks movies by how many training ratings they received.

    Ties break on mean rating, then on movie id, so the ranking is stable.
    The same list goes to every user, minus what they have already seen.
    """

    name = "popularity"

    def fit(self, ratings: pd.DataFrame) -> PopularityRecommender:
        self.rm_ = RatingMatrix(ratings)
        stats = ratings.groupby("movieId")["rating"].agg(["count", "mean"]).reset_index()
        order = stats.sort_values(["count", "mean", "movieId"], ascending=[False, False, True])[
            "movieId"
        ]
        self.ranking_: list[int] = [int(m) for m in order]
        self.counts_ = {
            int(m): int(c) for m, c in zip(stats["movieId"], stats["count"], strict=True)
        }
        return self

    def recommend(self, user_id: int, n: int = 10) -> list[tuple[int, float]]:
        seen = self.rm_.seen_movies(user_id)
        picked = []
        for movie_id in self.ranking_:
            if movie_id in seen:
                continue
            picked.append((movie_id, float(self.counts_[movie_id])))
            if len(picked) == n:
                break
        return picked


class ItemItemRecommender:
    """Item-item cosine similarity over the rating matrix.

    The full similarity matrix is too dense to keep, so each movie keeps
    only its ``neighbors`` most similar movies. A user's score for a movie
    is the similarity-weighted sum of their ratings for its neighbors.
    """

    name = "item_item"

    def __init__(self, neighbors: int = 50):
        self.neighbors = neighbors

    def fit(self, ratings: pd.DataFrame) -> ItemItemRecommender:
        self.rm_ = RatingMatrix(ratings)
        matrix = self.rm_.matrix.tocsc()
        norms = np.sqrt(matrix.multiply(matrix).sum(axis=0)).A1
        norms[norms == 0.0] = 1.0
        normalized = matrix @ sparse.diags(1.0 / norms)
        similarity = (normalized.T @ normalized).tocsr()
        similarity.setdiag(0.0)
        similarity.eliminate_zeros()
        self.similarity_ = _prune_rows(similarity, self.neighbors)
        return self

    def similar_movies(self, movie_id: int, n: int = 10) -> list[tuple[int, float]]:
        col = self.rm_.movie_index[movie_id]
        row = self.similarity_.getrow(col)
        pairs = [
            (self.rm_.movie_ids[j], float(v))
            for j, v in zip(row.indices, row.data, strict=True)
            if self.rm_.movie_ids[j] != movie_id
        ]
        pairs.sort(key=lambda pair: -pair[1])
        return pairs[:n]

    def recommend(self, user_id: int, n: int = 10) -> list[tuple[int, float]]:
        row = self.rm_.matrix.getrow(self.rm_.user_index[user_id])
        scores = (row @ self.similarity_).toarray().ravel()
        seen_idx = {self.rm_.movie_index[m] for m in self.rm_.seen_movies(user_id)}
        picked = _top_n(scores, seen_idx, n)
        return [(self.rm_.movie_ids[j], score) for j, score in picked]


def _prune_rows(matrix: sparse.csr_matrix, keep: int) -> sparse.csr_matrix:
    """Keep only the ``keep`` largest entries in each row of a CSR matrix."""
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for i in range(matrix.shape[0]):
        row = matrix.getrow(i)
        if row.nnz <= keep:
            chosen = list(zip(row.indices, row.data, strict=True))
        else:
            top = np.argpartition(-row.data, keep)[:keep]
            chosen = [(int(row.indices[t]), float(row.data[t])) for t in top]
        for j, v in chosen:
            if v > 0.0:
                rows.append(i)
                cols.append(int(j))
                data.append(float(v))
    return sparse.csr_matrix((data, (rows, cols)), shape=matrix.shape)


class SVDRecommender:
    """Matrix factorization with TruncatedSVD on user-mean-centered ratings.

    Unrated entries are zero after centering, so the factorization learns
    deviations from each user's average. Predicted scores add the user
    mean back and read as approximate star ratings.
    """

    name = "svd"

    def __init__(self, n_components: int = 64):
        self.n_components = n_components

    def fit(self, ratings: pd.DataFrame) -> SVDRecommender:
        self.rm_ = RatingMatrix(ratings)
        dense = self.rm_.matrix.toarray()
        rated = dense > 0
        counts = rated.sum(axis=1)
        counts[counts == 0] = 1
        self.user_means_ = dense.sum(axis=1) / counts
        centered = np.where(rated, dense - self.user_means_[:, None], 0.0)
        n_components = min(self.n_components, min(centered.shape) - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.predicted_ = svd.fit_transform(centered) @ svd.components_ + self.user_means_[:, None]
        self.explained_variance_ = float(svd.explained_variance_ratio_.sum())
        return self

    def recommend(self, user_id: int, n: int = 10) -> list[tuple[int, float]]:
        scores = self.predicted_[self.rm_.user_index[user_id]]
        seen_idx = {self.rm_.movie_index[m] for m in self.rm_.seen_movies(user_id)}
        picked = _top_n(scores, seen_idx, n)
        return [(self.rm_.movie_ids[j], score) for j, score in picked]


class GenreProfileRecommender:
    """Cold start: recommend from a handful of rated movies, no user history.

    The new user's ratings become a per-genre affinity: their mean rating
    for each genre minus 3.0, the neutral point of the 0.5 to 5.0 scale.
    Candidate movies score by the mean affinity over their genres,
    tie-broken by global popularity so the output stays sane even when
    every affinity is zero.
    """

    NEUTRAL_RATING = 3.0

    name = "genre_profile"

    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame) -> GenreProfileRecommender:
        self.genres_by_movie_: dict[int, list[str]] = {
            int(m): list(g) for m, g in zip(movies["movieId"], movies["genre_list"], strict=True)
        }
        counts = ratings.groupby("movieId").size()
        self.popularity_: dict[int, int] = {int(m): int(c) for m, c in counts.items()}
        self.catalog_: list[int] = sorted(self.genres_by_movie_)
        return self

    def recommend_from_ratings(
        self, rated: list[tuple[int, float]], n: int = 10
    ) -> list[tuple[int, float]]:
        if not rated:
            raise ValueError("at least one rated movie is required")
        rated_ids = {movie_id for movie_id, _ in rated}
        per_genre: dict[str, list[float]] = {}
        for movie_id, rating in rated:
            for genre in self.genres_by_movie_.get(movie_id, []):
                per_genre.setdefault(genre, []).append(rating - self.NEUTRAL_RATING)
        affinity = {genre: float(np.mean(vals)) for genre, vals in per_genre.items()}
        scored: list[tuple[int, float]] = []
        for movie_id in self.catalog_:
            if movie_id in rated_ids:
                continue
            genres = self.genres_by_movie_[movie_id]
            if not genres:
                continue
            score = float(np.mean([affinity.get(g, 0.0) for g in genres]))
            scored.append((movie_id, score))
        scored.sort(key=lambda pair: (-pair[1], -self.popularity_.get(pair[0], 0), pair[0]))
        return scored[:n]
