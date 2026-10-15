"""A sparse user-item rating matrix with id mappings."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


class RatingMatrix:
    """Ratings as a CSR matrix plus user and movie id lookups."""

    def __init__(self, ratings: pd.DataFrame):
        self.user_ids: list[int] = sorted(int(u) for u in ratings["userId"].unique())
        self.movie_ids: list[int] = sorted(int(m) for m in ratings["movieId"].unique())
        self.user_index = {u: i for i, u in enumerate(self.user_ids)}
        self.movie_index = {m: i for i, m in enumerate(self.movie_ids)}
        rows = ratings["userId"].map(self.user_index).to_numpy()
        cols = ratings["movieId"].map(self.movie_index).to_numpy()
        values = ratings["rating"].to_numpy(dtype=np.float64)
        self.matrix = sparse.csr_matrix(
            (values, (rows, cols)), shape=(len(self.user_ids), len(self.movie_ids))
        )

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_movies(self) -> int:
        return len(self.movie_ids)

    def has_user(self, user_id: int) -> bool:
        return user_id in self.user_index

    def has_movie(self, movie_id: int) -> bool:
        return movie_id in self.movie_index

    def seen_movies(self, user_id: int) -> set[int]:
        """Movie ids the user has rated (present in this matrix)."""
        row = self.matrix.getrow(self.user_index[user_id])
        return {self.movie_ids[j] for j in row.indices}

    def user_ratings(self, user_id: int) -> dict[int, float]:
        row = self.matrix.getrow(self.user_index[user_id])
        return {self.movie_ids[j]: float(v) for j, v in zip(row.indices, row.data, strict=True)}
