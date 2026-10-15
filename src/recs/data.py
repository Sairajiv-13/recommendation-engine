"""Loading and splitting MovieLens data.

The split used for evaluation is a per-user random holdout: every user with
enough ratings contributes the same fraction of their ratings to the test
set, so each user can be scored against items they actually rated.
"""

from __future__ import annotations

import pandas as pd

RELEVANCE_THRESHOLD = 4.0


def load_ratings(path: str) -> pd.DataFrame:
    ratings = pd.read_csv(
        path,
        dtype={"userId": "int64", "movieId": "int64", "rating": "float64"},
    )
    return ratings[["userId", "movieId", "rating", "timestamp"]]


def _split_genres(value: str) -> list[str]:
    if not value or value == "(no genres listed)":
        return []
    return value.split("|")


def load_movies(path: str) -> pd.DataFrame:
    movies = pd.read_csv(path, dtype={"movieId": "int64", "title": "string"})
    movies["genre_list"] = movies["genres"].fillna("").map(_split_genres)
    return movies[["movieId", "title", "genres", "genre_list"]]


def all_genres(movies: pd.DataFrame) -> list[str]:
    seen: list[str] = []
    for genres in movies["genre_list"]:
        for genre in genres:
            if genre not in seen:
                seen.append(genre)
    return sorted(seen)


def split_per_user(
    ratings: pd.DataFrame,
    test_frac: float = 0.2,
    seed: int = 42,
    min_ratings: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ratings into train and test with a per-user random holdout.

    Users with fewer than ``min_ratings`` ratings keep everything in train;
    they are too thin to evaluate against. Every other user contributes
    ``test_frac`` of their ratings (at least one row) to the test set.
    """
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, group in ratings.groupby("userId"):
        if len(group) < min_ratings:
            train_parts.append(group)
            continue
        shuffled = group.sample(frac=1.0, random_state=seed)
        n_test = max(1, int(round(len(group) * test_frac)))
        test_parts.append(shuffled.iloc[:n_test])
        train_parts.append(shuffled.iloc[n_test:])
    train = pd.concat(train_parts).reset_index(drop=True)
    test = pd.concat(test_parts).reset_index(drop=True) if test_parts else ratings.iloc[0:0].copy()
    return train, test


def relevant_items(
    test: pd.DataFrame, threshold: float = RELEVANCE_THRESHOLD
) -> dict[int, set[int]]:
    """Map each user to the test items they rated at or above ``threshold``."""
    liked = test[test["rating"] >= threshold]
    result: dict[int, set[int]] = {}
    for user_id, group in liked.groupby("userId"):
        result[int(user_id)] = set(int(m) for m in group["movieId"])
    return result
