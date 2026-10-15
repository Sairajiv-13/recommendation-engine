from __future__ import annotations

from recs.data import load_movies, load_ratings, relevant_items, split_per_user


def test_load_movies_parses_genres(tmp_path):
    path = tmp_path / "movies.csv"
    path.write_text(
        "movieId,title,genres\n"
        "1,Toy Story (1995),Adventure|Animation|Comedy\n"
        "2,Unknown Title,(no genres listed)\n"
    )
    movies = load_movies(str(path))
    assert movies.loc[0, "genre_list"] == ["Adventure", "Animation", "Comedy"]
    assert movies.loc[1, "genre_list"] == []


def test_load_ratings_columns(tmp_path):
    path = tmp_path / "ratings.csv"
    path.write_text("userId,movieId,rating,timestamp\n1,1,4.5,964982703\n")
    ratings = load_ratings(str(path))
    assert list(ratings.columns) == ["userId", "movieId", "rating", "timestamp"]
    assert ratings.loc[0, "rating"] == 4.5


def test_split_is_per_user_and_disjoint(ratings):
    bigger = ratings.copy()
    for extra in range(6, 10):
        bigger.loc[len(bigger)] = [1, 100 + extra, 3.0, 1_700_000_000]
    train, test = split_per_user(bigger, min_ratings=5)
    assert len(train) + len(test) == len(bigger)
    train_keys = set(zip(train["userId"], train["movieId"], strict=True))
    test_keys = set(zip(test["userId"], test["movieId"], strict=True))
    assert train_keys.isdisjoint(test_keys)
    assert set(test["userId"]) == {1}


def test_split_keeps_thin_users_in_train(ratings):
    train, test = split_per_user(ratings, min_ratings=10)
    assert len(test) == 0
    assert len(train) == len(ratings)


def test_relevant_items_threshold(ratings):
    relevant = relevant_items(ratings, threshold=4.0)
    assert relevant[1] == {101, 102}
    assert relevant[3] == {103, 104, 106}
