"""Build and load the artifacts the API serves.

An artifact directory holds three files:

- ``models.pkl``: the fitted popularity, item-item, SVD, and genre profile
  models. Pickle is used because the artifacts are built locally and
  trusted; never load a pickle from an untrusted source.
- ``movies.json``: movie metadata (title, genres) in row form.
- ``results.json``: evaluation metrics, counts, and the winning model
  name, written by the training run.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd

from recs.data import split_per_user
from recs.evaluate import evaluate_model
from recs.models import (
    GenreProfileRecommender,
    ItemItemRecommender,
    PopularityRecommender,
    SVDRecommender,
)

MODEL_NAMES = ("popularity", "item_item", "svd")


def build_artifacts(ratings: pd.DataFrame, movies: pd.DataFrame, out_dir: str | Path) -> dict:
    """Train all models, evaluate them on a holdout, and write artifacts.

    Returns the results dictionary that is also written to results.json.
    The models the API serves are fitted on the full ratings frame at the
    end, not on the training split, so production recommendations use
    every rating available.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train, test = split_per_user(ratings)
    catalog_size = int(ratings["movieId"].nunique())
    candidates = {
        "popularity": PopularityRecommender(),
        "item_item": ItemItemRecommender(),
        "svd": SVDRecommender(),
    }
    results: dict[str, dict] = {}
    for name, model in candidates.items():
        model.fit(train)
        results[name] = evaluate_model(model, test, catalog_size)
    winner = max(MODEL_NAMES, key=lambda name: results[name]["ndcg_at_10"])

    served = {
        "popularity": PopularityRecommender().fit(ratings),
        "item_item": ItemItemRecommender().fit(ratings),
        "svd": SVDRecommender().fit(ratings),
        "genre_profile": GenreProfileRecommender().fit(ratings, movies),
    }
    summary = {
        "winner": winner,
        "models": results,
        "ratings": int(len(ratings)),
        "users": int(ratings["userId"].nunique()),
        "movies": catalog_size,
        "split": "per-user random holdout, 20 percent test, seed 42",
        "relevance": "held-out rating of 4.0 or higher",
    }
    with open(out / "models.pkl", "wb") as handle:
        pickle.dump(served, handle)
    movies_out = movies[["movieId", "title", "genres"]].copy()
    movies_out.to_json(out / "movies.json", orient="records")
    with open(out / "results.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


class ArtifactStore:
    """Loaded artifacts with the lookups the API needs."""

    def __init__(self, artifact_dir: str | Path):
        self.dir = Path(artifact_dir)
        with open(self.dir / "models.pkl", "rb") as handle:
            self.models: dict = pickle.load(handle)
        with open(self.dir / "results.json") as handle:
            self.results: dict = json.load(handle)
        movies = pd.read_json(self.dir / "movies.json")
        self.movies: dict[int, dict] = {
            int(row.movieId): {"title": row.title, "genres": row.genres}
            for row in movies.itertuples()
        }

    @property
    def winner(self) -> str:
        return str(self.results["winner"])

    def describe(self, movie_id: int, score: float) -> dict:
        meta = self.movies.get(movie_id, {"title": f"Movie {movie_id}", "genres": ""})
        genre_list = [g for g in str(meta["genres"]).split("|") if g and g != "(no genres listed)"]
        return {
            "movieId": movie_id,
            "title": meta["title"],
            "genres": genre_list,
            "score": round(float(score), 4),
        }
