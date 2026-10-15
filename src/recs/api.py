"""FastAPI service over the trained recommendation artifacts.

Run ``python -m recs.train`` first, then serve with
``uvicorn recs.api:app``. Set RECS_ARTIFACTS to point at a different
artifact directory.
"""

from __future__ import annotations

import difflib
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from recs.artifacts import ArtifactStore

STATIC_DIR = Path(__file__).parent / "static"

store: ArtifactStore | None = None


def get_store() -> ArtifactStore:
    global store
    if store is None:
        store = ArtifactStore(os.environ.get("RECS_ARTIFACTS", "artifacts"))
    return store


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_store()
    yield


app = FastAPI(title="Recommendation Engine", version="1.0.0", lifespan=lifespan)


class RatedMovie(BaseModel):
    movie_id: int = Field(alias="movieId")
    rating: float = Field(ge=0.5, le=5.0)


class NewUserRequest(BaseModel):
    ratings: list[RatedMovie] = Field(min_length=1, max_length=100)
    n: int = Field(default=10, ge=1, le=50)


@app.get("/healthz")
def healthz() -> dict:
    loaded = get_store()
    return {
        "status": "ok",
        "serving_model": loaded.winner,
        "users": loaded.results["users"],
        "movies": loaded.results["movies"],
        "ratings": loaded.results["ratings"],
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/recommend/user/{user_id}")
def recommend_for_user(user_id: int, n: int = Query(default=10, ge=1, le=50)) -> dict:
    loaded = get_store()
    model = loaded.models[loaded.winner]
    if not model.rm_.has_user(user_id):
        raise HTTPException(
            status_code=404,
            detail=f"user {user_id} is not in the training data; "
            "use POST /recommend/new-user for cold start",
        )
    recs = model.recommend(user_id, n=n)
    return {
        "userId": user_id,
        "model": loaded.winner,
        "recommendations": [loaded.describe(movie_id, score) for movie_id, score in recs],
    }


@app.get("/recommend/similar/{movie_id}")
def recommend_similar(movie_id: int, n: int = Query(default=10, ge=1, le=50)) -> dict:
    loaded = get_store()
    item_item = loaded.models["item_item"]
    if movie_id not in loaded.movies:
        raise HTTPException(status_code=404, detail=f"movie {movie_id} is not in the catalog")
    if not item_item.rm_.has_movie(movie_id):
        raise HTTPException(
            status_code=404,
            detail=f"movie {movie_id} has no ratings, so no similar movies are known",
        )
    similar = item_item.similar_movies(movie_id, n=n)
    return {
        "movie": loaded.describe(movie_id, 1.0),
        "similar": [loaded.describe(mid, score) for mid, score in similar],
    }


@app.get("/search")
def search(title: str = Query(min_length=1), limit: int = Query(default=10, ge=1, le=25)) -> dict:
    loaded = get_store()
    query = title.strip().lower()
    substring = [
        (movie_id, meta)
        for movie_id, meta in loaded.movies.items()
        if query in meta["title"].lower()
    ]
    if substring:
        substring.sort(key=lambda pair: pair[1]["title"])
        results = substring[:limit]
    else:
        titles = {meta["title"]: movie_id for movie_id, meta in loaded.movies.items()}
        close = difflib.get_close_matches(title, list(titles), n=limit, cutoff=0.6)
        results = [(titles[name], loaded.movies[titles[name]]) for name in close]
    return {
        "query": title,
        "results": [
            {
                "movieId": movie_id,
                "title": meta["title"],
                "genres": loaded.describe(movie_id, 0.0)["genres"],
            }
            for movie_id, meta in results
        ],
    }


@app.post("/recommend/new-user")
def recommend_new_user(payload: NewUserRequest) -> dict:
    loaded = get_store()
    genre_model = loaded.models["genre_profile"]
    known: list[tuple[int, float]] = []
    ignored: list[int] = []
    for entry in payload.ratings:
        if entry.movie_id in loaded.movies:
            known.append((entry.movie_id, entry.rating))
        else:
            ignored.append(entry.movie_id)
    if not known:
        raise HTTPException(
            status_code=422,
            detail=f"none of the rated movies are in the catalog; unknown ids: {ignored}",
        )
    recs = genre_model.recommend_from_ratings(known, n=payload.n)
    return {
        "model": "genre_profile (cold start)",
        "basedOn": len(known),
        "ignoredMovieIds": ignored,
        "recommendations": [loaded.describe(movie_id, score) for movie_id, score in recs],
    }
