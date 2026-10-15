from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from recs import api
from recs.artifacts import build_artifacts


@pytest.fixture()
def client(tmp_path, ratings, movies):
    build_artifacts(ratings, movies, tmp_path)
    os.environ["RECS_ARTIFACTS"] = str(tmp_path)
    api.store = None
    with TestClient(api.app) as test_client:
        yield test_client
    api.store = None


def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["users"] == 5
    assert body["ratings"] == 20


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Recommendation Engine" in response.text


def test_recommend_for_user(client):
    response = client.get("/recommend/user/1", params={"n": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["userId"] == 1
    recs = body["recommendations"]
    assert len(recs) == 3
    assert all(rec["movieId"] not in {101, 102, 103, 105} for rec in recs)
    assert all(rec["title"] for rec in recs)


def test_recommend_unknown_user_404(client):
    response = client.get("/recommend/user/999")
    assert response.status_code == 404
    assert "not in the training data" in response.json()["detail"]


def test_similar_movies(client):
    response = client.get("/recommend/similar/101", params={"n": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["movie"]["movieId"] == 101
    assert all(item["movieId"] != 101 for item in body["similar"])


def test_similar_unknown_movie_404(client):
    assert client.get("/recommend/similar/999").status_code == 404


def test_search_substring(client):
    response = client.get("/search", params={"title": "alpha"})
    assert response.status_code == 200
    results = response.json()["results"]
    assert [r["movieId"] for r in results] == [101]


def test_search_no_match(client):
    response = client.get("/search", params={"title": "zzzzqqq"})
    assert response.status_code == 200
    assert response.json()["results"] == []


def test_new_user_cold_start(client):
    response = client.post(
        "/recommend/new-user",
        json={
            "ratings": [{"movieId": 101, "rating": 5.0}, {"movieId": 103, "rating": 1.0}],
            "n": 5,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["recommendations"][0]["movieId"] == 102
    assert body["ignoredMovieIds"] == []


def test_new_user_unknown_movies_422(client):
    response = client.post(
        "/recommend/new-user", json={"ratings": [{"movieId": 999, "rating": 5.0}]}
    )
    assert response.status_code == 422


def test_new_user_bad_rating_422(client):
    response = client.post(
        "/recommend/new-user", json={"ratings": [{"movieId": 101, "rating": 9.0}]}
    )
    assert response.status_code == 422
