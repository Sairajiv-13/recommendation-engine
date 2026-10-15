# Recommendation Engine

A movie recommendation system that trains three models on the same
per-user holdout, serves the winner behind a FastAPI API, and handles
brand new users with a genre based cold-start path. Training, evaluation,
and serving share one codebase, and the evaluation numbers in this
README come straight from `artifacts/results.json`, written by the
training run.

Live demo: https://recommendation-engine-a8e4.onrender.com (free tier, the first request after idle can take a minute)

## Problem

Given what a person has rated, rank the movies they are most likely to
rate highly next. The interesting parts are the ones tutorials skip:
comparing models on identical held-out data instead of trusting one
number, keeping a baseline honest, and doing something sensible for a
user with no rating history at all.

## Approach

Three models with different inductive biases are fitted and scored on
the same holdout: a popularity baseline, an item-item cosine model, and
a truncated SVD over the centered rating matrix. The winner by NDCG@10
is what the API serves. New users, who have no row in the rating
matrix, go to a separate genre-profile model built from a handful of
ratings they supply. Data is MovieLens latest-small from GroupLens:
100,836 ratings, 610 users, 9,724 movies with at least one rating
(9,742 in the catalog file).

## Results

Evaluation protocol: a per-user random holdout (20 percent of each
user's ratings, seed 42). A held-out item counts as relevant when the
user rated it 4.0 or higher. Metrics are averaged over the 600 users
who had at least one relevant held-out item. Catalog coverage is the
share of rated movies that appear in at least one user's top 10.

| Model | Precision@10 | Recall@10 | NDCG@10 | Coverage |
| --- | ---: | ---: | ---: | ---: |
| Popularity | 0.1205 | 0.0983 | 0.1561 | 0.0055 |
| Item-item cosine | 0.1640 | 0.1596 | 0.2237 | 0.0475 |
| SVD (64 components) | 0.1133 | 0.1115 | 0.1660 | 0.0650 |

Item-item cosine wins on ranking quality and serves in the API. The
tradeoff is catalog coverage: the winner reaches 4.75 percent of the
catalog in its top 10s, while SVD reaches 6.5 percent because its
scores spread across the catalog instead of concentrating on hits. A
production system would weigh that spread against the NDCG gap; at
this scale the ranking win is worth more. One more honest observation:
the popularity baseline is hard to beat on precision at this scale,
which is exactly why it stays in the table.

## Tech stack

- Python 3.11 or newer
- FastAPI and uvicorn for the serving layer
- pandas and NumPy for data handling
- SciPy sparse matrices for the rating matrix and item similarity
- scikit-learn TruncatedSVD for the factorization model
- pytest and pytest-cov for tests, ruff for lint and formatting
- Docker for packaging; the image trains at build time

## The three models

- **Popularity** (`PopularityRecommender`): one global ranking by rating
  count, ties broken by mean rating. The baseline every other model has
  to justify itself against.
- **Item-item cosine** (`ItemItemRecommender`): cosine similarity
  between movie rating vectors, pruned to the 50 nearest neighbors per
  movie. A user scores a movie by the similarity-weighted sum of their
  ratings for its neighbors.
- **SVD** (`SVDRecommender`): TruncatedSVD over the user-mean-centered
  rating matrix. Predicted scores read as approximate star ratings.

Scores are comparable within one model's output only. Item-item scores
are unnormalized weighted sums and run large; SVD scores sit near the
rating scale.

## Cold start

A new user has no row in the rating matrix, so none of the three models
can score them. `GenreProfileRecommender` takes a handful of rated
movies, turns them into a per-genre affinity (mean rating per genre
minus 3.0, the neutral point of the scale), and ranks unseen movies by
the mean affinity of their genres, tie-broken by popularity. It is
simple on purpose: with three to five ratings there is not enough signal
for anything smarter, and the failure mode is a popularity list with a
genre tilt, which is still a reasonable first screen.

## Architecture

```
ratings.csv, movies.csv
        |
   recs.train  ---- evaluates all three models on the holdout
        |
   artifacts/  ---- models.pkl, movies.json, results.json
        |
   recs.api (FastAPI) ---- /recommend/user, /recommend/similar,
                           /search, /recommend/new-user, demo page
```

Training fits the served models on the full ratings file after
evaluation, so production recommendations use every rating available.
Artifacts are pickles built locally by the training run; the API loads
them from the directory named by `RECS_ARTIFACTS` (default
`artifacts`).

## API

| Route | What it returns |
| --- | --- |
| `GET /healthz` | Status, serving model, dataset counts |
| `GET /` | A small demo page that calls the API |
| `GET /recommend/user/{user_id}?n=10` | Top n for a known user, seen items excluded |
| `GET /recommend/similar/{movie_id}?n=10` | Item-item neighbors for a movie |
| `GET /search?title=toy+story` | Substring title match, fuzzy fallback |
| `POST /recommend/new-user` | Cold start: body is `{"ratings": [{"movieId": 1, "rating": 5.0}], "n": 10}` |

Unknown user and movie ids return 404 with a message that points at the
right alternative. Ratings outside the 0.5 to 5.0 scale fail validation
with 422.

Example, captured from a local run against the trained artifacts
(`GET /recommend/user/1?n=5`):

```json
{
  "userId": 1,
  "model": "item_item",
  "recommendations": [
    {"movieId": 2918, "title": "Ferris Bueller's Day Off (1986)", "genres": ["Comedy"], "score": 163.0021},
    {"movieId": 2762, "title": "Sixth Sense, The (1999)", "genres": ["Drama", "Horror", "Mystery"], "score": 119.7416},
    {"movieId": 1527, "title": "Fifth Element, The (1997)", "genres": ["Action", "Adventure", "Comedy", "Sci-Fi"], "score": 111.2546},
    {"movieId": 1036, "title": "Die Hard (1988)", "genres": ["Action", "Crime", "Thriller"], "score": 99.0327},
    {"movieId": 1200, "title": "Aliens (1986)", "genres": ["Action", "Adventure", "Horror", "Sci-Fi"], "score": 94.1174}
  ]
}
```

## Quickstart

```bash
make setup                                   # venv with dev tools
make data                                    # download MovieLens
make train                                   # evaluate and build artifacts
make serve                                   # uvicorn on port 8000
make test                                    # pytest with coverage
make lint                                    # ruff
```

Then open http://localhost:8000 for the demo page or
http://localhost:8000/docs for the OpenAPI view.

## Deployment

Every target below has a free tier; nothing in this repo assumes a paid
plan.

- **Docker**: `docker build -t recommendation-engine .` then
  `docker run -p 8000:8000 recommendation-engine`. The image downloads
  the data and trains at build time, so the build needs network access
  and takes a few minutes; the running container does not. The server
  listens on the `PORT` environment variable when one is set, which is
  how most container hosts assign ports.
- **Render**: `render.yaml` defines a free web service from the
  Dockerfile with `/healthz` as the health check.
- **GHCR**: the `Publish container image` workflow builds and pushes
  `ghcr.io/sairajiv-13/recommendation-engine` on version tags and on
  manual dispatch.

## Design decisions

- [ADR 0001](docs/adr/0001-svd-over-deep-learning.md): why classical
  factorization instead of a deep model at this scale, and what the
  first evaluation run did to that plan.

## Limitations

- MovieLens latest-small is a research sample: 610 users, skewed toward
  active raters, and not a stand-in for production traffic.
- Ratings are explicit feedback. The system has no notion of views,
  clicks, or dwell time, and no implicit-feedback weighting.
- No online learning. New ratings change recommendations only after a
  retrain and a fresh artifact build.
- Relevance in evaluation is a hard threshold at 4.0 stars. A graded
  relevance protocol would rank the models slightly differently.
- MovieLens data is provided by GroupLens for research and education
  use; check their terms before using the dataset commercially.

## Testing

Tests run fully offline against a hand-built fixture in the MovieLens
schema (5 users, 8 movies, 20 ratings) whose expected outputs are
computable by hand: matrix shape and lookups, metric functions
(including NDCG of a perfect ranking at exactly 1.0 and a hand-computed
imperfect case), each model's recommendation shape and seen-item
exclusion, the cold-start path, and every API endpoint including the
404 and 422 cases.

33 tests, 89 percent coverage of the `recs` package at the time of
writing. The uncovered lines are mostly the `recs.train` command-line
wrapper, which every real training run exercises end to end. Run the
suite with `make test`.

## License

MIT. See [LICENSE](LICENSE).
