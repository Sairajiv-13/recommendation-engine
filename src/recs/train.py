"""Train and evaluate all models, then write serving artifacts.

Usage:
    python -m recs.train --data-dir data/ml-latest-small --out artifacts
"""

from __future__ import annotations

import argparse
from pathlib import Path

from recs.artifacts import build_artifacts
from recs.data import load_movies, load_ratings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/ml-latest-small")
    parser.add_argument("--out", default="artifacts")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    ratings = load_ratings(str(data_dir / "ratings.csv"))
    movies = load_movies(str(data_dir / "movies.csv"))
    print(
        f"ratings: {len(ratings)}  users: {ratings['userId'].nunique()}  "
        f"movies rated: {ratings['movieId'].nunique()}"
    )

    summary = build_artifacts(ratings, movies, args.out)
    header = f"{'model':<12} {'P@10':>8} {'R@10':>8} {'NDCG@10':>8} {'coverage':>9} {'users':>6}"
    print(header)
    for name in ("popularity", "item_item", "svd"):
        r = summary["models"][name]
        print(
            f"{name:<12} {r['precision_at_10']:>8.4f} {r['recall_at_10']:>8.4f} "
            f"{r['ndcg_at_10']:>8.4f} {r['coverage']:>9.4f} {r['users_evaluated']:>6}"
        )
    print(f"winner by NDCG@10: {summary['winner']}")
    print(f"artifacts written to {args.out}")


if __name__ == "__main__":
    main()
