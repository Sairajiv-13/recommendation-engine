"""Download and unpack the MovieLens latest-small dataset from GroupLens.

Source: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
The dataset holds about 100,000 ratings from 610 users on 9,000-plus
movies. It is provided by the GroupLens research group at the University
of Minnesota for research and education use.

Usage:
    python data/get_data.py [--out data]
"""

from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data")
    args = parser.parse_args()
    out = Path(args.out)
    target = out / "ml-latest-small"
    if (target / "ratings.csv").exists():
        print(f"already present: {target}")
        return
    print(f"downloading {URL}")
    with urllib.request.urlopen(URL) as response:
        payload = response.read()
    archive = zipfile.ZipFile(io.BytesIO(payload))
    archive.extractall(out)
    ratings = target / "ratings.csv"
    movies = target / "movies.csv"
    n_ratings = sum(1 for _ in open(ratings)) - 1
    n_movies = sum(1 for _ in open(movies)) - 1
    print(f"ratings rows: {n_ratings}")
    print(f"movies rows: {n_movies}")
    print(f"extracted to {target}")


if __name__ == "__main__":
    main()
