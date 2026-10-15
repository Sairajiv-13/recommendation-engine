"""A tiny fixture in the MovieLens schema.

Five users, eight movies, twenty ratings. The patterns are deliberate:
user 1 and 2 lean Action, user 3 leans Comedy, user 4 leans Drama, and
movie 101 is the most rated title. Expected outputs in the tests are
computable by hand from these rows.
"""

from __future__ import annotations

import pandas as pd
import pytest

MOVIES = [
    (101, "Alpha Action (2000)", "Action"),
    (102, "Beta Action (2001)", "Action"),
    (103, "Gamma Comedy (2002)", "Comedy"),
    (104, "Delta Comedy (2003)", "Comedy"),
    (105, "Epsilon Drama (2004)", "Drama"),
    (106, "Zeta Action Comedy (2005)", "Action|Comedy"),
    (107, "Eta Drama (2006)", "Drama"),
    (108, "Theta Action (2007)", "Action"),
]

RATINGS = [
    (1, 101, 5.0),
    (1, 102, 4.0),
    (1, 103, 1.0),
    (1, 105, 2.0),
    (2, 101, 5.0),
    (2, 102, 5.0),
    (2, 106, 4.0),
    (2, 108, 4.0),
    (3, 103, 5.0),
    (3, 104, 5.0),
    (3, 106, 4.0),
    (3, 105, 1.0),
    (4, 105, 5.0),
    (4, 107, 5.0),
    (4, 101, 2.0),
    (4, 103, 2.0),
    (5, 101, 4.0),
    (5, 103, 4.0),
    (5, 105, 4.0),
    (5, 107, 3.0),
]


@pytest.fixture()
def movies() -> pd.DataFrame:
    frame = pd.DataFrame(MOVIES, columns=["movieId", "title", "genres"])
    frame["genre_list"] = frame["genres"].str.split("|")
    return frame


@pytest.fixture()
def ratings() -> pd.DataFrame:
    frame = pd.DataFrame(RATINGS, columns=["userId", "movieId", "rating"])
    frame["timestamp"] = 1_700_000_000
    return frame
