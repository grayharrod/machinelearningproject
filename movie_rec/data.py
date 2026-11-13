"""Download MovieLens 1M, clean it, and split it into train/validation/test sets.

Run with:  python -m movie_rec.data
"""
import json
import shutil
import ssl
import urllib.request
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import truststore
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

from movie_rec.config import PROCESSED_DIR, RAW_DIR, set_seed

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ML_DIR = RAW_DIR / "ml-1m"

GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

VAL_FRACTION = 0.1
TEST_FRACTION = 0.1


def download() -> None:
    if (ML_DIR / "ratings.dat").exists():
        print(f"MovieLens 1M already downloaded at {ML_DIR}")
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "ml-1m.zip"
    print(f"Downloading {URL} ...")
    # Verify certificates against the Windows/macOS trust store instead of Python's bundled one.
    context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    with urllib.request.urlopen(URL, context=context) as response, open(zip_path, "wb") as f:
        shutil.copyfileobj(response, f)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR)
    zip_path.unlink()
    print(f"Extracted to {ML_DIR}")


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(
        ML_DIR / "ratings.dat", sep="::", engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )
    movies = pd.read_csv(
        ML_DIR / "movies.dat", sep="::", engine="python",
        names=["movie_id", "title", "genres"], encoding="latin-1",
    )
    return ratings, movies


def build_movies(movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Keep only movies that have ratings and give them contiguous 0-based indices."""
    movies = movies[movies["movie_id"].isin(ratings["movie_id"].unique())]
    movies = movies.sort_values("movie_id").reset_index(drop=True)
    movies["movie_idx"] = np.arange(len(movies))
    year = movies["title"].str.extract(r"\((\d{4})\)\s*$")[0]
    movies["year"] = pd.to_numeric(year, errors="coerce").astype("Int64")
    return movies[["movie_idx", "movie_id", "title", "year", "genres"]]


def index_ratings(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """Replace raw user/movie IDs with contiguous 0-based indices for embedding lookups."""
    user_ids = np.sort(ratings["user_id"].unique())
    user_map = pd.Series(np.arange(len(user_ids)), index=user_ids)
    movie_map = pd.Series(movies["movie_idx"].values, index=movies["movie_id"].values)
    return pd.DataFrame({
        "user_idx": ratings["user_id"].map(user_map).values,
        "movie_idx": ratings["movie_id"].map(movie_map).values,
        "rating": ratings["rating"].astype(np.float32).values,
        "timestamp": ratings["timestamp"].values,
    })


def split_by_time(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per user, the oldest 80% of ratings go to train, the next 10% to validation,
    and the most recent 10% to test, so the model is always predicting the future."""
    ratings = ratings.sort_values(["user_idx", "timestamp", "movie_idx"], kind="stable")
    position = ratings.groupby("user_idx").cumcount().values
    count = ratings.groupby("user_idx")["movie_idx"].transform("size").values
    n_test = np.maximum(1, np.round(count * TEST_FRACTION)).astype(int)
    n_val = np.maximum(1, np.round(count * VAL_FRACTION)).astype(int)

    is_test = position >= count - n_test
    is_val = (position >= count - n_test - n_val) & ~is_test
    is_train = ~(is_test | is_val)
    return (
        ratings[is_train].reset_index(drop=True),
        ratings[is_val].reset_index(drop=True),
        ratings[is_test].reset_index(drop=True),
    )


def drop_unseen_movies(train: pd.DataFrame, other: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Movies that never appear in train have untrained embeddings, so they can't be evaluated."""
    seen = other["movie_idx"].isin(train["movie_idx"].unique())
    return other[seen].reset_index(drop=True), int((~seen).sum())


def prepare() -> None:
    download()
    raw_ratings, raw_movies = load_raw()
    movies = build_movies(raw_movies, raw_ratings)
    ratings = index_ratings(raw_ratings, movies)
    train, val, test = split_by_time(ratings)
    val, val_dropped = drop_unseen_movies(train, val)
    test, test_dropped = drop_unseen_movies(train, test)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    movies.to_csv(PROCESSED_DIR / "movies.csv", index=False)
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)

    n_users = int(ratings["user_idx"].nunique())
    n_movies = len(movies)
    meta = {
        "n_users": n_users,
        "n_movies": n_movies,
        "genres": GENRES,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "val_dropped_unseen": val_dropped,
        "test_dropped_unseen": test_dropped,
    }
    (PROCESSED_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    per_user = ratings.groupby("user_idx").size()
    print(f"\nUsers:    {n_users:,}")
    print(f"Movies:   {n_movies:,} (of {len(raw_movies):,} listed; the rest have no ratings)")
    print(f"Ratings:  {len(ratings):,}")
    print(f"Density:  {len(ratings) / (n_users * n_movies):.2%} of the user-movie matrix is filled")
    print(f"Ratings per user: min {per_user.min()}, median {int(per_user.median())}, max {per_user.max()}")
    print(f"Mean rating: {ratings['rating'].mean():.3f}")
    print("Rating distribution:")
    for stars, share in ratings["rating"].value_counts(normalize=True).sort_index().items():
        print(f"  {int(stars)} stars  {share:6.1%}  {'#' * int(share * 100)}")
    print(f"\nSplit:  train {len(train):,}  |  val {len(val):,}  |  test {len(test):,}")
    print(f"Dropped for movies unseen in train:  val {val_dropped}  |  test {test_dropped}")
    print(f"Saved processed data to {PROCESSED_DIR}")


@dataclass
class MovieLensData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    movies: pd.DataFrame
    n_users: int
    n_movies: int

    def genre_matrix(self) -> torch.Tensor:
        """Multi-hot (n_movies, n_genres) tensor; row i marks the genres of movie_idx i."""
        matrix = torch.zeros(self.n_movies, len(GENRES))
        for row, genres in zip(self.movies["movie_idx"], self.movies["genres"]):
            for genre in genres.split("|"):
                matrix[row, GENRES.index(genre)] = 1.0
        return matrix


def load_processed() -> MovieLensData:
    if not (PROCESSED_DIR / "meta.json").exists():
        raise FileNotFoundError("Processed data not found. Run: python -m movie_rec.data")
    meta = json.loads((PROCESSED_DIR / "meta.json").read_text())
    return MovieLensData(
        train=pd.read_csv(PROCESSED_DIR / "train.csv"),
        val=pd.read_csv(PROCESSED_DIR / "val.csv"),
        test=pd.read_csv(PROCESSED_DIR / "test.csv"),
        movies=pd.read_csv(PROCESSED_DIR / "movies.csv"),
        n_users=meta["n_users"],
        n_movies=meta["n_movies"],
    )


class RatingsDataset(Dataset):
    def __init__(self, ratings: pd.DataFrame):
        self.users = torch.tensor(ratings["user_idx"].values, dtype=torch.long)
        self.movies = torch.tensor(ratings["movie_idx"].values, dtype=torch.long)
        self.ratings = torch.tensor(ratings["rating"].values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]


def make_loader(ratings: pd.DataFrame, batch_size: int = 1024, shuffle: bool = True) -> DataLoader:
    """Yields (users, movies, ratings) batches.

    The sampler hands the dataset a whole list of indices at once, so each batch is one
    tensor slice instead of `batch_size` separate __getitem__ calls plus collation.
    """
    dataset = RatingsDataset(ratings)
    order = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    sampler = BatchSampler(order, batch_size=batch_size, drop_last=False)
    return DataLoader(dataset, sampler=sampler, batch_size=None)


if __name__ == "__main__":
    set_seed()
    prepare()

    data = load_processed()
    users, movies, ratings = next(iter(make_loader(data.train, batch_size=4)))
    print(f"\nSample training batch:\n  users   {users.tolist()}\n  movies  {movies.tolist()}\n  ratings {ratings.tolist()}")
    print(f"Genre matrix shape: {tuple(data.genre_matrix().shape)}")
