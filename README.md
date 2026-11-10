# Neural Movie Recommender

A movie recommendation system built in PyTorch and trained on the
[MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) dataset
(1,000,209 ratings from 6,040 users on about 3,700 movies).

## Plan

- **Baseline:** matrix factorization (user and movie embeddings with biases).
- **Main model:** Neural Collaborative Filtering (NeuMF) with movie genre features.
- **Evaluation:** RMSE for rating prediction, Hit Rate@10 and NDCG@10 for ranking.
- **App:** a web app where a new user rates a few movies, the app learns an
  embedding for them on the fly, and shows personalized top-10 recommendations.

## Setup

Requires Python 3.12+ and an NVIDIA GPU is recommended (CPU works, just slower).

```powershell
python -m venv C:\Users\<you>\venvs\movierec
C:\Users\<you>\venvs\movierec\Scripts\Activate.ps1
pip install -r requirements.txt
python -m movie_rec.check_env
```

`requirements.txt` pins the CUDA 12.6 build of PyTorch. On a machine without an
NVIDIA GPU, change that line to `torch==2.14.1` to get the CPU build.

The virtual environment is kept outside the repo so it isn't committed or synced
to cloud storage.

## Data

```powershell
python -m movie_rec.data
```

This downloads MovieLens 1M into `data/raw/`, then writes cleaned files to
`data/processed/`:

- User and movie IDs are remapped to contiguous 0-based indices so they can be
  used directly as embedding rows. The 177 listed movies with no ratings are dropped,
  leaving 3,706 movies.
- Each user's ratings are split **by time**: the oldest 80% go to training, the
  next 10% to validation, and the most recent 10% to test. This mirrors real use,
  where the model predicts what a user will rate next, and avoids training on a
  user's future ratings. Every user appears in all three splits.
- The few validation/test ratings for movies that never appear in training
  (18 and 36 ratings) are dropped, because those movies have no trained embedding.
- Genres are kept as a multi-hot feature (18 genres) for the NeuMF model.

| Split      | Ratings |
|------------|--------:|
| Train      | 800,309 |
| Validation |  99,932 |
| Test       |  99,914 |

## Project layout

```
movie_rec/          Python package with all project code
  config.py         paths, random seed, device selection
  check_env.py      verifies library versions and GPU training
  data.py           download, cleaning, time-based split, PyTorch data loaders
data/               downloaded and processed datasets (not committed)
models/             trained model checkpoints (not committed)
```
