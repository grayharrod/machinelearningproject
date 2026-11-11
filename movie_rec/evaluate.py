"""Evaluate trained models as recommenders: how good is each user's top-10 list?

For every user, all movies they haven't already rated are scored and the 10 best are
recommended. A recommendation counts as a hit if the user rated that movie 4 or 5 stars
in the held-out period. Ranking against the full catalog (rather than against ~100
sampled movies) avoids the distortions described by Krichene & Rendle (2020).

Run with:  python -m movie_rec.evaluate
"""
import json
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch

from movie_rec.config import MODELS_DIR, get_device
from movie_rec.data import MovieLensData, load_processed
from movie_rec.train import RESULTS_DIR, load_model, rmse

K = 10
LIKED_THRESHOLD = 4.0
USER_BATCH = 256
ALPHAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0)
# (checkpoint name, whether the model predicts star ratings and so has an RMSE)
MODELS = (("mf", True), ("neumf", True), ("neumf_implicit", False))

Scorer = Callable[[torch.Tensor], torch.Tensor]


def interaction_matrix(n_users: int, n_movies: int, ratings: pd.DataFrame, device: torch.device) -> torch.Tensor:
    matrix = torch.zeros(n_users, n_movies, dtype=torch.bool, device=device)
    users = torch.tensor(ratings["user_idx"].values, device=device)
    movies = torch.tensor(ratings["movie_idx"].values, device=device)
    matrix[users, movies] = True
    return matrix


def model_scorer(model: torch.nn.Module, n_movies: int, device: torch.device) -> Scorer:
    """Scores every movie for a batch of users using the model's predicted rating."""
    all_movies = torch.arange(n_movies, device=device)

    @torch.no_grad()
    def score(users: torch.Tensor) -> torch.Tensor:
        user_col = users.repeat_interleave(n_movies)
        movie_col = all_movies.repeat(len(users))
        return model(user_col, movie_col).view(len(users), n_movies)

    return score


def log_popularity(train: pd.DataFrame, n_movies: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(train["movie_idx"].values, minlength=n_movies)
    return torch.log1p(torch.tensor(counts, dtype=torch.float32, device=device))


def popularity_scorer(train: pd.DataFrame, n_movies: int, device: torch.device) -> Scorer:
    """Non-personalized baseline: every user gets the movies with the most training ratings."""
    popularity = log_popularity(train, n_movies, device)

    def score(users: torch.Tensor) -> torch.Tensor:
        return popularity.expand(len(users), n_movies)

    return score


def blended_scorer(score: Scorer, train: pd.DataFrame, n_movies: int, alpha: float, device: torch.device) -> Scorer:
    """predicted rating + alpha * log(1 + number of ratings).

    A rating model estimates how much a user would like a movie *if* they watched it;
    the popularity term estimates how likely they are to watch it at all.
    """
    popularity = log_popularity(train, n_movies, device)

    def blended(users: torch.Tensor) -> torch.Tensor:
        return score(users) + alpha * popularity

    return blended


@torch.no_grad()
def ranking_metrics(score: Scorer, seen: torch.Tensor, liked: torch.Tensor, k: int = K) -> dict:
    """Hit rate, precision, and NDCG at k, averaged over users with at least one liked movie.

    seen:  (n_users, n_movies) movies each user already rated; never recommended again
    liked: (n_users, n_movies) held-out movies each user rated >= LIKED_THRESHOLD
    """
    device = seen.device
    users = torch.nonzero(liked.any(dim=1)).squeeze(1)
    discounts = 1.0 / torch.log2(torch.arange(2, k + 2, device=device, dtype=torch.float32))
    ideal_dcg = torch.cumsum(discounts, dim=0)

    hit, precision, ndcg = [], [], []
    recommended = torch.zeros(seen.shape[1], dtype=torch.bool, device=device)
    for start in range(0, len(users), USER_BATCH):
        batch = users[start:start + USER_BATCH]
        scores = score(batch).masked_fill(seen[batch], float("-inf"))
        top_k = scores.topk(k, dim=1).indices
        hits = liked[batch].gather(1, top_k).float()
        n_liked = liked[batch].sum(dim=1).clamp(max=k)

        hit.append(hits.any(dim=1).float())
        precision.append(hits.mean(dim=1))
        ndcg.append((hits * discounts).sum(dim=1) / ideal_dcg[n_liked - 1])
        recommended[top_k.flatten()] = True

    return {
        f"hit_rate@{k}": torch.cat(hit).mean().item(),
        f"precision@{k}": torch.cat(precision).mean().item(),
        f"ndcg@{k}": torch.cat(ndcg).mean().item(),
        "coverage": recommended.float().mean().item(),
        "n_users": len(users),
    }


def evaluate_split(score: Scorer, data: MovieLensData, split: str, device: torch.device) -> dict:
    """Validation ranks against train history; test ranks against train + validation history."""
    if split == "val":
        history, held_out = data.train, data.val
    else:
        history, held_out = pd.concat([data.train, data.val]), data.test
    seen = interaction_matrix(data.n_users, data.n_movies, history, device)
    liked = interaction_matrix(
        data.n_users, data.n_movies, held_out[held_out["rating"] >= LIKED_THRESHOLD], device
    )
    return ranking_metrics(score, seen, liked)


def tune_alpha(score: Scorer, data: MovieLensData, device: torch.device, name: str) -> float:
    """Pick the popularity weight with the best validation NDCG@10."""
    best_alpha, best_ndcg = 0.0, -1.0
    for alpha in ALPHAS:
        metrics = evaluate_split(blended_scorer(score, data.train, data.n_movies, alpha, device), data, "val", device)
        print(f"  {name} alpha {alpha:<4}  val HR@10 {metrics[f'hit_rate@{K}']:.4f}  "
              f"NDCG@10 {metrics[f'ndcg@{K}']:.4f}  coverage {metrics['coverage']:.1%}")
        if metrics[f"ndcg@{K}"] > best_ndcg:
            best_alpha, best_ndcg = alpha, metrics[f"ndcg@{K}"]
    print(f"  -> best alpha for {name}: {best_alpha}")
    return best_alpha


def print_table(rows: list[dict], split: str) -> None:
    print(f"\n{split} set (top-{K} recommendations from the full catalog)")
    print(f"{'model':<40}{'RMSE':>8}{'HR@10':>9}{'Prec@10':>9}{'NDCG@10':>9}{'Coverage':>10}")
    for r in rows:
        rmse_text = f"{r['rmse']:.4f}" if r["rmse"] is not None else "-"
        print(f"{r['name']:<40}{rmse_text:>8}{r[f'hit_rate@{K}']:>9.4f}{r[f'precision@{K}']:>9.4f}"
              f"{r[f'ndcg@{K}']:>9.4f}{r['coverage']:>10.1%}")


def main() -> None:
    device = get_device()
    data = load_processed()

    scorers: dict[str, tuple[Scorer, torch.nn.Module | None]] = {
        "popularity": (popularity_scorer(data.train, data.n_movies, device), None),
    }
    tuned = {}
    for name, predicts_ratings in MODELS:
        path = MODELS_DIR / f"{name}.pt"
        if not path.exists():
            print(f"Skipping {name}: {path} not found")
            continue
        model = load_model(path, data, device)
        score = model_scorer(model, data.n_movies, device)
        rating_model = model if predicts_ratings else None
        scorers[name] = (score, rating_model)
        print(f"Tuning popularity weight for {name} on the validation set:")
        alpha = tune_alpha(score, data, device, name)
        tuned[name] = alpha
        if alpha > 0:
            scorers[f"{name} + popularity (a={alpha})"] = (
                blended_scorer(score, data.train, data.n_movies, alpha, device), rating_model
            )

    results = {}
    for split in ("val", "test"):
        ratings = data.val if split == "val" else data.test
        rows = []
        for name, (score, model) in scorers.items():
            metrics = evaluate_split(score, data, split, device)
            metrics["rmse"] = rmse(model, ratings, device) if model is not None else None
            rows.append({"name": name, **metrics})
        print_table(rows, split)
        results[split] = rows

    results["tuned_alpha"] = tuned
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "ranking.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {RESULTS_DIR / 'ranking.json'}")


if __name__ == "__main__":
    main()
