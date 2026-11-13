"""Train a recommender on the MovieLens 1M training split and evaluate it.

Run with:  python -m movie_rec.train --model mf
"""
import argparse
import copy
import json
import math
import time

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from movie_rec.config import MODELS_DIR, PROJECT_ROOT, get_device, set_seed
from movie_rec.data import MovieLensData, load_processed, make_loader
from movie_rec.models import MatrixFactorization, NeuMF

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"
MIN_RATING, MAX_RATING = 1.0, 5.0


def build_model(args: argparse.Namespace, data: MovieLensData) -> nn.Module:
    global_mean = float(data.train["rating"].mean())
    if args.model == "mf":
        return MatrixFactorization(data.n_users, data.n_movies, dim=args.dim, global_mean=global_mean)
    if args.model == "neumf":
        return NeuMF(
            data.n_users,
            data.n_movies,
            genres=data.genre_matrix(),
            dim=args.dim,
            mlp_layers=tuple(int(n) for n in args.mlp_layers.split(",")),
            genre_dim=args.genre_dim,
            dropout=args.dropout,
            global_mean=global_mean,
        )
    raise ValueError(f"Unknown model: {args.model}")


def load_model(path, data: MovieLensData, device: torch.device) -> nn.Module:
    """Rebuild a model from a checkpoint saved by this script and load its weights."""
    checkpoint = torch.load(path, map_location=device)
    defaults = vars(build_parser().parse_args([]))
    args = argparse.Namespace(**{**defaults, **checkpoint["hparams"]})
    model = build_model(args, data).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict(model: nn.Module, ratings: pd.DataFrame, device: torch.device, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    preds = []
    for users, movies, _ in make_loader(ratings, batch_size=batch_size, shuffle=False):
        preds.append(model(users.to(device), movies.to(device)).clamp(MIN_RATING, MAX_RATING).cpu())
    return torch.cat(preds).numpy()


def rmse(model: nn.Module, ratings: pd.DataFrame, device: torch.device) -> float:
    errors = predict(model, ratings, device) - ratings["rating"].values
    return float(np.sqrt(np.mean(errors**2)))


def fit(
    model: nn.Module,
    data: MovieLensData,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    patience: int,
) -> list[dict]:
    """Train with Adam on MSE loss, early-stopping on validation RMSE.

    Returns per-epoch history; the model is left holding the best-validation weights.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)
    history = []
    best_val, best_state, bad_epochs = math.inf, None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        start = time.perf_counter()
        total_sq_error = torch.zeros((), device=device)
        for users, movies, ratings in make_loader(data.train, batch_size=batch_size):
            users, movies, ratings = users.to(device), movies.to(device), ratings.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model(users, movies), ratings)
            loss.backward()
            optimizer.step()
            total_sq_error += loss.detach() * len(ratings)

        train_rmse = math.sqrt(total_sq_error.item() / len(data.train))
        val_rmse = rmse(model, data.val, device)
        history.append({"epoch": epoch, "train_rmse": train_rmse, "val_rmse": val_rmse})
        print(f"epoch {epoch:2d}  train RMSE {train_rmse:.4f}  val RMSE {val_rmse:.4f}  "
              f"({time.perf_counter() - start:.1f}s)")

        if val_rmse < best_val:
            best_val, best_state, bad_epochs = val_rmse, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Validation RMSE hasn't improved for {patience} epochs, stopping early.")
                break

    model.load_state_dict(best_state)
    return history


def save_curve(history: list[dict], title: str, path) -> None:
    epochs = [h["epoch"] for h in history]
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, [h["train_rmse"] for h in history], marker="o", label="train")
    plt.plot(epochs, [h["val_rmse"] for h in history], marker="o", label="validation")
    plt.xlabel("epoch")
    plt.ylabel("RMSE")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["mf", "neumf"], default="mf")
    parser.add_argument("--name", help="run name for saved files (defaults to the model name)")
    parser.add_argument("--dim", type=int, default=32, help="embedding size")
    parser.add_argument("--mlp-layers", default="64,32,16", help="NeuMF hidden layer sizes")
    parser.add_argument("--genre-dim", type=int, default=8, help="NeuMF genre embedding size (0 = no genres)")
    parser.add_argument("--dropout", type=float, default=0.5, help="NeuMF dropout between hidden layers")
    parser.add_argument("--init-from", help="NeuMF only: path to a trained mf checkpoint to start from")
    parser.add_argument("--freeze-pretrained", action="store_true",
                        help="with --init-from: keep the copied matrix factorization weights fixed")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    name = args.name or args.model

    set_seed()
    device = get_device()
    data = load_processed()
    print(f"Training '{name}' on {device} with {len(data.train):,} ratings")

    global_mean = data.train["rating"].mean()
    mean_rmse = float(np.sqrt(np.mean((data.test["rating"] - global_mean) ** 2)))
    print(f"Reference: always predicting the training mean ({global_mean:.3f}) gives test RMSE {mean_rmse:.4f}\n")

    model = build_model(args, data).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params:,} parameters")
    if args.init_from:
        model.init_from_mf(torch.load(args.init_from, map_location=device)["state_dict"])
        print(f"Initialized from {args.init_from}: val RMSE before fine-tuning {rmse(model, data.val, device):.4f}")
        if args.freeze_pretrained:
            model.freeze_mf_parts()
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Froze matrix factorization weights; {n_trainable:,} parameters left to train")

    start = time.perf_counter()
    history = fit(model, data, device, args.epochs, args.lr, args.weight_decay, args.batch_size, args.patience)
    train_seconds = time.perf_counter() - start

    best = min(history, key=lambda h: h["val_rmse"])
    test_rmse = rmse(model, data.test, device)
    print(f"\nBest epoch {best['epoch']}: val RMSE {best['val_rmse']:.4f}")
    print(f"Test RMSE {test_rmse:.4f}  (training mean baseline: {mean_rmse:.4f})")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": args.model, "hparams": vars(args), "state_dict": model.state_dict()},
        MODELS_DIR / f"{name}.pt",
    )
    results = {
        "name": name,
        "hparams": vars(args),
        "n_params": n_params,
        "train_seconds": round(train_seconds, 1),
        "best_epoch": best["epoch"],
        "best_val_rmse": best["val_rmse"],
        "test_rmse": test_rmse,
        "mean_baseline_test_rmse": mean_rmse,
        "history": history,
    }
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(results, indent=2))
    save_curve(history, f"{name}: RMSE per epoch", RESULTS_DIR / f"{name}_curve.png")
    print(f"Saved model to {MODELS_DIR / f'{name}.pt'} and results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
