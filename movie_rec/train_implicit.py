"""Train NeuMF the way it was originally designed: to rank movies, not predict stars.

Each training example asks "did this user interact with this movie?" Positives are the
user's training ratings (all of them, or only 4-5 star ones with --positives liked); for
every positive, --negatives random movies are drawn as presumed non-interactions. The
model is trained with binary cross-entropy and early-stopped on validation NDCG@10.

Run with:  python -m movie_rec.train_implicit
"""
import argparse
import copy
import json
import time

import matplotlib
import torch
from torch.nn import functional as F

from movie_rec.config import MODELS_DIR, get_device, set_seed
from movie_rec.data import load_processed
from movie_rec.evaluate import K, LIKED_THRESHOLD, evaluate_split, model_scorer
from movie_rec.models import NeuMF
from movie_rec.train import RESULTS_DIR

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def save_curve(history: list[dict], name: str, path) -> None:
    fig, (ax_loss, ax_ndcg) = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [h["epoch"] for h in history]
    ax_loss.plot(epochs, [h["train_loss"] for h in history], marker="o")
    ax_loss.set(xlabel="epoch", ylabel="training loss (BCE)", title="Training loss")
    ax_ndcg.plot(epochs, [h[f"ndcg@{K}"] for h in history], marker="o", color="tab:orange")
    ax_ndcg.set(xlabel="epoch", ylabel="NDCG@10", title="Validation NDCG@10")
    for ax in (ax_loss, ax_ndcg):
        ax.grid(alpha=0.3)
    fig.suptitle(f"{name}: implicit-feedback NeuMF")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="neumf_implicit")
    parser.add_argument("--positives", choices=["liked", "all"], default="liked")
    parser.add_argument("--negatives", type=int, default=4, help="random negatives per positive")
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--mlp-layers", default="64,32,16")
    parser.add_argument("--genre-dim", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    set_seed()
    device = get_device()
    data = load_processed()

    positives = data.train if args.positives == "all" else data.train[data.train["rating"] >= LIKED_THRESHOLD]
    pos_users = torch.tensor(positives["user_idx"].values, device=device)
    pos_movies = torch.tensor(positives["movie_idx"].values, device=device)
    n_pos = len(positives)
    print(f"Training '{args.name}' on {device}: {n_pos:,} positives ({args.positives}), "
          f"{args.negatives} negatives each")

    model = NeuMF(
        data.n_users, data.n_movies, genres=data.genre_matrix(), dim=args.dim,
        mlp_layers=tuple(int(n) for n in args.mlp_layers.split(",")),
        genre_dim=args.genre_dim, dropout=args.dropout, global_mean=0.0,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    score = model_scorer(model, data.n_movies, device)

    users = pos_users.repeat(args.negatives + 1)
    labels = torch.cat([torch.ones(n_pos, device=device), torch.zeros(n_pos * args.negatives, device=device)])
    history = []
    best_ndcg, best_state, bad_epochs = -1.0, None, 0
    start_all = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.perf_counter()
        negatives = torch.randint(0, data.n_movies, (n_pos * args.negatives,), device=device)
        movies = torch.cat([pos_movies, negatives])
        order = torch.randperm(len(labels), device=device)
        total_loss = torch.zeros((), device=device)
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size]
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(users[idx], movies[idx]), labels[idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.detach() * len(idx)

        model.eval()
        val = evaluate_split(score, data, "val", device)
        train_loss = total_loss.item() / len(order)
        history.append({"epoch": epoch, "train_loss": train_loss, **{k: val[k] for k in (f"hit_rate@{K}", f"ndcg@{K}")}})
        print(f"epoch {epoch:2d}  loss {train_loss:.4f}  val HR@10 {val[f'hit_rate@{K}']:.4f}  "
              f"NDCG@10 {val[f'ndcg@{K}']:.4f}  ({time.perf_counter() - start:.1f}s)")

        if val[f"ndcg@{K}"] > best_ndcg:
            best_ndcg, best_state, bad_epochs = val[f"ndcg@{K}"], copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Validation NDCG@10 hasn't improved for {args.patience} epochs, stopping early.")
                break

    model.load_state_dict(best_state)
    model.eval()
    best = max(history, key=lambda h: h[f"ndcg@{K}"])
    test = evaluate_split(score, data, "test", device)
    print(f"\nBest epoch {best['epoch']}: val NDCG@10 {best[f'ndcg@{K}']:.4f}")
    print(f"Test HR@10 {test[f'hit_rate@{K}']:.4f}  NDCG@10 {test[f'ndcg@{K}']:.4f}  coverage {test['coverage']:.1%}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    hparams = {**vars(args), "model": "neumf", "objective": "implicit"}
    torch.save({"model": "neumf", "hparams": hparams, "state_dict": model.state_dict()},
               MODELS_DIR / f"{args.name}.pt")
    results = {
        "name": args.name,
        "hparams": hparams,
        "train_seconds": round(time.perf_counter() - start_all, 1),
        "best_epoch": best["epoch"],
        "test": test,
        "history": history,
    }
    (RESULTS_DIR / f"{args.name}.json").write_text(json.dumps(results, indent=2))
    save_curve(history, args.name, RESULTS_DIR / f"{args.name}_curve.png")
    print(f"Saved model to {MODELS_DIR / f'{args.name}.pt'} and results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
