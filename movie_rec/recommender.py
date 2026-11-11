"""Recommendation logic behind the web app, built on the trained implicit NeuMF model."""
import json

import pandas as pd
import torch
from torch.nn import functional as F

from movie_rec.config import MODELS_DIR, get_device
from movie_rec.data import GENRES, load_processed
from movie_rec.evaluate import LIKED_THRESHOLD, interaction_matrix, log_popularity
from movie_rec.train import RESULTS_DIR, load_model

DEFAULT_ALPHA = 0.75
FOLD_IN_STEPS = 100
FOLD_IN_LR = 0.1
FOLD_IN_L2 = 1e-3
NEGATIVES_PER_LIKE = 4


class Recommender:
    def __init__(self, checkpoint: str = "neumf_implicit"):
        self.device = get_device()
        self.data = load_processed()
        self.movies = self.data.movies.set_index("movie_idx")
        self.n_movies = self.data.n_movies

        self.model = load_model(MODELS_DIR / f"{checkpoint}.pt", self.data, self.device)
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.popularity = log_popularity(self.data.train, self.n_movies, self.device)
        self.all_movies = torch.arange(self.n_movies, device=self.device)
        self.default_alpha = self._tuned_alpha(checkpoint)

        # The average trained user is the starting point for someone we know nothing about.
        self.mean_gmf = self.model.user_gmf.weight.mean(dim=0)
        self.mean_mlp = self.model.user_mlp.weight.mean(dim=0)
        self.mean_bias = self.model.user_bias.weight.mean()

        movie_vectors = torch.cat([self.model.movie_gmf.weight, self.model.movie_mlp.weight], dim=1)
        self.movie_vectors = F.normalize(movie_vectors, dim=1)

        history = pd.concat([self.data.train, self.data.val])
        self.seen = interaction_matrix(self.data.n_users, self.n_movies, history, self.device)
        future_liked = self.data.test[self.data.test["rating"] >= LIKED_THRESHOLD]
        self.future_liked = interaction_matrix(self.data.n_users, self.n_movies, future_liked, self.device)

    @staticmethod
    def _tuned_alpha(checkpoint: str) -> float:
        path = RESULTS_DIR / "ranking.json"
        if path.exists():
            return float(json.loads(path.read_text()).get("tuned_alpha", {}).get(checkpoint, DEFAULT_ALPHA))
        return DEFAULT_ALPHA

    def title_choices(self) -> list[str]:
        """All movie titles, most-rated first so familiar movies are easy to find."""
        order = torch.argsort(self.popularity, descending=True).tolist()
        return [self.movies.at[i, "title"] for i in order]

    def titles_to_idx(self, titles: list[str]) -> list[int]:
        lookup = pd.Series(self.movies.index, index=self.movies["title"])
        return [int(lookup[t]) for t in titles]

    def fold_in(self, liked: list[int], disliked: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Learn vectors for a brand-new user from a few liked/disliked movies.

        The trained model stays frozen; only the new user's GMF vector, MLP vector, and
        bias are optimized, with the same objective the model was trained on: liked movies
        are positives, disliked movies plus random movies are negatives. An L2 penalty
        keeps the vectors close to the average user when there's little evidence.
        """
        u_gmf = self.mean_gmf.clone().requires_grad_(True)
        u_mlp = self.mean_mlp.clone().requires_grad_(True)
        u_bias = self.mean_bias.clone().requires_grad_(True)
        if not liked:
            return u_gmf.detach(), u_mlp.detach(), u_bias.detach()

        positives = torch.tensor(liked, device=self.device)
        explicit_negatives = torch.tensor(disliked, dtype=torch.long, device=self.device)
        optimizer = torch.optim.Adam([u_gmf, u_mlp, u_bias], lr=FOLD_IN_LR)
        generator = torch.Generator(device=self.device).manual_seed(0)

        # Drawing every step's random negatives at once avoids hundreds of tiny GPU calls.
        n_random = NEGATIVES_PER_LIKE * len(liked)
        random_negatives = torch.randint(
            0, self.n_movies, (FOLD_IN_STEPS, n_random), device=self.device, generator=generator
        )
        not_liked = torch.ones(self.n_movies, dtype=torch.bool, device=self.device)
        not_liked[positives] = False
        n_fixed = len(positives) + len(explicit_negatives)
        labels = torch.zeros(n_fixed + n_random, device=self.device)
        labels[: len(positives)] = 1.0
        # Random draws that hit a liked movie get zero weight instead of a wrong label.
        weights = torch.ones(FOLD_IN_STEPS, n_fixed + n_random, device=self.device)
        weights[:, n_fixed:] = not_liked[random_negatives].float()

        for step in range(FOLD_IN_STEPS):
            movies = torch.cat([positives, explicit_negatives, random_negatives[step]])
            n = len(movies)
            logits = self.model.score(u_gmf.expand(n, -1), u_mlp.expand(n, -1), u_bias.expand(n), movies)
            penalty = ((u_gmf - self.mean_gmf) ** 2).sum() + ((u_mlp - self.mean_mlp) ** 2).sum()
            bce = F.binary_cross_entropy_with_logits(logits, labels, weight=weights[step])
            loss = bce + FOLD_IN_L2 * penalty
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return u_gmf.detach(), u_mlp.detach(), u_bias.detach()

    @torch.no_grad()
    def _top_k(self, scores: torch.Tensor, exclude: torch.Tensor, alpha: float, k: int,
               genres: list[str] | None) -> torch.Tensor:
        scores = scores + alpha * self.popularity
        scores = scores.masked_fill(exclude, float("-inf"))
        if genres:
            genre_cols = torch.tensor([GENRES.index(g) for g in genres], device=self.device)
            has_genre = self.model.genres[:, genre_cols].sum(dim=1) > 0
            scores = scores.masked_fill(~has_genre, float("-inf"))
        return scores.topk(k).indices

    def _describe(self, movie_idx: list[int]) -> pd.DataFrame:
        rows = self.movies.loc[movie_idx]
        return pd.DataFrame({
            "Rank": range(1, len(movie_idx) + 1),
            "Movie": rows["title"].values,
            "Genres": rows["genres"].str.replace("|", ", ", regex=False).values,
        })

    @torch.no_grad()
    def recommend_new_user(self, liked_titles: list[str], disliked_titles: list[str],
                           alpha: float, k: int = 10, genres: list[str] | None = None) -> pd.DataFrame:
        liked = self.titles_to_idx(liked_titles)
        disliked = self.titles_to_idx(disliked_titles)
        with torch.enable_grad():
            u_gmf, u_mlp, u_bias = self.fold_in(liked, disliked)

        n = self.n_movies
        scores = self.model.score(u_gmf.expand(n, -1), u_mlp.expand(n, -1), u_bias.expand(n), self.all_movies)
        exclude = torch.zeros(n, dtype=torch.bool, device=self.device)
        exclude[liked + disliked] = True
        top = self._top_k(scores, exclude, alpha, k, genres).tolist()

        table = self._describe(top)
        if liked:
            similarity = self.movie_vectors[top] @ self.movie_vectors[liked].T
            closest = similarity.argmax(dim=1).tolist()
            table["Because you liked"] = [self.movies.at[liked[i], "title"] for i in closest]
        return table

    @torch.no_grad()
    def recommend_existing_user(self, user_id: int, alpha: float, k: int = 10) -> tuple[pd.DataFrame, pd.DataFrame, str]:
        """Recommend for a real MovieLens user and check against what they rated 4-5 stars
        in the held-out test period (which the model never saw)."""
        user = user_id - 1
        users = torch.full((self.n_movies,), user, device=self.device)
        scores = self.model(users, self.all_movies)
        top = self._top_k(scores, self.seen[user], alpha, k, None).tolist()

        table = self._describe(top)
        hits = self.future_liked[user, top].tolist()
        table["Rated 4-5 stars later?"] = ["yes" if h else "" for h in hits]

        history = self.data.train[self.data.train["user_idx"] == user]
        favorites = history.sort_values(["rating", "timestamp"], ascending=False).head(k)
        favorites_table = pd.DataFrame({
            "Movie": self.movies.loc[favorites["movie_idx"], "title"].values,
            "Stars": favorites["rating"].astype(int).values,
        })

        n_future = int(self.future_liked[user].sum())
        summary = (f"User {user_id} rated {len(history)} movies in training. In the test period they "
                   f"rated **{n_future}** movies 4-5 stars; **{sum(hits)}** of those are in this top {k}.")
        return favorites_table, table, summary
