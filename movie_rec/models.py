import torch
from torch import nn


class MatrixFactorization(nn.Module):
    """Classic biased matrix factorization.

    predicted rating = global mean + user bias + movie bias + (user vector . movie vector)

    The biases capture "this user rates high" and "this movie is liked by everyone";
    the dot product captures how well a user's tastes match a movie's traits.
    """

    def __init__(self, n_users: int, n_movies: int, dim: int = 32, global_mean: float = 3.5):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.movie_emb = nn.Embedding(n_movies, dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.movie_bias = nn.Embedding(n_movies, 1)
        self.global_bias = nn.Parameter(torch.tensor(global_mean))

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.movie_emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)

    def forward(self, users: torch.Tensor, movies: torch.Tensor) -> torch.Tensor:
        dot = (self.user_emb(users) * self.movie_emb(movies)).sum(dim=-1)
        bias = self.user_bias(users).squeeze(-1) + self.movie_bias(movies).squeeze(-1)
        return self.global_bias + bias + dot
