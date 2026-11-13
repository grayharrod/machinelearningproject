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


class NeuMF(nn.Module):
    """Neural Collaborative Filtering (He et al., 2017) adapted to predict star ratings,
    with movie genres as an extra input.

    Two branches look at the same user-movie pair, each with its own embeddings:
      - GMF: element-wise product of the user and movie vectors. Matrix factorization sums
        this product with equal weights; here a learned layer weights each dimension.
      - MLP: the user vector, movie vector, and an embedding of the movie's genres are
        concatenated and passed through hidden layers, which can learn non-linear
        interactions that a dot product can't express.
    A final linear layer combines both branches. The global/user/movie biases from matrix
    factorization are kept because they explain much of the variance in explicit ratings.
    """

    def __init__(
        self,
        n_users: int,
        n_movies: int,
        genres: torch.Tensor,
        dim: int = 32,
        mlp_layers: tuple[int, ...] = (64, 32, 16),
        genre_dim: int = 8,
        dropout: float = 0.5,
        global_mean: float = 3.5,
    ):
        super().__init__()
        self.user_gmf = nn.Embedding(n_users, dim)
        self.movie_gmf = nn.Embedding(n_movies, dim)
        self.user_mlp = nn.Embedding(n_users, dim)
        self.movie_mlp = nn.Embedding(n_movies, dim)

        self.use_genres = genre_dim > 0
        if self.use_genres:
            self.register_buffer("genres", genres)
            self.genre_proj = nn.Linear(genres.shape[1], genre_dim)

        layers = []
        in_dim = 2 * dim + genre_dim
        for out_dim in mlp_layers:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(dim + in_dim, 1)

        self.user_bias = nn.Embedding(n_users, 1)
        self.movie_bias = nn.Embedding(n_movies, 1)
        self.global_bias = nn.Parameter(torch.tensor(global_mean))

        for emb in (self.user_gmf, self.movie_gmf, self.user_mlp, self.movie_mlp):
            nn.init.normal_(emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)

    @torch.no_grad()
    def init_from_mf(self, mf_state: dict) -> None:
        """Start from a trained MatrixFactorization so this model initially predicts exactly
        what it does: copy its vectors into the GMF branch and its biases, weight every GMF
        dimension by 1 (a plain dot product), and give the MLP branch zero weight to start."""
        dim = self.user_gmf.embedding_dim
        if mf_state["user_emb.weight"].shape[1] != dim:
            raise ValueError("Matrix factorization embedding size must match NeuMF --dim")
        self.user_gmf.weight.copy_(mf_state["user_emb.weight"])
        self.movie_gmf.weight.copy_(mf_state["movie_emb.weight"])
        self.user_bias.weight.copy_(mf_state["user_bias.weight"])
        self.movie_bias.weight.copy_(mf_state["movie_bias.weight"])
        self.global_bias.copy_(mf_state["global_bias"])
        self.output.weight.zero_()
        self.output.weight[:, :dim] = 1.0
        self.output.bias.zero_()

    def freeze_mf_parts(self) -> None:
        """Stop training the parts copied from matrix factorization, so the MLP branch and
        genres learn only to correct what matrix factorization gets wrong."""
        for param in (self.user_gmf.weight, self.movie_gmf.weight, self.user_bias.weight,
                      self.movie_bias.weight, self.global_bias):
            param.requires_grad_(False)

    def forward(self, users: torch.Tensor, movies: torch.Tensor) -> torch.Tensor:
        return self.score(
            self.user_gmf(users), self.user_mlp(users), self.user_bias(users).squeeze(-1), movies
        )

    def score(
        self,
        user_gmf: torch.Tensor,
        user_mlp: torch.Tensor,
        user_bias: torch.Tensor,
        movies: torch.Tensor,
    ) -> torch.Tensor:
        """Score movies from user vectors directly, so a new user who isn't in the
        embedding tables can be scored with vectors learned on the fly."""
        gmf = user_gmf * self.movie_gmf(movies)

        mlp_inputs = [user_mlp, self.movie_mlp(movies)]
        if self.use_genres:
            mlp_inputs.append(self.genre_proj(self.genres[movies]))
        mlp = self.mlp(torch.cat(mlp_inputs, dim=-1))

        interaction = self.output(torch.cat([gmf, mlp], dim=-1)).squeeze(-1)
        return self.global_bias + user_bias + self.movie_bias(movies).squeeze(-1) + interaction
