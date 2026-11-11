"""Web app: personalized movie recommendations from the trained NeuMF model.

Run with:  python -m movie_rec.app   (then open http://127.0.0.1:7860)
"""
import gradio as gr

from movie_rec.data import GENRES
from movie_rec.recommender import Recommender

INTRO = """
# Neural Movie Recommender
NeuMF trained on 1 million MovieLens ratings. Pick a few movies you liked (and optionally
some you disliked); the app learns a taste profile for you in under a second and
recommends 10 movies you haven't picked.
"""

EXISTING_INTRO = """
Pick a real MovieLens user. The model only saw their older ratings; their most recent
ratings were held out as the test set. **"Rated 4-5 stars later?"** shows which
recommendations that user actually went on to rate highly.
"""

ALPHA_INFO = "Higher = more mainstream picks, lower = more niche. Default was tuned on the validation set."

TASTE_EXAMPLES = [
    [["Star Wars: Episode IV - A New Hope (1977)", "Alien (1979)", "Terminator, The (1984)", "Matrix, The (1999)"], [], []],
    [["Fargo (1996)", "Pulp Fiction (1994)", "Reservoir Dogs (1992)"], ["Titanic (1997)"], []],
    [["Toy Story (1995)", "Lion King, The (1994)", "Aladdin (1992)"], ["Alien (1979)"], ["Comedy"]],
    [["When Harry Met Sally... (1989)", "Sleepless in Seattle (1993)", "Pretty Woman (1990)"], [], []],
]
# Users whose held-out favorites the model ranks well, found by scanning every user's top 10.
USER_EXAMPLES = [[2873], [2247], [1820], [3412]]


def build_app(rec: Recommender) -> gr.Blocks:
    titles = rec.title_choices()

    def new_user(liked, disliked, genres, alpha):
        if not liked:
            raise gr.Error("Pick at least one movie you liked.")
        overlap = set(liked) & set(disliked or [])
        if overlap:
            raise gr.Error(f"A movie can't be both liked and disliked: {', '.join(overlap)}")
        return rec.recommend_new_user(liked, disliked or [], alpha, genres=genres or None)

    def existing_user(user_id, alpha):
        if user_id is None or not 1 <= int(user_id) <= rec.data.n_users:
            raise gr.Error(f"User ID must be between 1 and {rec.data.n_users}.")
        return rec.recommend_existing_user(int(user_id), alpha)

    with gr.Blocks(title="Neural Movie Recommender") as app:
        gr.Markdown(INTRO)

        with gr.Tab("Recommend for me"):
            with gr.Row():
                with gr.Column():
                    liked = gr.Dropdown(titles, multiselect=True, label="Movies you liked")
                    disliked = gr.Dropdown(titles, multiselect=True, label="Movies you disliked (optional)")
                    genres = gr.Dropdown(GENRES, multiselect=True, label="Only these genres (optional)")
                    alpha = gr.Slider(0, 2, value=rec.default_alpha, step=0.05,
                                      label="Popularity weight", info=ALPHA_INFO)
                    button = gr.Button("Recommend", variant="primary")
                with gr.Column(scale=2):
                    results = gr.Dataframe(
                        headers=["Rank", "Movie", "Genres", "Because you liked"],
                        column_widths=["8%", "36%", "26%", "30%"],
                        label="Your top 10", interactive=False, wrap=True,
                    )
            gr.Examples(TASTE_EXAMPLES, [liked, disliked, genres], label="Example tastes (click to fill in)")
            button.click(new_user, [liked, disliked, genres, alpha], results)

        with gr.Tab("Check against a real user"):
            gr.Markdown(EXISTING_INTRO)
            with gr.Row():
                user_id = gr.Number(value=1, precision=0, label=f"User ID (1-{rec.data.n_users})")
                user_alpha = gr.Slider(0, 2, value=rec.default_alpha, step=0.05,
                                       label="Popularity weight", info=ALPHA_INFO)
                user_button = gr.Button("Recommend", variant="primary")
            gr.Examples(USER_EXAMPLES, [user_id], label="Users to try")
            summary = gr.Markdown()
            with gr.Row():
                favorites = gr.Dataframe(
                    headers=["Movie", "Stars"], column_widths=["80%", "20%"],
                    label="Their favorite movies (training data)", interactive=False, wrap=True, scale=2,
                )
                user_results = gr.Dataframe(
                    headers=["Rank", "Movie", "Genres", "Rated 4-5 stars later?"],
                    column_widths=["12%", "38%", "30%", "20%"],
                    label="Model's top 10 for them", interactive=False, wrap=True, scale=3,
                )
            user_button.click(existing_user, [user_id, user_alpha], [favorites, user_results, summary])

    return app


if __name__ == "__main__":
    print("Loading model and data...")
    build_app(Recommender()).launch()
