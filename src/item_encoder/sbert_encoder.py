"""
sBERT Item Encoder.

Encodes movie metadata (title + genres + plot summary) into dense
384-dimensional semantic vectors using Sentence-BERT.

These vectors capture the semantic content of each movie and are
stored in a vector database (ChromaDB) for similarity retrieval.
"""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class ItemEncoder:
    """
    Encode movie metadata into dense semantic vectors using sBERT.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        """
        Args:
            model_name: Sentence-Transformers model identifier.
            device: Device for encoding ('cuda', 'cpu', or None for auto).
        """
        logger.info("Loading sBERT model: %s", model_name)
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info("sBERT loaded. Embedding dim: %d", self.embedding_dim)

    def _build_item_text(
        self,
        title: str,
        genres: list,
        plot_summary: str = "",
    ) -> str:
        """
        Construct the text representation of a movie for sBERT encoding.

        Args:
            title: Movie title (e.g., "Toy Story (1995)").
            genres: List of genre strings.
            plot_summary: TMDB plot summary (may be empty).

        Returns:
            Concatenated text string.
        """
        parts = [title]
        if genres:
            parts.append(f"Genres: {', '.join(genres)}.")
        if plot_summary:
            parts.append(plot_summary)
        return " ".join(parts)

    def encode_single(
        self,
        title: str,
        genres: list,
        plot_summary: str = "",
    ) -> np.ndarray:
        """
        Encode a single movie into a dense vector.

        Returns:
            numpy array of shape (embedding_dim,).
        """
        text = self._build_item_text(title, genres, plot_summary)
        return self.model.encode(text, convert_to_numpy=True)

    def encode_batch(
        self,
        titles: list,
        genres_list: list,
        plot_summaries: list = None,
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode a batch of movies into dense vectors.

        Args:
            titles: List of movie title strings.
            genres_list: List of genre lists.
            plot_summaries: Optional list of plot summary strings.
            batch_size: Encoding batch size.
            show_progress: Show progress bar.

        Returns:
            numpy array of shape (num_movies, embedding_dim).
        """
        if plot_summaries is None:
            plot_summaries = [""] * len(titles)

        texts = [
            self._build_item_text(t, g, p)
            for t, g, p in zip(titles, genres_list, plot_summaries)
        ]

        logger.info("Encoding %d movies with sBERT (batch_size=%d)...", len(texts), batch_size)
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        logger.info("Encoding complete. Shape: %s", embeddings.shape)
        return embeddings

    def encode_movies_df(
        self,
        movies_df,
        plot_summaries: dict = None,
        batch_size: int = 64,
    ) -> dict:
        """
        Encode all movies in a DataFrame.

        Args:
            movies_df: DataFrame with movie_id, title, genres columns.
            plot_summaries: Optional dict mapping movie_id → plot summary.
            batch_size: Encoding batch size.

        Returns:
            dict mapping movie_id → numpy embedding array.
        """
        if plot_summaries is None:
            plot_summaries = {}

        titles = movies_df["title"].tolist()
        genres_list = movies_df["genres"].tolist()
        plots = [plot_summaries.get(mid, "") for mid in movies_df["movie_id"]]
        movie_ids = movies_df["movie_id"].tolist()

        embeddings = self.encode_batch(titles, genres_list, plots, batch_size)

        result = {}
        for mid, emb in zip(movie_ids, embeddings):
            result[mid] = emb

        return result
