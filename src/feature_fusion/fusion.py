"""
Feature Fusion Module.

Constructs the final feature vector for each (user, item) pair by combining:
  1. User profile features from I2P-BERT (CLS embedding + structured fields)
  2. Item features from sBERT (semantic embedding + raw metadata)
  3. Cross-features (user-item similarity signals)
  4. Collaborative signals (k-NN based)

The output feature matrix is fed into XGBoost for rating prediction.
"""

import json
import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
from sqlalchemy import create_engine, text

from src.data.download import GENRE_LIST
from src.data.preprocess import ERA_LABELS

logger = logging.getLogger(__name__)

# One-hot encodings
ACTIVITY_LEVELS = ["low", "medium", "high"]


class FeatureFusion:
    """
    Constructs unified feature vectors for (user, item) pairs.
    """

    def __init__(
        self,
        user_db_path: str,
        item_embeddings: dict,
        movies_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
        pca_dim: int = 128,
        knn_k: int = 10,
    ):
        """
        Args:
            user_db_path: Path to SQLite database with user profiles.
            item_embeddings: dict mapping movie_id → numpy embedding.
            movies_df: Movies DataFrame.
            ratings_df: Training ratings DataFrame.
            pca_dim: Dimension to compress user CLS embeddings via PCA.
            knn_k: Number of neighbors for collaborative signals.
        """
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        self.item_embeddings = item_embeddings
        self.knn_k = knn_k
        self.pca_dim = pca_dim

        # Load user profiles from SQLite
        logger.info("Loading user profiles from %s...", user_db_path)
        engine = create_engine(f"sqlite:///{user_db_path}")
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT * FROM user_profiles")).fetchall()
            columns = conn.execute(text("SELECT * FROM user_profiles LIMIT 1")).keys()

        self.user_profiles = {}
        cls_embeddings = []
        user_id_order = []

        for row in rows:
            row_dict = dict(zip(columns, row))
            uid = row_dict["user_id"]
            self.user_profiles[uid] = {
                "cls_embedding": np.array(json.loads(row_dict["cls_embedding"])),
                "genre_affinity": json.loads(row_dict["genre_affinity"]),
                "activity_level": row_dict["activity_level"],
                "preference_era": row_dict["preference_era"],
                "rating_tendency": row_dict["rating_tendency"],
            }
            cls_embeddings.append(self.user_profiles[uid]["cls_embedding"])
            user_id_order.append(uid)

        logger.info("Loaded %d user profiles", len(self.user_profiles))

        # Fit PCA on all user CLS embeddings
        cls_matrix = np.stack(cls_embeddings)
        actual_pca_dim = min(pca_dim, cls_matrix.shape[0], cls_matrix.shape[1])
        self.pca = PCA(n_components=actual_pca_dim, random_state=42)
        self.pca.fit(cls_matrix)
        logger.info(
            "PCA fitted: %d → %d dims (explained variance: %.2f%%)",
            cls_matrix.shape[1], actual_pca_dim,
            self.pca.explained_variance_ratio_.sum() * 100,
        )

        # Precompute PCA-reduced CLS embeddings
        self.user_cls_pca = {}
        pca_embeddings = self.pca.transform(cls_matrix)
        for uid, emb in zip(user_id_order, pca_embeddings):
            self.user_cls_pca[uid] = emb

        # Build user-item rating lookup for collaborative signals
        self._build_rating_lookup()

    def _build_rating_lookup(self):
        """Build efficient rating lookup structures."""
        # user → {item: rating}
        self.user_ratings = {}
        for _, row in self.ratings_df.iterrows():
            uid, mid, rating = row["user_id"], row["movie_id"], row["rating"]
            if uid not in self.user_ratings:
                self.user_ratings[uid] = {}
            self.user_ratings[uid][mid] = rating

        # item → {user: rating}
        self.item_ratings = {}
        for _, row in self.ratings_df.iterrows():
            uid, mid, rating = row["user_id"], row["movie_id"], row["rating"]
            if mid not in self.item_ratings:
                self.item_ratings[mid] = {}
            self.item_ratings[mid][uid] = rating

    def _get_user_features(self, user_id: int) -> np.ndarray:
        """
        Extract user feature vector from stored profile.

        Returns concatenation of:
          - PCA-reduced CLS embedding (pca_dim)
          - Genre affinity values (18)
          - Activity level one-hot (3)
          - Preference era one-hot (4)
          - Rating tendency (1)

        Total: pca_dim + 26
        """
        profile = self.user_profiles.get(user_id)
        if profile is None:
            # Unknown user — return zeros
            return np.zeros(self.pca_dim + 26)

        # PCA CLS
        cls_pca = self.user_cls_pca[user_id]

        # Genre affinity
        genre_vals = np.array([profile["genre_affinity"].get(g, 0.0) for g in GENRE_LIST])

        # Activity one-hot
        activity_oh = np.zeros(len(ACTIVITY_LEVELS))
        if profile["activity_level"] in ACTIVITY_LEVELS:
            activity_oh[ACTIVITY_LEVELS.index(profile["activity_level"])] = 1.0

        # Era one-hot
        era_oh = np.zeros(len(ERA_LABELS))
        if profile["preference_era"] in ERA_LABELS:
            era_oh[ERA_LABELS.index(profile["preference_era"])] = 1.0

        # Tendency
        tendency = np.array([profile["rating_tendency"]])

        return np.concatenate([cls_pca, genre_vals, activity_oh, era_oh, tendency])

    def _get_item_features(self, movie_id: int) -> np.ndarray:
        """
        Extract item feature vector.

        Returns concatenation of:
          - sBERT embedding (384)
          - Genre binary flags (18)
          - Normalized year (1)

        Total: 403
        """
        # sBERT embedding
        emb = self.item_embeddings.get(movie_id)
        if emb is None:
            emb = np.zeros(384)

        # Genre flags
        movie_row = self.movies_df[self.movies_df["movie_id"] == movie_id]
        if len(movie_row) > 0:
            genre_flags = np.array([movie_row.iloc[0].get(f"genre_{g}", 0) for g in GENRE_LIST],
                                   dtype=np.float32)
            year = movie_row.iloc[0].get("year", 1995)
            year_normalized = (float(year) - 1920) / 80.0 if year else 0.5
        else:
            genre_flags = np.zeros(len(GENRE_LIST))
            year_normalized = 0.5

        return np.concatenate([emb, genre_flags, [year_normalized]])

    def _get_cross_features(self, user_id: int, movie_id: int) -> np.ndarray:
        """
        Compute cross-features between user and item.

        Returns:
          - Cosine similarity between user CLS and item embedding (1)
          - Genre overlap score (1)

        Total: 2
        """
        # User-item cosine similarity
        user_cls = self.user_cls_pca.get(user_id, np.zeros(self.pca_dim))
        item_emb = self.item_embeddings.get(movie_id, np.zeros(384))

        # Use first min(pca_dim, 384) dims for cross-similarity
        dim = min(len(user_cls), len(item_emb))
        if dim > 0 and np.linalg.norm(user_cls[:dim]) > 0 and np.linalg.norm(item_emb[:dim]) > 0:
            cos_sim = float(
                sk_cosine(user_cls[:dim].reshape(1, -1), item_emb[:dim].reshape(1, -1))[0, 0]
            )
        else:
            cos_sim = 0.0

        # Genre overlap
        profile = self.user_profiles.get(user_id)
        genre_affinity = np.array([profile["genre_affinity"].get(g, 0.0)
                                   for g in GENRE_LIST]) if profile else np.zeros(len(GENRE_LIST))
        movie_row = self.movies_df[self.movies_df["movie_id"] == movie_id]
        if len(movie_row) > 0:
            genre_flags = np.array([movie_row.iloc[0].get(f"genre_{g}", 0) for g in GENRE_LIST],
                                   dtype=np.float32)
        else:
            genre_flags = np.zeros(len(GENRE_LIST))

        genre_overlap = float(np.dot(genre_affinity, genre_flags))

        return np.array([cos_sim, genre_overlap])

    def _get_collaborative_features(self, user_id: int, movie_id: int) -> np.ndarray:
        """
        Compute k-NN collaborative filtering features.

        Returns:
          - Mean rating of k-NN users on this item (1)
          - Std of k-NN user ratings on this item (1)
          - Num k-NN users who rated this item (1)
          - Mean of user's ratings on k similar items (1)
          - Std of user's ratings on k similar items (1)
          - Num similar items rated by user (1)

        Total: 6
        """
        # --- k-NN user ratings on this item ---
        item_raters = self.item_ratings.get(movie_id, {})
        if len(item_raters) > 0 and user_id in self.user_cls_pca:
            # Find similar users who rated this item
            user_emb = self.user_cls_pca[user_id]
            rater_ids = [uid for uid in item_raters if uid in self.user_cls_pca and uid != user_id]

            if rater_ids:
                rater_embs = np.stack([self.user_cls_pca[uid] for uid in rater_ids])
                sims = sk_cosine(user_emb.reshape(1, -1), rater_embs)[0]
                top_k_idx = np.argsort(sims)[-self.knn_k:]
                knn_ratings = [item_raters[rater_ids[i]] for i in top_k_idx]
                knn_user_mean = np.mean(knn_ratings)
                knn_user_std = np.std(knn_ratings) if len(knn_ratings) > 1 else 0.0
                knn_user_count = len(knn_ratings)
            else:
                knn_user_mean, knn_user_std, knn_user_count = 0.0, 0.0, 0
        else:
            knn_user_mean, knn_user_std, knn_user_count = 0.0, 0.0, 0

        # --- User's ratings on k similar items ---
        item_emb = self.item_embeddings.get(movie_id)
        user_rated = self.user_ratings.get(user_id, {})

        if item_emb is not None and len(user_rated) > 0:
            rated_with_emb = [(mid, r) for mid, r in user_rated.items()
                              if mid in self.item_embeddings and mid != movie_id]
            if rated_with_emb:
                rated_ids, rated_vals = zip(*rated_with_emb)
                rated_embs = np.stack([self.item_embeddings[mid] for mid in rated_ids])
                sims = sk_cosine(item_emb.reshape(1, -1), rated_embs)[0]
                top_k_idx = np.argsort(sims)[-self.knn_k:]
                knn_item_ratings = [rated_vals[i] for i in top_k_idx]
                knn_item_mean = np.mean(knn_item_ratings)
                knn_item_std = np.std(knn_item_ratings) if len(knn_item_ratings) > 1 else 0.0
                knn_item_count = len(knn_item_ratings)
            else:
                knn_item_mean, knn_item_std, knn_item_count = 0.0, 0.0, 0
        else:
            knn_item_mean, knn_item_std, knn_item_count = 0.0, 0.0, 0

        return np.array([
            knn_user_mean, knn_user_std, float(knn_user_count) / self.knn_k,
            knn_item_mean, knn_item_std, float(knn_item_count) / self.knn_k,
        ])

    def build_feature_vector(self, user_id: int, movie_id: int) -> np.ndarray:
        """
        Build the complete feature vector for a single (user, item) pair.

        This is one row in the XGBoost training/test matrix.

        Returns:
            numpy array of shape (feature_dim,).
        """
        user_feats = self._get_user_features(user_id)       # pca_dim + 26
        item_feats = self._get_item_features(movie_id)      # 403
        cross_feats = self._get_cross_features(user_id, movie_id)  # 2
        collab_feats = self._get_collaborative_features(user_id, movie_id)  # 6

        return np.concatenate([user_feats, item_feats, cross_feats, collab_feats])

    def build_feature_matrix(
        self,
        pairs_df: pd.DataFrame,
        show_progress: bool = True,
    ) -> tuple:
        """
        Build feature matrix for all (user, item) pairs.

        Args:
            pairs_df: DataFrame with user_id, movie_id, rating columns.
            show_progress: Log progress every N rows.

        Returns:
            (X, y) where X is the feature matrix and y is the rating array.
        """
        n = len(pairs_df)
        logger.info("Building feature matrix for %d pairs...", n)

        # Get feature dim from a sample
        sample_vec = self.build_feature_vector(
            pairs_df.iloc[0]["user_id"],
            pairs_df.iloc[0]["movie_id"],
        )
        feature_dim = len(sample_vec)
        logger.info("Feature dimension: %d", feature_dim)

        X = np.zeros((n, feature_dim), dtype=np.float32)
        y = np.zeros(n, dtype=np.float32)

        for i, (_, row) in enumerate(pairs_df.iterrows()):
            X[i] = self.build_feature_vector(int(row["user_id"]), int(row["movie_id"]))
            y[i] = float(row["rating"])

            if show_progress and (i + 1) % 10000 == 0:
                logger.info("Feature construction: %d/%d (%.1f%%)", i + 1, n, (i + 1) / n * 100)

        logger.info("Feature matrix built: X=%s, y=%s", X.shape, y.shape)
        return X, y
