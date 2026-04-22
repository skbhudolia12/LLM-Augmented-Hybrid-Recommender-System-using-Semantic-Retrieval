"""
Interaction Text Builder for I2P-BERT.

Converts raw user data (demographics + rating history) into a structured
natural language summary that serves as input to the BERT encoder.

This is the key bridge between tabular interaction data and the language model:
  raw user data → natural language text → BERT → structured profile

The text template captures:
  - Demographics (gender, age bracket, occupation)
  - Rating behavior (count, mean, variance)
  - Genre preferences (top-5 by average rating)
  - Rating style (generous vs. selective)
  - Temporal preference (preferred movie era)
  - Activity level
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.data.download import GENRE_LIST
from src.data.preprocess import (
    compute_genre_affinity,
    compute_era_distribution,
    classify_activity_level,
    ERA_LABELS,
)

logger = logging.getLogger(__name__)


from joblib import Parallel, delayed

def build_interaction_text(
    user_row: dict,
    user_ratings: pd.DataFrame,
    movies_df: pd.DataFrame,
    top_k_genres: int = 5,
) -> str:
    """
    Construct a natural language interaction summary for a single user.

    Args:
        user_row: User attributes dictionary.
        user_ratings: subset of Ratings DataFrame.
        movies_df: Movies DataFrame.
        top_k_genres: Number of top genres to include.

    Returns:
        Interaction text string.
    """
    # --- Demographics ---
    gender = "Male" if user_row["gender"] == "M" else "Female"
    age = user_row["age"]
    occupation = user_row["occupation"]

    # --- Rating behavior ---
    num_ratings = len(user_ratings)

    if num_ratings == 0:
        return (
            f"{gender}, age {age}, occupation: {occupation}. "
            f"No rating history available. New user."
        )

    avg_rating = user_ratings["rating"].mean()
    rating_std = user_ratings["rating"].std() if num_ratings > 1 else 0.0

    # --- Genre preferences ---
    genre_aff = compute_genre_affinity(user_ratings, movies_df)
    # Sort genres by average rating, keep only those with ratings
    rated_genres = {g: r for g, r in genre_aff.items() if r > 0}
    sorted_genres = sorted(rated_genres.items(), key=lambda x: x[1], reverse=True)
    top_genres = sorted_genres[:top_k_genres]

    genre_str = ", ".join(f"{g} ({(r*100):.1f}% interaction ratio)" for g, r in top_genres)

    # --- Rating style ---
    style = "N/A (Implicit Feedback)"

    # --- Era preference ---
    era_dist = compute_era_distribution(user_ratings, movies_df)
    preferred_era = max(era_dist, key=era_dist.get) if era_dist else "mixed"

    # --- Activity level ---
    activity = classify_activity_level(num_ratings)

    # --- Construct text ---
    text = (
        f"{gender}, age {age}, occupation: {occupation}. "
        f"Rated {num_ratings} movies with average {avg_rating:.1f} "
        f"(standard deviation {rating_std:.2f}). "
        f"Top genres: {genre_str}. "
        f"Rating style: {style}. "
        f"Prefers {preferred_era} era films. "
        f"{'Highly active' if activity == 'high' else 'Moderately active' if activity == 'medium' else 'Light'} user."
    )
    return text


def build_all_interaction_texts(
    users_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    user_ids: Optional[list] = None,
) -> dict:
    """
    Build interaction texts for all (or specified) users.

    Args:
        users_df: Users DataFrame.
        ratings_df: Ratings DataFrame (training split).
        movies_df: Movies DataFrame.
        user_ids: Optional list of user IDs. If None, processes all.

    Returns:
        dict mapping user_id → interaction text string.
    """
    if user_ids is None:
        user_ids = users_df["user_id"].unique()

    logger.info("Building interaction texts for %d users...", len(user_ids))
    
    # Pre-group ratings for O(1) lookup
    ratings_grouped = dict(tuple(ratings_df.groupby("user_id")))
    users_dict = users_df.set_index("user_id").to_dict("index")

    def process_user(uid):
        user_row = users_dict[uid]
        user_ratings = ratings_grouped.get(uid, pd.DataFrame(columns=ratings_df.columns))
        return uid, build_interaction_text(user_row, user_ratings, movies_df)

    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(process_user)(uid) for uid in user_ids
    )

    texts = dict(results)
    logger.info("Interaction text construction complete.")
    return texts
