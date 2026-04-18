"""
Baseline Models for comparison.

Implements standard recommendation baselines using the Surprise library
and custom implementations:
  - Global Mean
  - User/Item Mean
  - SVD
  - SVD++
  - NCF (Neural Collaborative Filtering) via PyTorch
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)


def evaluate_predictions(y_true, y_pred, name: str = "Model") -> dict:
    """Compute RMSE and MAE for a set of predictions."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    logger.info("%s — RMSE: %.4f, MAE: %.4f", name, rmse, mae)
    return {"name": name, "rmse": rmse, "mae": mae}


def global_mean_baseline(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Predict the global mean rating for all test pairs."""
    global_mean = train["rating"].mean()
    predictions = np.full(len(test), global_mean)
    return evaluate_predictions(test["rating"].values, predictions, "Global Mean")


def user_item_mean_baseline(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Predict using user mean + item mean - global mean."""
    global_mean = train["rating"].mean()
    user_means = train.groupby("user_id")["rating"].mean()
    item_means = train.groupby("movie_id")["rating"].mean()

    predictions = []
    for _, row in test.iterrows():
        u_mean = user_means.get(row["user_id"], global_mean)
        i_mean = item_means.get(row["movie_id"], global_mean)
        pred = u_mean + i_mean - global_mean
        predictions.append(np.clip(pred, 1, 5))

    return evaluate_predictions(test["rating"].values, np.array(predictions), "User/Item Mean")


def svd_baseline(train: pd.DataFrame, test: pd.DataFrame, n_factors: int = 100) -> dict:
    """SVD baseline using Surprise library."""
    try:
        from surprise import SVD, Dataset, Reader
        from surprise.model_selection import cross_validate
    except ImportError:
        logger.warning("Surprise library not installed. Skipping SVD baseline.")
        logger.warning("Install with: pip install scikit-surprise")
        return {"name": "SVD", "rmse": None, "mae": None}

    reader = Reader(rating_scale=(1, 5))
    train_data = Dataset.load_from_df(
        train[["user_id", "movie_id", "rating"]], reader
    )

    trainset = train_data.build_full_trainset()
    model = SVD(n_factors=n_factors, random_state=42)
    model.fit(trainset)

    predictions = []
    for _, row in test.iterrows():
        pred = model.predict(row["user_id"], row["movie_id"])
        predictions.append(pred.est)

    return evaluate_predictions(test["rating"].values, np.array(predictions), "SVD")


def svdpp_baseline(train: pd.DataFrame, test: pd.DataFrame, n_factors: int = 20) -> dict:
    """SVD++ baseline using Surprise library."""
    try:
        from surprise import SVDpp, Dataset, Reader
    except ImportError:
        logger.warning("Surprise library not installed. Skipping SVD++ baseline.")
        return {"name": "SVD++", "rmse": None, "mae": None}

    reader = Reader(rating_scale=(1, 5))
    train_data = Dataset.load_from_df(
        train[["user_id", "movie_id", "rating"]], reader
    )

    trainset = train_data.build_full_trainset()
    model = SVDpp(n_factors=n_factors, random_state=42)
    model.fit(trainset)

    predictions = []
    for _, row in test.iterrows():
        pred = model.predict(row["user_id"], row["movie_id"])
        predictions.append(pred.est)

    return evaluate_predictions(test["rating"].values, np.array(predictions), "SVD++")


def run_all_baselines(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """
    Run all baseline models and return a comparison DataFrame.

    Returns:
        DataFrame with columns: name, rmse, mae
    """
    logger.info("=" * 60)
    logger.info("Running Baselines")
    logger.info("=" * 60)

    results = []
    results.append(global_mean_baseline(train, test))
    results.append(user_item_mean_baseline(train, test))
    results.append(svd_baseline(train, test))
    results.append(svdpp_baseline(train, test))

    results_df = pd.DataFrame(results)
    logger.info("\n%s", results_df.to_string(index=False))
    return results_df
