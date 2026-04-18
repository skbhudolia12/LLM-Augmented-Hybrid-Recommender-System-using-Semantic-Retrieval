"""
Main Pipeline Orchestrator.

Runs the full I2P-BERT hybrid recommendation pipeline end-to-end:
  Stage 1: Data download, preprocessing, and augmentation
  Stage 2: Item encoding (sBERT → ChromaDB)
  Stage 3: User encoding (I2P-BERT → SQLite)
  Stage 4: Feature fusion + XGBoost training and evaluation

Usage:
    python -m scripts.run_pipeline                     # Full pipeline
    python -m scripts.run_pipeline --stage 2           # Start from Stage 2
    python -m scripts.run_pipeline --skip-tmdb         # Skip TMDB augmentation
"""

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.download import download_ml1m, load_all
from src.data.preprocess import timestamp_split, compute_all_user_labels
from src.data.augment import TMDBClient
from src.item_encoder.sbert_encoder import ItemEncoder
from src.item_encoder.vector_store import VectorStore
from src.user_encoder.interaction_text import build_all_interaction_texts
from src.user_encoder.i2p_bert import I2PBERT
from src.user_encoder.train import train_i2p_bert, generate_profiles
from src.feature_fusion.fusion import FeatureFusion
from src.predictor.xgboost_model import train_xgboost, evaluate, analyze_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def stage1_data(config: dict, skip_tmdb: bool = False):
    """
    Stage 1: Download data, preprocess, compute labels.

    Returns:
        dict with all data artifacts needed by later stages.
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: Data Preparation")
    logger.info("=" * 60)

    # 1a. Download & parse ML-1M
    data_dir = download_ml1m(config["data"]["raw_dir"])
    data = load_all(data_dir)

    ratings = data["ratings"]
    users = data["users"]
    movies = data["movies"]

    logger.info("Dataset: %d ratings, %d users, %d movies",
                len(ratings), len(users), len(movies))

    # 1b. Timestamp-based split
    train, val, test = timestamp_split(
        ratings,
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        test_ratio=config["data"]["test_ratio"],
    )

    # Save splits
    processed_dir = Path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(processed_dir / "train.parquet", index=False)
    val.to_parquet(processed_dir / "val.parquet", index=False)
    test.to_parquet(processed_dir / "test.parquet", index=False)
    logger.info("Saved splits to %s", processed_dir)

    # 1c. Compute ground-truth labels for I2P-BERT
    labels = compute_all_user_labels(train, movies)
    labels.to_parquet(processed_dir / "user_labels.parquet", index=False)
    logger.info("Computed labels for %d users", len(labels))

    # 1d. TMDB augmentation (optional)
    plot_summaries = {}
    if not skip_tmdb and config["data"]["tmdb_api_key"]:
        tmdb = TMDBClient(
            api_key=config["data"]["tmdb_api_key"],
            cache_path=config["data"]["tmdb_cache_path"],
        )
        plot_summaries = tmdb.augment_movies(movies)
    else:
        logger.info("Skipping TMDB augmentation (no API key or --skip-tmdb)")

    return {
        "train": train,
        "val": val,
        "test": test,
        "users": users,
        "movies": movies,
        "labels": labels,
        "plot_summaries": plot_summaries,
    }


def stage2_items(config: dict, movies: pd.DataFrame, plot_summaries: dict):
    """
    Stage 2: Encode items with sBERT → store in ChromaDB.

    Returns:
        (item_embeddings dict, VectorStore instance)
    """
    logger.info("=" * 60)
    logger.info("STAGE 2: Item Encoding (sBERT)")
    logger.info("=" * 60)

    encoder = ItemEncoder(
        model_name=config["item_encoder"]["model_name"],
        device=config["project"]["device"],
    )

    # Encode all movies
    item_embeddings = encoder.encode_movies_df(
        movies,
        plot_summaries=plot_summaries,
        batch_size=config["item_encoder"]["batch_size"],
    )

    # Store in ChromaDB
    vector_store = VectorStore(
        persist_dir=config["item_encoder"]["vector_db_path"],
        collection_name="items",
    )

    # Build metadata for each movie
    metadata = {}
    for _, row in movies.iterrows():
        mid = row["movie_id"]
        metadata[mid] = {
            "title": str(row["title"]),
            "year": int(row["year"]) if pd.notna(row["year"]) else 0,
            "genres": "|".join(row["genres"]),
        }

    vector_store.add_items(
        movie_ids=list(item_embeddings.keys()),
        embeddings=item_embeddings,
        metadata=metadata,
    )

    logger.info("Stored %d item embeddings in ChromaDB", vector_store.count)
    return item_embeddings, vector_store


def stage3_users(
    config: dict,
    users: pd.DataFrame,
    train: pd.DataFrame,
    movies: pd.DataFrame,
    labels: pd.DataFrame,
):
    """
    Stage 3: Build interaction texts → train I2P-BERT → generate profiles.

    Returns:
        Trained I2PBERT model.
    """
    logger.info("=" * 60)
    logger.info("STAGE 3: User Encoding (I2P-BERT)")
    logger.info("=" * 60)

    # 3a. Build interaction texts
    interaction_texts = build_all_interaction_texts(users, train, movies)
    logger.info("Built %d interaction texts", len(interaction_texts))

    # 3b. Train I2P-BERT
    model = train_i2p_bert(
        interaction_texts=interaction_texts,
        labels_df=labels,
        config=config["user_encoder"],
    )

    # 3c. Generate profiles for all users → SQLite
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(config["user_encoder"]["bert_model"])

    generate_profiles(
        model=model,
        interaction_texts=interaction_texts,
        tokenizer=tokenizer,
        db_path=config["user_encoder"]["db_path"],
        batch_size=config["user_encoder"]["batch_size"],
        device=config["project"]["device"],
    )

    return model


def stage4_predict(
    config: dict,
    item_embeddings: dict,
    movies: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
):
    """
    Stage 4: Feature fusion + XGBoost training + evaluation.

    Returns:
        (trained model, test results dict)
    """
    logger.info("=" * 60)
    logger.info("STAGE 4: Feature Fusion + XGBoost")
    logger.info("=" * 60)

    # 4a. Initialize feature fusion
    fusion = FeatureFusion(
        user_db_path=config["user_encoder"]["db_path"],
        item_embeddings=item_embeddings,
        movies_df=movies,
        ratings_df=train,
        pca_dim=config["feature_fusion"]["user_cls_pca_dim"],
        knn_k=config["feature_fusion"]["knn_k"],
    )

    # 4b. Build feature matrices
    logger.info("Building training feature matrix...")
    X_train, y_train = fusion.build_feature_matrix(train)

    logger.info("Building validation feature matrix...")
    X_val, y_val = fusion.build_feature_matrix(val)

    logger.info("Building test feature matrix...")
    X_test, y_test = fusion.build_feature_matrix(test)

    # Save feature matrices for reproducibility
    processed_dir = Path(config["data"]["processed_dir"])
    np.savez_compressed(
        processed_dir / "features_train.npz", X=X_train, y=y_train,
    )
    np.savez_compressed(
        processed_dir / "features_val.npz", X=X_val, y=y_val,
    )
    np.savez_compressed(
        processed_dir / "features_test.npz", X=X_test, y=y_test,
    )
    logger.info("Feature matrices saved to %s", processed_dir)

    # 4c. Train XGBoost
    model = train_xgboost(X_train, y_train, X_val, y_val, config["predictor"])

    # 4d. Evaluate on test set
    results = evaluate(model, X_test, y_test)

    # 4e. SHAP analysis
    analyze_features(model, X_test, save_dir="results")

    return model, results


def main():
    parser = argparse.ArgumentParser(description="I2P-BERT Pipeline")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    parser.add_argument("--stage", type=int, default=1, help="Start from stage (1-4)")
    parser.add_argument("--skip-tmdb", action="store_true", help="Skip TMDB augmentation")
    args = parser.parse_args()

    config = load_config(args.config)

    # Set random seeds
    seed = config["project"]["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info("=" * 60)
    logger.info("I2P-BERT Hybrid Recommendation Pipeline")
    logger.info("=" * 60)
    logger.info("Config: %s", args.config)
    logger.info("Starting from stage: %d", args.stage)
    logger.info("Device: %s", config["project"]["device"])

    # --- Stage 1 ---
    processed_dir = Path(config["data"]["processed_dir"])
    if args.stage <= 1:
        artifacts = stage1_data(config, skip_tmdb=args.skip_tmdb)
    else:
        # Load pre-computed artifacts
        logger.info("Loading pre-computed data artifacts...")
        data_dir = config["data"]["raw_dir"]
        data = load_all(data_dir)
        artifacts = {
            "train": pd.read_parquet(processed_dir / "train.parquet"),
            "val": pd.read_parquet(processed_dir / "val.parquet"),
            "test": pd.read_parquet(processed_dir / "test.parquet"),
            "users": data["users"],
            "movies": data["movies"],
            "labels": pd.read_parquet(processed_dir / "user_labels.parquet"),
            "plot_summaries": {},
        }

    # --- Stage 2 ---
    if args.stage <= 2:
        item_embeddings, vector_store = stage2_items(
            config, artifacts["movies"], artifacts["plot_summaries"],
        )
    else:
        logger.info("Loading pre-computed item embeddings...")
        vector_store = VectorStore(persist_dir=config["item_encoder"]["vector_db_path"])
        item_embeddings = vector_store.get_all_embeddings()

    # --- Stage 3 ---
    if args.stage <= 3:
        stage3_users(
            config,
            artifacts["users"],
            artifacts["train"],
            artifacts["movies"],
            artifacts["labels"],
        )

    # --- Stage 4 ---
    if args.stage <= 4:
        model, results = stage4_predict(
            config,
            item_embeddings,
            artifacts["movies"],
            artifacts["train"],
            artifacts["val"],
            artifacts["test"],
        )

        logger.info("=" * 60)
        logger.info("FINAL RESULTS")
        logger.info("=" * 60)
        logger.info("RMSE: %.4f", results["rmse"])
        logger.info("MAE:  %.4f", results["mae"])
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
