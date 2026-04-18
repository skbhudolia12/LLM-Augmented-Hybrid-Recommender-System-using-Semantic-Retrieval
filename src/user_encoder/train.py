"""
Training loop for I2P-BERT.

Handles:
  - PyTorch Dataset/DataLoader for user interaction texts + labels
  - Training loop with multi-task loss
  - Validation and early stopping
  - Model checkpointing
  - Profile generation and storage in SQLite
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
import pandas as pd
from sqlalchemy import create_engine, text

from src.user_encoder.i2p_bert import I2PBERT, I2PBERTLoss
from src.data.download import GENRE_LIST
from src.data.preprocess import ERA_LABELS

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Dataset                                                            #
# ------------------------------------------------------------------ #

class UserProfileDataset(Dataset):
    """
    PyTorch Dataset for I2P-BERT training.

    Each sample is a (interaction_text, label_dict) pair.
    """

    def __init__(
        self,
        interaction_texts: dict,
        labels_df: pd.DataFrame,
        tokenizer: BertTokenizer,
        max_length: int = 256,
    ):
        """
        Args:
            interaction_texts: dict mapping user_id → text string.
            labels_df: DataFrame with columns: user_id, genre_affinities,
                       activity_label, era_label, tendency_score.
            tokenizer: BERT tokenizer.
            max_length: Max token sequence length.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Align texts and labels by user_id
        self.user_ids = []
        self.texts = []
        self.genre_targets = []
        self.activity_targets = []
        self.era_targets = []
        self.tendency_targets = []

        for _, row in labels_df.iterrows():
            uid = row["user_id"]
            if uid not in interaction_texts:
                continue
            self.user_ids.append(uid)
            self.texts.append(interaction_texts[uid])
            self.genre_targets.append(row["genre_affinities"])
            self.activity_targets.append(row["activity_label"])
            self.era_targets.append(row["era_label"])
            self.tendency_targets.append(row["tendency_score"])

        logger.info("UserProfileDataset: %d samples", len(self.user_ids))

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "user_id": self.user_ids[idx],
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "genre_affinities": torch.tensor(self.genre_targets[idx], dtype=torch.float32),
            "activity_label": torch.tensor(self.activity_targets[idx], dtype=torch.long),
            "era_label": torch.tensor(self.era_targets[idx], dtype=torch.long),
            "tendency_score": torch.tensor([self.tendency_targets[idx]], dtype=torch.float32),
        }


# ------------------------------------------------------------------ #
#  Training                                                           #
# ------------------------------------------------------------------ #

def train_i2p_bert(
    interaction_texts: dict,
    labels_df: pd.DataFrame,
    config: dict,
    save_dir: str = "checkpoints/i2p_bert",
) -> I2PBERT:
    """
    Train the I2P-BERT model.

    Args:
        interaction_texts: dict mapping user_id → text.
        labels_df: DataFrame with ground-truth labels.
        config: User encoder config dict from YAML.
        save_dir: Directory to save checkpoints.

    Returns:
        Trained I2PBERT model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training I2P-BERT on %s", device)

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(config["bert_model"])

    # Split labels into train/val (90/10 of training users)
    labels_shuffled = labels_df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(labels_shuffled) * 0.9)
    train_labels = labels_shuffled.iloc[:split_idx]
    val_labels = labels_shuffled.iloc[split_idx:]

    # Datasets
    train_dataset = UserProfileDataset(interaction_texts, train_labels, tokenizer)
    val_dataset = UserProfileDataset(interaction_texts, val_labels, tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0
    )

    # Model
    model = I2PBERT(
        bert_model_name=config["bert_model"],
        num_genres=config["num_genres"],
        num_eras=config["num_eras"],
        num_activity_levels=config["num_activity_levels"],
        dropout=config["dropout"],
    ).to(device)

    # Loss
    loss_fn = I2PBERTLoss(weights=config["loss_weights"]).to(device)

    # Optimizer + scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    total_steps = len(train_loader) * config["epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Training loop
    best_val_loss = float("inf")
    patience = 3
    patience_counter = 0
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(config["epochs"]):
        # --- Train ---
        model.train()
        train_losses = []

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = {
                "genre_affinities": batch["genre_affinities"].to(device),
                "activity_label": batch["activity_label"].to(device),
                "era_label": batch["era_label"].to(device),
                "tendency_score": batch["tendency_score"].to(device),
            }

            optimizer.zero_grad()
            predictions = model(input_ids, attention_mask)
            losses = loss_fn(predictions, targets)
            losses["total_loss"].backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_losses.append(losses["total_loss"].item())

        avg_train_loss = np.mean(train_losses)

        # --- Validate ---
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = {
                    "genre_affinities": batch["genre_affinities"].to(device),
                    "activity_label": batch["activity_label"].to(device),
                    "era_label": batch["era_label"].to(device),
                    "tendency_score": batch["tendency_score"].to(device),
                }
                predictions = model(input_ids, attention_mask)
                losses = loss_fn(predictions, targets)
                val_losses.append(losses["total_loss"].item())

        avg_val_loss = np.mean(val_losses)

        logger.info(
            "Epoch %d/%d — train_loss: %.4f, val_loss: %.4f",
            epoch + 1, config["epochs"], avg_train_loss, avg_val_loss,
        )

        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path / "best_model.pt")
            logger.info("  ✓ Saved best model (val_loss=%.4f)", best_val_loss)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("  ✗ Early stopping at epoch %d", epoch + 1)
                break

    # Load best model
    model.load_state_dict(torch.load(save_path / "best_model.pt", weights_only=True))
    logger.info("Training complete. Best val_loss: %.4f", best_val_loss)
    return model


# ------------------------------------------------------------------ #
#  Profile Generation                                                 #
# ------------------------------------------------------------------ #

def generate_profiles(
    model: I2PBERT,
    interaction_texts: dict,
    tokenizer: BertTokenizer,
    db_path: str,
    batch_size: int = 64,
    device: str = "cuda",
):
    """
    Generate structured JSON profiles for all users and store in SQLite.

    Args:
        model: Trained I2PBERT model.
        interaction_texts: dict mapping user_id → text.
        tokenizer: BERT tokenizer.
        db_path: Path to SQLite database file.
        batch_size: Batch size for inference.
        device: Device string.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Set up SQLite
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id INTEGER PRIMARY KEY,
                cls_embedding TEXT,
                genre_affinity TEXT,
                activity_level TEXT,
                preference_era TEXT,
                rating_tendency REAL,
                interaction_text TEXT
            )
        """))
        conn.commit()

    activity_labels = ["low", "medium", "high"]
    era_labels = ERA_LABELS
    user_ids = list(interaction_texts.keys())
    total = len(user_ids)

    logger.info("Generating profiles for %d users...", total)

    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_uids = user_ids[start:end]
            batch_texts = [interaction_texts[uid] for uid in batch_uids]

            # Tokenize
            encoding = tokenizer(
                batch_texts,
                max_length=256,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask)

            # Process each user in batch
            records = []
            for i, uid in enumerate(batch_uids):
                cls_emb = outputs["cls_embedding"][i].cpu().numpy().tolist()
                genre_pred = outputs["genre_affinity"][i].cpu().numpy().tolist()
                activity_idx = outputs["activity_level"][i].argmax().item()
                era_idx = outputs["preference_era"][i].argmax().item()
                tendency = outputs["rating_tendency"][i].item()

                # Build genre affinity dict
                genre_dict = {g: round(float(v), 3) for g, v in zip(GENRE_LIST, genre_pred)}

                records.append({
                    "user_id": int(uid),
                    "cls_embedding": json.dumps(cls_emb),
                    "genre_affinity": json.dumps(genre_dict),
                    "activity_level": activity_labels[activity_idx],
                    "preference_era": era_labels[era_idx],
                    "rating_tendency": round(tendency, 4),
                    "interaction_text": interaction_texts[uid],
                })

            # Insert into SQLite
            with engine.connect() as conn:
                for rec in records:
                    conn.execute(
                        text("""
                            INSERT OR REPLACE INTO user_profiles
                            (user_id, cls_embedding, genre_affinity, activity_level,
                             preference_era, rating_tendency, interaction_text)
                            VALUES (:user_id, :cls_embedding, :genre_affinity,
                                    :activity_level, :preference_era, :rating_tendency,
                                    :interaction_text)
                        """),
                        rec,
                    )
                conn.commit()

            if (end) % 500 == 0 or end == total:
                logger.info("Generated profiles: %d/%d", end, total)

    logger.info("All profiles stored in %s", db_path)
