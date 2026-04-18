"""
I2P-BERT: Interaction-to-Profile BERT Model.

Core novelty of the pipeline. Takes natural language interaction summaries
as input and predicts structured user profile fields via multi-task heads:

    Input:  "Male, age 25-34, occupation: engineer. Rated 127 movies..."
            ↓
    BERT Encoder (fine-tuned bert-base-uncased)
            ↓
    [CLS] token embedding (768-dim)
            ↓
    Multi-Task Heads:
      ├── Genre Affinity Regression  (18 floats: avg rating per genre)
      ├── Activity Level Classifier  (3 classes: low/medium/high)
      ├── Preference Era Classifier  (4 classes: classic/80s/90s/2000s)
      └── Rating Tendency Regression (1 float: generous ↔ harsh z-score)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer


class I2PBERT(nn.Module):
    """
    Interaction-to-Profile BERT.

    Encodes user interaction text via BERT and produces structured
    profile predictions through multiple task-specific heads.
    """

    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        num_genres: int = 18,
        num_eras: int = 4,
        num_activity_levels: int = 3,
        dropout: float = 0.1,
        freeze_bert_layers: int = 0,
    ):
        """
        Args:
            bert_model_name: HuggingFace BERT model identifier.
            num_genres: Number of genre categories (ML-1M has 18).
            num_eras: Number of era categories.
            num_activity_levels: Number of activity level categories.
            dropout: Dropout rate for task heads.
            freeze_bert_layers: Number of BERT layers to freeze from bottom.
                                0 = fine-tune all layers.
        """
        super().__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden = self.bert.config.hidden_size  # 768 for bert-base

        # Optionally freeze lower BERT layers for efficiency
        if freeze_bert_layers > 0:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            for i in range(min(freeze_bert_layers, len(self.bert.encoder.layer))):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = False

        # ============================================================
        # Multi-Task Heads
        # ============================================================

        # Head 1: Genre Affinity Regression
        # Predicts average rating (0-5) for each of the 18 genres
        self.genre_head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_genres),
        )

        # Head 2: Activity Level Classification (low / medium / high)
        self.activity_head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_activity_levels),
        )

        # Head 3: Preference Era Classification (classic / 80s / 90s / 2000s)
        self.era_head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_eras),
        )

        # Head 4: Rating Tendency Regression (scalar: generous ↔ harsh)
        self.tendency_head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        self._num_genres = num_genres
        self._num_eras = num_eras
        self._num_activity_levels = num_activity_levels

    def forward(self, input_ids, attention_mask):
        """
        Forward pass.

        Args:
            input_ids: Tokenized input (batch_size, seq_len).
            attention_mask: Attention mask (batch_size, seq_len).

        Returns:
            dict with keys:
              - cls_embedding: (batch_size, 768) — BERT [CLS] representation
              - genre_affinity: (batch_size, 18) — predicted genre ratings
              - activity_level: (batch_size, 3) — activity class logits
              - preference_era: (batch_size, 4) — era class logits
              - rating_tendency: (batch_size, 1) — tendency z-score
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        return {
            "cls_embedding": cls_embedding,
            "genre_affinity": self.genre_head(cls_embedding),
            "activity_level": self.activity_head(cls_embedding),
            "preference_era": self.era_head(cls_embedding),
            "rating_tendency": self.tendency_head(cls_embedding),
        }


class I2PBERTLoss(nn.Module):
    """
    Multi-task loss for I2P-BERT.

    Combines regression (MSE) and classification (CrossEntropy) losses
    with configurable weights per task.
    """

    def __init__(self, weights: dict = None):
        """
        Args:
            weights: dict mapping task name → loss weight float.
                     Default: genre=1.0, activity=0.5, era=0.5, tendency=0.3
        """
        super().__init__()
        self.weights = weights or {
            "genre_affinity": 1.0,
            "activity_level": 0.5,
            "preference_era": 0.5,
            "rating_tendency": 0.3,
        }
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, predictions: dict, targets: dict) -> dict:
        """
        Compute multi-task loss.

        Args:
            predictions: dict from I2PBERT.forward()
            targets: dict with keys:
              - genre_affinities: (batch, 18) float tensor
              - activity_label: (batch,) long tensor
              - era_label: (batch,) long tensor
              - tendency_score: (batch, 1) float tensor

        Returns:
            dict with individual losses and total_loss.
        """
        # Regression losses
        genre_loss = self.mse(predictions["genre_affinity"], targets["genre_affinities"])
        tendency_loss = self.mse(
            predictions["rating_tendency"], targets["tendency_score"]
        )

        # Classification losses
        activity_loss = self.ce(predictions["activity_level"], targets["activity_label"])
        era_loss = self.ce(predictions["preference_era"], targets["era_label"])

        # Weighted total
        total = (
            self.weights["genre_affinity"] * genre_loss
            + self.weights["activity_level"] * activity_loss
            + self.weights["preference_era"] * era_loss
            + self.weights["rating_tendency"] * tendency_loss
        )

        return {
            "total_loss": total,
            "genre_loss": genre_loss.item(),
            "activity_loss": activity_loss.item(),
            "era_loss": era_loss.item(),
            "tendency_loss": tendency_loss.item(),
        }
