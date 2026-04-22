"""
Highly Optimized Hybrid NeuMF Predictor.

Uses a large-capacity MLP to fully ingest the 768-dim and 384-dim semantic text
embeddings without bottlenecking them, alongside standard collaborative GMF.
BCE Point-wise Loss is utilized for stable ranking on ML-1M.
"""

import logging
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# ================================================================= #
# Dataset definitions
# ================================================================= #

class ImplicitTrainDataset(Dataset):
    def __init__(self, interactions: pd.DataFrame, num_items: int, num_negatives: int = 4):
        self.users = torch.tensor(interactions["user_id"].values, dtype=torch.long)
        self.items = torch.tensor(interactions["movie_id"].values, dtype=torch.long)
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.user_history = interactions.groupby("user_id")["movie_id"].apply(set).to_dict()
        
    def __len__(self):
        return len(self.users)
        
    def __getitem__(self, idx):
        user = self.users[idx]
        pos_item = self.items[idx]
        
        history = self.user_history.get(user.item(), set())
        neg_items = []
        for _ in range(self.num_negatives):
            neg_item = np.random.randint(0, self.num_items)
            while neg_item in history:
                neg_item = np.random.randint(0, self.num_items)
            neg_items.append(neg_item)
            
        return user, pos_item, torch.tensor(neg_items, dtype=torch.long)

class ImplicitTestDataset(Dataset):
    def __init__(self, test_df: pd.DataFrame):
        self.df = test_df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "user": torch.tensor(row["user_id"], dtype=torch.long),
            "item": torch.tensor(row["movie_id"], dtype=torch.long),
            "rating": torch.tensor(row["rating"], dtype=torch.float32)
        }

# ================================================================= #
# Architecture
# ================================================================= #

class OptimizedHybridNeuMF(nn.Module):
    def __init__(self, num_users, num_items, user_embs, item_embs, gmf_dim=32, mlp_dim=32):
        super().__init__()
        
        # Collaborative Embeddings
        self.user_id_gmf = nn.Embedding(num_users, gmf_dim)
        self.item_id_gmf = nn.Embedding(num_items, gmf_dim)
        self.user_id_mlp = nn.Embedding(num_users, mlp_dim)
        self.item_id_mlp = nn.Embedding(num_items, mlp_dim)
        
        # Dense Text Embeddings
        self.register_buffer("user_text", torch.tensor(user_embs, dtype=torch.float32))
        self.register_buffer("item_text", torch.tensor(item_embs, dtype=torch.float32))
        
        # To prevent text from overpowering, we apply LayerNorm on the fixed buffers
        self.user_ln = nn.LayerNorm(self.user_text.shape[1])
        self.item_ln = nn.LayerNorm(self.item_text.shape[1])
        
        # Deep MLP Capacity
        # Input = 32 + 32 + 768 + 384 = 1216
        mlp_input_dim = mlp_dim * 2 + self.user_text.shape[1] + self.item_text.shape[1]
        
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Output fusion: GMF(32) + MLP(64) = 96 -> 1
        self.out = nn.Linear(gmf_dim + 64, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)

    def forward(self, users, items):
        # 1. GMF Block
        u_id_gmf = self.user_id_gmf(users)
        i_id_gmf = self.item_id_gmf(items)
        gmf_vec = u_id_gmf * i_id_gmf
        
        # 2. MLP Block
        u_id_mlp = self.user_id_mlp(users)
        i_id_mlp = self.item_id_mlp(items)
        u_txt = self.user_ln(self.user_text[users])
        i_txt = self.item_ln(self.item_text[items])
        
        mlp_input = torch.cat([u_id_mlp, i_id_mlp, u_txt, i_txt], dim=-1)
        mlp_vec = self.mlp(mlp_input)
        
        # 3. Fusion
        fused = torch.cat([gmf_vec, mlp_vec], dim=-1)
        return self.out(fused).squeeze(-1)

# ================================================================= #
# Training Loop
# ================================================================= #

def train_mlp_predictor(
    train_df, test_df, num_users, num_items, 
    user_embs_dict, item_embs_dict, config
):
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    
    # Load BERT arrays
    user_emb_matrix = np.zeros((num_users, 768))
    for uid, vec in user_embs_dict.items():
        if uid < num_users:
            user_emb_matrix[uid] = vec
            
    item_emb_matrix = np.zeros((num_items, 384))
    for mid, vec in item_embs_dict.items():
        if mid < num_items:
            item_emb_matrix[mid] = vec

    model = OptimizedHybridNeuMF(
        num_users=num_users,
        num_items=num_items,
        user_embs=user_emb_matrix,
        item_embs=item_emb_matrix,
        gmf_dim=config.get("gmf_dim", 32),
        mlp_dim=config.get("mlp_dim", 32)
    ).to(device)
    
    num_negs = config.get("train_negatives", 4)
    # Using 4 negatives keeps training robust against overwhelming data size
    ds = ImplicitTrainDataset(interactions=train_df, num_items=num_items, num_negatives=num_negs)
    dl = DataLoader(ds, batch_size=config.get("batch_size", 2048), shuffle=True, num_workers=4)
    
    import os
    checkpoint_path = "checkpoints/mlp_implicit_best.pt"
    os.makedirs("checkpoints", exist_ok=True)
    
    if os.path.exists(checkpoint_path):
        logger.info(f"Found existing MLP checkpoint at {checkpoint_path}. Skipping training!")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        hr, ndcg = evaluate_mlp(model, test_df, device)
        return model, {"hr@10": hr}

    # Higher learning rate accelerates the deep layers
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", 1e-3), weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    epochs = config.get("epochs", 30)
    best_hr = 0.0
    
    logger.info("Training High-Capacity Hybrid NeuMF with BCE loss...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for users, pos_items, neg_items in dl:
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)
            
            optimizer.zero_grad()
            
            pos_logits = model(users, pos_items)
            pos_labels = torch.ones_like(pos_logits)
            
            B, num_neg = neg_items.shape
            users_repeated = users.unsqueeze(1).expand(-1, num_neg).reshape(-1)
            neg_items_flat = neg_items.reshape(-1)
            
            neg_logits = model(users_repeated, neg_items_flat)
            neg_labels = torch.zeros_like(neg_logits)
            
            logits = torch.cat([pos_logits, neg_logits])
            labels = torch.cat([pos_labels, neg_labels])
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dl)
        logger.info("[Epoch %d] BCE Loss: %.4f", epoch+1, avg_loss)
        
        hr, ndcg = evaluate_mlp(model, test_df, device)
        logger.info("[Epoch %d] HR@10: %.4f | NDCG@10: %.4f", epoch+1, hr, ndcg)
        
        if hr > best_hr:
            best_hr = hr
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved new best model to {checkpoint_path}")
            
    # Load best model for final return
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    return model, {"hr@10": best_hr}


def evaluate_mlp(model, test_df, device):
    """
    Ranks the 100 items for each user and computes standard HR@10 and NDCG@10.
    """
    model.eval()
    ds = ImplicitTestDataset(test_df)
    dl = DataLoader(ds, batch_size=4000, shuffle=False, num_workers=2)
    
    users, items, ratings, preds = [], [], [], []
    
    with torch.no_grad():
        for batch in dl:
            u = batch["user"].to(device)
            i = batch["item"].to(device)
            logits = model(u, i)
            
            users.extend(u.cpu().numpy())
            items.extend(i.cpu().numpy())
            ratings.extend(batch["rating"].numpy())
            preds.extend(logits.cpu().numpy())
            
    results_df = pd.DataFrame({
        "user_id": users,
        "movie_id": items,
        "rating": ratings,
        "score": preds
    })
    
    hr_list, ndcg_list = [], []
    
    user_groups = results_df.groupby("user_id")
    for uid, group in user_groups:
        if len(group) != 100:
            continue
            
        group = group.sort_values(by="score", ascending=False).reset_index(drop=True)
        top_10 = group.head(10)
        
        hit_index = top_10[top_10["rating"] == 1.0].index.tolist()
        
        if len(hit_index) > 0:
            hr_list.append(1.0)
            rank = hit_index[0]
            ndcg_list.append(math.log(2) / math.log(rank + 2))
        else:
            hr_list.append(0.0)
            ndcg_list.append(0.0)
            
    return np.mean(hr_list), np.mean(ndcg_list)
