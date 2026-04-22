"""
SLM Persona-Based Ranking Predictor (Point-wise Implicit).
Fine-tunes Phi-3-mini-4k-instruct with LoRA to predict if a user will interact
with an item given the user's persona text + movie description.
Outputs a scalar ranking score (logit of 'Yes' token) to sort 100 test items per user.
"""

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# ================================================================ #
# Prompt Construction
# ================================================================ #

def build_item_text(movie_row, genre_list):
    """Build movie description from DataFrame row."""
    title = movie_row.get("title", "Unknown")
    year = movie_row.get("year", "")
    genres = [g for g in genre_list if movie_row.get(f"genre_{g}", 0) == 1]
    genre_str = ", ".join(genres) if genres else "Unknown"
    return f"{title} ({int(year)}) | Genres: {genre_str}"

def build_prompt(persona_text, item_text):
    """Construct the SLM prompt for interaction prediction."""
    return (
        f"Based on this user's profile and preferences, will they watch the given movie?\n\n"
        f"User Profile: {persona_text}\n\n"
        f"Movie: {item_text}\n\n"
        f"Answer 'Yes' or 'No':"
    )

# ================================================================ #
# Dataset
# ================================================================ #

class SLMImplicitDataset(Dataset):
    """Dataset class mapping prompt strings to 0/1 tensors."""
    def __init__(self, df, user_texts, item_texts, tokenizer, max_length=256, is_train=True, num_negatives=4):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.user_texts = user_texts
        self.item_texts = item_texts
        
        # When evaluating, test_df has 100 entries per user (1 pos, 99 neg).
        # We can just iterate through them linearly.
        # But if it's train_df, it only contains positives. So we sample negatives.
        self.is_train = is_train
        
        if self.is_train:
            self.users = df["user_id"].values
            self.items = df["movie_id"].values
            self.all_items = list(item_texts.keys())
            self.history = df.groupby("user_id")["movie_id"].apply(set).to_dict()
            self.num_negatives = num_negatives
        else:
            self.df = df
            
    def __len__(self):
        if self.is_train:
            return len(self.users)
        return len(self.df)

    def __getitem__(self, idx):
        if self.is_train:
            uid = self.users[idx]
            pos_mid = self.items[idx]
            
            # One positive, and one negative randomly sampled per positive for SLM
            # (To save VRAM/time, we only do 1 negative during SLM LoRA training per pos)
            neg_mid = np.random.choice(self.all_items)
            while neg_mid in self.history.get(uid, set()):
                neg_mid = np.random.choice(self.all_items)
                
            pos_prompt = build_prompt(self.user_texts.get(uid, ""), self.item_texts.get(pos_mid, ""))
            neg_prompt = build_prompt(self.user_texts.get(uid, ""), self.item_texts.get(neg_mid, ""))
            
            return {
                "pos_prompt": pos_prompt,
                "neg_prompt": neg_prompt,
                "pos_label": 1.0,
                "neg_label": 0.0
            }
        else:
            row = self.df.iloc[idx]
            uid = row["user_id"]
            mid = row["movie_id"]
            label = float(row["rating"])
            
            prompt = build_prompt(self.user_texts.get(uid, ""), self.item_texts.get(mid, ""))
            
            encoding = self.tokenizer(
                prompt, truncation=True, max_length=self.max_length,
                padding="max_length", return_tensors="pt"
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "label": torch.tensor(label, dtype=torch.float32),
                "user_id": uid,
                "movie_id": mid
            }

# ================================================================ #
# Training & Evaluation 
# ================================================================ #

def get_yes_logit(logits, yes_token_id):
    """
    Extract the raw logit score of the 'Yes' token prediction at the end of the sequence.
    This serves as our continuous ranking score.
    """
    return logits[:, -1, yes_token_id]

def train_slm_predictor(
    train_df, test_df, user_texts, item_texts, config, save_dir="checkpoints/slm_implicit"
):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    model_name = config.get("model_name", "microsoft/Phi-3-mini-4k-instruct")
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    
    logger.info("Loading SLM tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    logger.info(f"Loading SLM model in bfloat16 with {config.get('attn_implementation', 'sdpa')}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation=config.get("attn_implementation", "sdpa"),
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    
    # ID for "Yes" token to be used for BCE loss
    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    
    max_len = config.get("max_seq_length", 256)
    batch_size = config.get("batch_size", 8)
    grad_accum = config.get("gradient_accumulation_steps", 4)
    
    # Subsample training data for speed
    max_train = config.get("max_train_samples", 50000)
    if len(train_df) > max_train:
        logger.info(f"Subsampling training interactions from {len(train_df)} to {max_train}")
        train_df_sampled = train_df.sample(n=max_train, random_state=42)
    else:
        train_df_sampled = train_df

    train_ds = SLMImplicitDataset(
        train_df_sampled, user_texts, item_texts, tokenizer, max_length=max_len, is_train=True
    )
    # Use smaller batch sizes for raw text forward passes
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 2e-4)))
    criterion = torch.nn.BCEWithLogitsLoss()
    
    epochs = config.get("epochs", 2)
    logger.info("Starting SLM LoRA point-wise ranking training (%d epochs).", epochs)
    
    best_hr = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for step, batch in enumerate(train_loader):
            # Tokenize pos and neg combined
            prompts = list(batch["pos_prompt"]) + list(batch["neg_prompt"])
            labels = torch.cat([torch.ones(batch_size), torch.zeros(batch_size)]).to(device)
            
            encodings = tokenizer(
                prompts, truncation=True, max_length=max_len,
                padding="max_length", return_tensors="pt"
            ).to(device)
            
            outputs = model(**encodings)
            yes_logits = get_yes_logit(outputs.logits, yes_token_id)
            
            loss = criterion(yes_logits, labels) / grad_accum
            loss.backward()
            total_loss += loss.item() * grad_accum
            
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
            if (step + 1) % (grad_accum * 50) == 0:
                logger.info("[Epoch %d | Step %d] Loss: %.4f", epoch + 1, step + 1, total_loss / (step + 1))
        
        # Eval Phase
        hr, ndcg = evaluate_slm(model, tokenizer, yes_token_id, test_df, user_texts, item_texts, config, device)
        logger.info("[Epoch %d] SLM HR@10: %.4f | NDCG@10: %.4f", epoch + 1, hr, ndcg)
        
        if hr > best_hr:
            best_hr = hr
            model.save_pretrained(save_path / "best_lora")
            
    return model, {"hr@10": best_hr}


def evaluate_slm(model, tokenizer, yes_token_id, test_df, user_texts, item_texts, config, device):
    """
    Ranks 100 items per user by forward passing each and tracking 'Yes' logits.
    """
    model.eval()
    max_len = config.get("max_seq_length", 256)
    
    # Subsample evaluation users for speed
    max_users = config.get("max_eval_users", 500)
    unique_users = test_df["user_id"].unique()
    if len(unique_users) > max_users:
        logger.info(f"Subsampling evaluation users from {len(unique_users)} to {max_users}")
        sampled_users = np.random.choice(unique_users, max_users, replace=False)
        test_df_sampled = test_df[test_df["user_id"].isin(sampled_users)]
    else:
        test_df_sampled = test_df

    test_ds = SLMImplicitDataset(
        test_df_sampled, user_texts, item_texts, tokenizer, max_length=max_len, is_train=False
    )
    # Eager attention uses more memory, keep batch size small
    eval_batch = config.get("batch_size", 8)
    dl = DataLoader(test_ds, batch_size=eval_batch, shuffle=False, num_workers=2)
    
    users, items, ratings, preds = [], [], [], []
    
    logger.info("Evaluating SLM over 100 items per test user...")
    with torch.no_grad():
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            logits = get_yes_logit(outputs.logits, yes_token_id)
            
            users.extend(batch["user_id"].numpy())
            items.extend(batch["movie_id"].numpy())
            ratings.extend(batch["label"].numpy())
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
            ndcg_list.append(math.log(2) / math.log(hit_index[0] + 2))
        else:
            hr_list.append(0.0)
            ndcg_list.append(0.0)
            
    return np.mean(hr_list), np.mean(ndcg_list)
