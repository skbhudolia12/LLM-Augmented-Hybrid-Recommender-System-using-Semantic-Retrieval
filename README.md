# LLM-Augmented Hybrid Recommender System using Semantic Retrieval

A novel hybrid recommendation pipeline that frames collaborative user profiling as a structured multi-task prediction problem from interaction-derived natural language. 

## System Architecture

1. **I2P-BERT User Encoder**: Translates tabular user interaction history (demographics, rating history) into natural language, then uses a fine-tuned BERT model with multi-task heads to predict structured genre affinities, activity levels, preference eras, and rating tendencies. Profiles are stored in SQLite.
2. **sBERT Item Encoder**: Encodes movie metadata (title + genres + TMDB plot summary) into 384-dimensional dense semantic vectors using Sentence-Transformers, stored in a ChromaDB vector database.
3. **Feature Fusion**: Combines I2P-BERT user profiles, sBERT item embeddings, cross-features (cosine similarity, genre overlap), and k-NN collaborative signals into a unified dense feature vector (~580 dimensions).
4. **XGBoost Rating Predictor**: Uses the fused feature vectors to train an XGBoost regressor (with Optuna hyperparameter tuning) to predict discrete user ratings (1-5). SHAP analysis provides explicit interpretability.
5. **SLM Module (Future Phase)**: Employs a fine-tuned Small Language Model to extract deep-reasoning properties.

## Setup & Execution

### 1. Installation
```powershell
# Clone the repository
git clone https://github.com/skbhudolia12/LLM-Augmented-Hybrid-Recommender-System-using-Semantic-Retrieval.git
cd LLM-Augmented-Hybrid-Recommender-System-using-Semantic-Retrieval

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration & API Keys
Before running the item encoder, you must configure your TMDB API Key to retrieve plot summaries:
1. Register for an API key at [TMDB](https://www.themoviedb.org/settings/api).
2. Open `config/config.yaml`.
3. Locate the `data:` section and update `tmdb_api_key:`
   ```yaml
   data:
     tmdb_api_key: "YOUR_API_KEY_HERE"  # <-- Paste the v3 API Key (hex string) here, NOT the JWT Read Access Token.
   ```
4. **Configure Hardware**: In the same `config.yaml`, set `device: "cuda"` if running on an Nvidia GPU (e.g. an A100 server) or `device: "cpu"` if testing locally.

### 3. Running the Pipeline
You can run the entire pipeline end-to-end with a single command:
```powershell
python -m scripts.run_pipeline
```

If you don't have a TMDB API Key, face network timeouts during fetching, or just want to quickly test the pipeline baseline without TMDB data, you can bypass the augmentation entirely:
```powershell
python -m scripts.run_pipeline --skip-tmdb
```

To resume from a specific stage if previous stages are already cached (e.g., jump straight to Stage 3 I2P-BERT Training):
```powershell
python -m scripts.run_pipeline --stage 3
```

## Project Structure
- `config/config.yaml`: Central hyperparameters, device mappings, and secrets.
- `scripts/run_pipeline.py`: Main execution orchestrator separating logic into stages.
- `src/data/`: Auto ML-1M downloading, timestamp-based splitting, and TMDB plot augmentation.
- `src/user_encoder/`: I2P-BERT architecture defining natural language generation, BERT tokenizer configurations, multi-task tracking, and PyTorch dataloading + training loops.
- `src/item_encoder/`: Semantic processing encoding and scaling up to ChromaDB vector stores.
- `src/feature_fusion/`: Generates matrices composing user vectors, item vectors, cross features, and neighborhood interactions.
- `src/predictor/`: XGBoost predictor coupled with Optuna parameter tuning and automated SHAP value plotting for pipeline transparency.
- `src/evaluation/`: Implementations of multiple baselines (e.g. SVD, Global Mean).
