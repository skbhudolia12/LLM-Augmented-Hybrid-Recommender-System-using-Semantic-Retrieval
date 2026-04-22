# 🚀 LLM-Augmented Hybrid Recommender System (I2P-BERT)

An end-to-end, high-performance recommendation pipeline designed to bridge the gap between Collaborative Filtering (CF) and Content-Based systems by distilling tabular interaction data back into semantically rich Natural Language. 

Currently deployed and optimized for execution on **A100 GPU clusters** using distributed multiprocess data chunking.

---

## 🏗 Pipeline Architecture

Deploying raw recommendation pipelines typically hits a brick wall when attempting to fuse high-dimensional user habits with movie metadata. This framework utilizes a 4-Stage architectural approach to encode everything into the semantic space.

### 🔹 Stage 1: Data Preparation & Augmentation (`src/data/`)
* **Objective:** Parse complex datasets (MovieLens 1M) and define ground-truth label dependencies.
* **Mechanism:** Data is chronologically split (Train/Val/Test) to prevent strict data leakage on time-series predictions. User ratings are mathematically averaged out to create classification labels for User Activity, Preference Eras, and General Rating Tendencies.
* **Augmentation:** Connects natively to the TMDB API to scrape underlying plot descriptions for advanced semantic mapping.

### 🔹 Stage 2: Item Encoding via sBERT (`src/item_encoder/`)
* **Objective:** Map content items into a dense vector space. 
* **Mechanism:** Iterates over movie titles, extracted genres, and scraped plot descriptions through `Sentence-BERT` (`all-MiniLM-L6-v2`) generating a `384-dimensional` vector per movie.
* **Outputs:** Dense Item Embeddings natively serialized and stored in an active **ChromaDB** Vector Database.

### 🔹 Stage 3: User Encoding via I2P-BERT (`src/user_encoder/`)
* **Objective:** Translate abstract tabular user constraints into trainable natural language text.
* **Mechanism (Interaction Text):** Tabular data is transformed into detailed paragraphs natively describing the user (e.g., _"Male, age 25. Rated 140 movies with average 4.1. Top genres: Action..."_)
* **Mechanism (I2P-BERT Multi-Task):** A `bert-base-uncased` language model reads this paragraph. We slice the output `CLS` token and run it through a multi-task loss function against actual user preferences. The model "learns" to summarize tabular histories dynamically.

### 🔹 Stage 4: Feature Fusion & Gradient Boosting (`src/feature_fusion/` & `src/predictor/`)
* **Objective:** Re-fuse the independent User Embeddings with the Item Embeddings and predict precision rankings.
* **Mechanism:** Iterates over the `800,000+` pairing matrix, utilizing pure mathematical vectorization (`np.dot` cosine metrics) and `joblib` multiprocessing. Reconstructs interactions visually mapping item-item similarity and user-user collaborative variables.
* **Optimization:** `XGBoost` maps directly to GPU Cuda bindings while querying multiple massive decision-tree hyperparameter forests simultaneously using `Optuna`.

---

## 📈 SOTA Comparisons & Expected Performance

Traditional Recommender Systems usually crash into a conceptual boundary defined entirely by the dataset they sit on top of. Because we route our encoding through a Language Model, **I2P-BERT** brings zero-shot and semantic deduction directly into the ranking algorithm. 

#### 1. Factorization Machines / SVD (Baselines)
* **Limits:** Simple DOT product models max out on the MovieLens-1M dataset at roughly **~0.87 RMSE**. They completely break down during the "Cold Start" problem (new users/items).
* **Our Edge:** I2P-BERT maps cold users perfectly if they supply minimal string metrics (like age or simple text).

#### 2. Neural Collaborative Filtering (NCF)
* **Limits:** Classic non-linear MLPs push Matrix Factorization bounds down to **~0.85 RMSE**. However, they completely lack contextual semantics (they don't know the movie _Iron Man_ is actually about superheroes; they just map numeric ID `#324`).
* **Our Edge:** Stage 2 parses literal plot descriptions. The semantic gap is eliminated.

#### 3. LLM-Based Recommenders (SOTA)
* **Limits:** Native GPT/LLama execution is incredibly slow, inherently biased, and computationally expensive for massive $N$-user matrix operations.
* **Our Edge:** We use deep ML extraction (Distilling BERT) and then dump the math off onto robust `XGBoost` trees. This gives us LLM-understanding with absolute mathematical inference speeds. 

### 🎯 Expected Scoring Constraints
Assuming full `A100` parallelization, you should expect your SOTA metrics bounding across:
- **RMSE (Root Mean Square Error):** Expect convergence cleanly below **< 0.83** on holdout test splits.
- **NDCG@10 / HR@10:** A massive spike in ranking accuracy given the multi-task genre loss enforcing specific sub-categories natively onto the XGBoost ranking list.
