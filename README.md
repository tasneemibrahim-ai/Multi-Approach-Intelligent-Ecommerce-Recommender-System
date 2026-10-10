# Intelligent E-Commerce Recommendation System
**Course: AIE425 — Alamein International University**

---

## Project Overview
A fully modular, production-grade intelligent recommender system implementing **3 recommendation approaches** and **9+ algorithms**, with a professional Streamlit dashboard and comprehensive evaluation.

---

## Folder Structure
```
recommender_system/
├── main.py                          # Master integration file (run this first)
├── requirements.txt
│
├── utils/
│   ├── data_loader.py               # Abstract, pluggable data loading
│   └── result_format.py             # Unified RecommendationResult format
│
├── models/
│   ├── collaborative/
│   │   ├── cf_base.py               # Shared Surprise utilities
│   │   ├── user_knn.py              # ① User-Based KNN
│   │   ├── item_knn.py              # ② Item-Based KNN
│   │   ├── svd_model.py             # ③ SVD (Matrix Factorization)
│   │   ├── svdpp_model.py           # ④ SVD++ (Implicit Feedback)
│   │   ├── knn_means.py             # ⑤ KNN with Means (Bias-Corrected)
│   │   └── nmf_model.py             # ⑥ NMF (Non-Negative MF)
│   │
│   ├── content/
│   │   ├── content_base.py          # Text preprocessing & cosine similarity
│   │   ├── tfidf_genre.py           # ① TF-IDF on Categories/Genres
│   │   ├── tfidf_keywords.py        # ② TF-IDF on Keywords/Tags
│   │   └── tfidf_description.py     # ③ TF-IDF on Full Descriptions
│   │
│   └── knowledge/
│       ├── simple_filter.py         # ① Hard constraint filtering
│       ├── rule_based.py            # ② IF-THEN rule engine (8 rules)
│       └── constraint_based.py      # ③ Weighted constraint scoring
│
├── evaluation/
│   └── evaluator.py                 # RMSE, MAE, P@K, R@K, F1@K, Coverage, Diversity
│
├── app/
│   └── streamlit_app.py             # Full Streamlit UI + evaluation dashboard
│
└── data/                            # Place real datasets here (MovieLens / Amazon)
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the CLI demo
```bash
cd recommender_system
python main.py
```

### 3. Launch the Streamlit dashboard
```bash
streamlit run app/streamlit_app.py
```

### 4. Run full evaluation (slower, ~2 min)
```bash
RUN_FULL_EVAL=1 python main.py
```

---

## System Requirements Coverage

| Requirement | Status | Details |
|-------------|--------|---------|
| Collaborative Filtering | ✅ | 6 methods: User-KNN, Item-KNN, SVD, SVD++, KNN-Means, NMF |
| Content-Based | ✅ | 3 modes: Genre, Keywords, Description TF-IDF |
| Knowledge-Based | ✅ | 3 methods: Simple Filter, Rule-Based, Constraint Scoring |
| ≥4 CF methods | ✅ | 6 implemented |
| ≥4 evaluation metrics | ✅ | RMSE, MAE, Precision@K, Recall@K, F1@K, Coverage, Diversity |
| CF vs CF comparison | ✅ | Cross-validation table + bar charts |
| CF vs Content vs Knowledge | ✅ | Full ranking metric comparison |
| Explanations | ✅ | Every recommendation includes a human-readable reason |
| Streamlit UI | ✅ | Professional dark-theme UI with Plotly charts |
| Evaluation dashboard | ✅ | Separate tab with full analysis report |
| Pluggable dataset | ✅ | Change `source=` in `load_dataset()` |

---

## Switching Datasets

### MovieLens
1. Download `ml-latest-small.zip` from https://grouplens.org/datasets/movielens/
2. Extract `ratings.csv` and `movies.csv` to `data/`
3. In `main.py`, change: `load_dataset(source="movielens")`

### Amazon (Electronics)
1. Download from https://nijianmo.github.io/amazon/index.html
2. Rename columns to `user_id`, `item_id`, `rating` and save to `data/amazon_ratings.csv`
3. Change: `load_dataset(source="amazon")`

---

## Algorithms Explained

### Collaborative Filtering
| Method | How It Works | Strength | Weakness |
|--------|-------------|----------|----------|
| User-KNN | Find similar users, use their ratings | Intuitive | Slow at scale |
| Item-KNN | Find similar items to what user rated | Stable | Item cold-start |
| SVD | Latent factor decomposition | Accurate | Less interpretable |
| SVD++ | SVD + implicit feedback | Best accuracy | Slowest |
| KNN-Means | KNN with bias correction | Better than KNN | Still sparse |
| NMF | Non-negative factorization | Interpretable factors | May underfit |

### Content-Based
| Method | Feature Used | Best For |
|--------|-------------|----------|
| TF-IDF Genre | Category name | Category-browsing users |
| TF-IDF Keywords | Product tags | Feature-driven similarity |
| TF-IDF Description | Full description | Semantic similarity |

### Knowledge-Based
| Method | Mechanism | Transparency |
|--------|-----------|-------------|
| Simple Filter | Hard pass/fail on constraints | Maximum |
| Rule-Based | Weighted IF-THEN rules (8 rules) | High |
| Constraint Scoring | Soft weighted constraint satisfaction | High, flexible |

---

## Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| RMSE | √(Σ(pred-actual)²/n) | Rating accuracy — lower is better |
| MAE | Σ|pred-actual|/n | Rating accuracy — lower is better |
| Precision@K | relevant_in_topK / K | Quality of top-K list |
| Recall@K | relevant_in_topK / total_relevant | Completeness of retrieval |
| F1@K | 2·P·R/(P+R) | Harmonic balance of P and R |
| Coverage | unique_recommended / catalog_size | Catalog exploration |
| Diversity | avg pairwise category dissimilarity | Variety within a list |

---

## Team & Course
- **Course:** AIE425 — Intelligent Recommender System
- **University:** Alamein International University
- **Faculty:** Computer Science & Engineering

---

## Author

Tasneem Ibrahim - Rana Sakr - Marc Ihab
