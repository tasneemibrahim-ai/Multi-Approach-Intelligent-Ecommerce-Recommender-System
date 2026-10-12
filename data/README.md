# data/

Place real datasets here to swap away from the synthetic dataset.

## Option 1 — MovieLens (small, recommended for testing)
Download: https://grouplens.org/datasets/movielens/latest/
Files needed:
  - ratings.csv  (columns: userId, movieId, rating, timestamp)
  - movies.csv   (columns: movieId, title, genres)

Then in main.py or streamlit_app.py change:
    load_dataset(source="movielens")

## Option 2 — Amazon Product Reviews
Download: https://nijianmo.github.io/amazon/index.html
Files needed:
  - amazon_ratings.csv  (columns: user_id, item_id, rating)
  - amazon_products.csv (columns: item_id, name, category, keywords, description, price, brand)

Then change:
    load_dataset(source="amazon")

## Default — Synthetic Dataset
No files needed. The system auto-generates 200 users × 36 products × 3000 ratings
with realistic category preferences. Change source="synthetic" (default).
