# Netflix-Style Movie Recommendation System

A movie recommendation system built on the MovieLens dataset that combines **collaborative filtering** and **content-based filtering**, including a weighted hybrid recommender that blends both signals.

## Overview

This project implements three recommendation approaches to suggest movies to users:

1. **Collaborative Filtering (item-based)**: recommends movies that are rated similarly by the same users, using item-item cosine similarity on the user-item ratings matrix
2. **Content-Based Filtering**: recommends movies with similar genre profiles
3. **Hybrid Recommender**: a weighted blend of the collaborative and content-based similarity scores

## Dataset

- **Source:** [MovieLens Latest Small Dataset](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip)
- **Movies:** 9,742
- **Ratings:** 100,836
- **Users:** 610

## Methodology

### Content-Based Filtering
- Movie genres converted into TF-IDF vectors
- Cosine similarity computed between movies based on genre overlap
- Recommends movies with similar genre profiles

### Collaborative Filtering (Item-Based)
- Built a user-item ratings matrix
- Computed item-item cosine similarity based on how users rated each pair of movies
- Predicts a user's rating for a movie as a similarity-weighted average of the ratings that user gave to similar movies
- Recommends movies that are rated similarly by the same users

### Hybrid Recommender
- Combines the item-item similarity scores from both approaches using a weighted average:

  `final_score = α × CF_score + (1 − α) × content_score`

- Default weight is α = 0.7 (favoring behavioral signal over genre signal)
- The content score still contributes for movies with few ratings, which helps with the cold-start problem

### Evaluation
- Train/test split (80/20) on the ratings data
- Predicted ratings using weighted similarity scores
- Evaluated using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
- Compared against simple baselines (global mean rating and per-movie mean rating)

## Results

| Model | MSE | RMSE |
|-------|-----|------|
| Baseline: global mean rating | X.XX | X.XX |
| Baseline: per-movie mean rating | X.XX | X.XX |
| Item-based Collaborative Filtering | 0.89 | 0.94 |

> The hybrid recommender is evaluated qualitatively (see example output below). Formal ranking evaluation (Precision@K / Recall@K) is listed under future work.

## Example Output

**For "Forrest Gump (1994)":**

- *Collaborative filtering:* Braveheart, Pulp Fiction, Shawshank Redemption, Jurassic Park, Silence of the Lambs
- *Content-based:* Life Is Beautiful, Train of Life, and other drama/emotional films
- *Hybrid (α = 0.7):* PASTE YOUR HYBRID OUTPUT HERE

This highlights the difference between the approaches: collaborative filtering captures actual viewing behavior patterns, while content-based filtering focuses purely on genre similarity. The hybrid balances the two.

## Tech Stack

`Python` `Pandas` `NumPy` `scikit-learn`

## Key Learnings

- Item-based collaborative filtering suffers from the **cold-start problem** for new users and movies with few ratings
- The ratings matrix is highly **sparse**, which limits the reliability of similarity scores
- Content-based filtering works well for genre-similar recommendations but doesn't capture nuanced user preferences
- Blending both signals in a hybrid lets the content score compensate when collaborative data is thin

## Limitations & Future Work

- Genres are a coarse content feature; richer features (user-assigned tags, plot descriptions) would improve content-based results
- Missing ratings are not mean-centered; mean-centering or matrix factorization (e.g., SVD) would likely improve prediction accuracy
- The 80/20 split is random; a time-based split would better mimic real-world use
- MSE measures rating-prediction accuracy, not recommendation quality; add Precision@K and Recall@K to evaluate ranked recommendations
- Tune the hybrid weight α on validation data instead of using a fixed default

## How to Run

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt` (pandas, numpy, scikit-learn)
3. Open `netflix_recommendation.ipynb` in Google Colab or Jupyter Notebook
4. Run all cells sequentially. The dataset downloads automatically via `wget`
