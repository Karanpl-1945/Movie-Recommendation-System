# Movie Recommendation System API

A **hybrid movie recommendation system** served through a **FastAPI** REST API, with a **Streamlit** demo UI. Give it a movie title, get back similar movies — powered by a blend of three different signals rather than a single similarity score.

---

## Overview

The naive version of this kind of project computes TF-IDF similarity on genres/tags and calls it done. This one blends **three independent signals** into a single ranking, each capturing something the others can't:

| Signal | What it captures | Where it comes from |
|---|---|---|
| **Content-based similarity** | "These movies are *about* similar things" (genre, tags) | TF-IDF on tags + one-hot genres → cosine similarity |
| **Item-based collaborative filtering (CF)** | "People who watched this also watched that" | Movie-movie similarity learned from `ratings.csv` |
| **Weighted popularity** | "This movie is reliably good, not just a fluke" | Bayesian-shrinkage average rating |

These are combined into one score per candidate movie:

```
score = α · content_similarity + β · item_CF_similarity + γ · weighted_popularity
```

with tuned weights **α = 0.1, β = 0.4, γ = 0.5** (see [Evaluation](#evaluation--how-the-weights-were-chosen) below for how these were picked, not guessed).

---

## Why three signals instead of one

Each signal fails differently on its own:

- **Content-only** can't tell that people who like *Inception* also tend to like *The Matrix* unless that connection shows up in the text — it only knows "both are sci-fi."
- **CF-only** doesn't work for a brand-new movie with zero ratings — there's no co-rating history to learn from.
- **Popularity-only** just recommends the same blockbusters to everyone, regardless of what was asked.

Blending them means the system still works for new/rarely-rated movies (content), still captures behavioral patterns TF-IDF can't see (CF), and doesn't rank a 5.0-from-2-ratings movie above a 4.5-from-50,000-ratings movie (weighted popularity).

This mirrors how production recommenders (Netflix/YouTube-style) are actually built: multiple candidate sources feeding one ranking step. See [`movie_recommendation_system.md`](movie_recommendation_system.md) for the full reference on recommendation-system techniques, including how this maps to a real production pipeline (§9).

---

## How each signal is built

### 1. Content-based similarity
- Movie **tags** (free-text, from `tags.csv`) are cleaned (lowercased, stopwords removed, stemmed) and vectorized with **TF-IDF** (top 5,000 terms).
- Movie **genres** (`movies.csv`) are one-hot encoded with `MultiLabelBinarizer`.
- Both feature sets are concatenated and compared pairwise with **cosine similarity** → one movie-by-movie similarity matrix.

### 2. Item-based collaborative filtering
- Built from `ratings.csv` (32 million ratings): a sparse **movies × users** matrix, where each user's ratings are **mean-centered** first (adjusted cosine) so a user who rates everything 5★ doesn't distort things.
- Reduced with **TruncatedSVD** (50 latent dimensions) into a compact per-movie embedding.
- At request time, the query movie's embedding is compared against all others with cosine similarity — cheap enough to compute per-request, so only a ~15MB embeddings file is stored instead of a full 2.8GB movie-movie matrix.

### 3. Weighted popularity
- Plain average rating is misleading — a movie rated 5.0 by 2 people shouldn't outrank one rated 4.5 by 50,000. Fixed with the IMDB-style **Bayesian shrinkage** formula:

  ```
  weighted_rating = (v / (v + m)) · R + (m / (v + m)) · C
  ```
  where `v` = number of ratings, `R` = the movie's average rating, `m` = the 90th-percentile rating count (minimum votes to be trusted), `C` = the global mean rating. Movies with few ratings get pulled toward the global average; only as evidence (`v`) grows does the movie's own average dominate.

---

## Evaluation — how the weights were chosen

Rather than guessing α/β/γ, they were **grid-searched against held-out data** (`models/evaluate.py`):

1. **Temporal per-user split**: each eligible user's ratings are sorted by time; the earliest ~80% is "train," the most recent ~20% is "test."
2. Multiple **seed movies per user** (up to 5 liked movies from their train set, not just their single favorite) are used as recommendation queries.
3. For each seed, the top-10 recommendations are generated and checked against the user's held-out liked movies, measuring **Precision@10**, **Recall@10**, and **NDCG@10**.
4. This is repeated across a **grid of (α, β, γ) combinations**, and additionally **stratified** into popular-seed vs. niche-seed buckets, to check whether the hybrid genuinely helps less-popular movies or is just riding popularity bias.

**Result:** the metric-maximizing combination pushed α toward 0 and γ toward 1 in every stratum tested (even niche movies) — but that reflects **popularity bias in the metric itself** (popular movies get rated highly by almost everyone, regardless of what was queried), not truly better recommendations. The final weights (`α=0.1, β=0.4, γ=0.5`) sit close to the measured optimum while keeping a small, deliberate content-based contribution — the only signal that can work for a movie with **zero** ratings, a case that, by definition, can never appear in a ratings-history-based evaluation like this one.

---

## Dataset

Uses the **[MovieLens 32M dataset](https://grouplens.org/datasets/movielens/)** from GroupLens — 32 million ratings, ~87,000 movies, 2 million tag applications.

Download and extract it into:

```
task4/ml-32m/
    movies.csv
    ratings.csv
    tags.csv
    links.csv
```

---

## Large files not included in the repository

These are generated locally, not committed (GitHub's 100MB file limit, and they're multi-GB):

```
task4/model/
    similarity.pkl        (~2.8 GB)  content-based movie-movie similarity matrix
    movies_data.pkl       (~40 MB)   processed movie metadata + weighted ratings
    cf_embeddings.pkl     (~7.5 MB)  item-CF SVD embeddings
```

---

## Setup

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download the dataset

Place the MovieLens 32M CSVs in `task4/ml-32m/` (see [Dataset](#dataset) above).

### 3. Generate the model artifacts

Either run the standalone script:

```bash
python models/generate_model.py
```

or run `models/model.ipynb` cell-by-cell (same pipeline, plus EDA plots — genre distribution, rating distributions, long-tail ratings-per-movie/user, tag frequency).

Both produce `model/similarity.pkl`, `model/movies_data.pkl`, and `model/cf_embeddings.pkl`.

### 4. (Optional) Evaluate and re-tune the hybrid weights

```bash
python models/evaluate.py
```

Prints Precision@10/Recall@10/NDCG@10 across a weight grid, plus a popularity-stratified breakdown. Adjust `ALPHA`, `BETA`, `GAMMA` in `models/recommend.py` if you regenerate the model against a different dataset slice.

### 5. Run the API

```bash
uvicorn app:app --reload
```

Server starts at `http://127.0.0.1:8000`. Interactive docs (Swagger UI) at `http://127.0.0.1:8000/docs`.

### 6. Run the Streamlit demo

```bash
streamlit run streamlit_app.py
```

The demo opens in your browser and uses the same `models/recommend.py` logic as the API. It also shows a per-result `confidence` score alongside each recommendation.

---

## API

### `POST /recommend`

Request:

```json
{
  "movie": "Toy Story",
  "no_recommendation": 5
}
```

- `movie` — title match is case/whitespace-insensitive, and **without the year suffix** (titles are stored as `"Toy Story"`, not `"Toy Story (1995)"`).
- `no_recommendation` — optional, defaults to 5, must be between 1 and 19.

Response:

```json
{
  "query_movie": "toy story",
  "recommendations": [
    {
      "movieId": 3114,
      "title": "Toy Story 2",
      "average_rating": 3.81,
      "rating_count": 32683,
      "confidence": 0.873
    }
  ]
}
```

Returns `404` if the movie title isn't found in the modeled dataset (movies with ≤30 ratings are filtered out during model generation).

### `GET /health_check`

Basic liveness check.

---

## Project structure

```
task4/
├── app.py                       FastAPI app (routes: /, /health_check, /recommend)
├── streamlit_app.py             Streamlit demo UI for the recommender
├── requirements.txt
├── README.md
├── movie_recommendation_system.md   Reference notes on recommender techniques
│
├── schema/
│   ├── input.py                 Request schema (movie, no_recommendation)
│   └── response.py               Response schema
│
├── models/
│   ├── recommend.py              Loads model artifacts, serves hybrid recommendations
│   ├── generate_model.py         Standalone pipeline: EDA + content/CF/popularity model building
│   ├── evaluate.py               Offline evaluation + weight grid search
│   └── model.ipynb                Notebook version of generate_model.py, with EDA plots
│
├── ml-32m/                        MovieLens 32M dataset (not committed)
└── model/                         Generated artifacts (not committed)
    ├── similarity.pkl
    ├── movies_data.pkl
    └── cf_embeddings.pkl
```

---

## Technologies used

Python · FastAPI · Pydantic · scikit-learn · pandas · NumPy · scipy · NLTK · Uvicorn

---

## Known limitations

- Title matching is exact (case/whitespace-insensitive only) — no fuzzy search, so typos or partial titles won't match.
- The evaluation's niche-movie sample is still fairly small (46 cases) relative to the popular-movie sample (~7,000) — see `models/evaluate.py`'s output for the full breakdown.
- True cold-start (a movie with zero ratings) is not empirically validated by the offline evaluation, since such movies can't appear in a ratings-history-based test set — the small `α` weight is a deliberate hedge, not a measured result.
