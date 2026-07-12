"""
Offline evaluation for the hybrid recommender.

Builds a temporal per-user train/test split from ratings.csv, then measures
Precision@K, Recall@K and NDCG@K for a grid of (alpha, beta, gamma) hybrid
blend weights, so the weights used in recommend.py are tuned against held-out
data rather than guessed.

Methodology (item-to-item recommendation, evaluated per user):
  1. For each eligible user, sort their ratings (restricted to movies in our
     modeled set) by time; the earliest ~80% is "train", the most recent
     ~20% is "test".
  2. The user's highest-rated train movie is used as the query ("seed").
  3. We generate top-K recommendations for that seed (excluding movies the
     user already rated in train), and check how many of the user's
     positively-rated (>=4.0) test movies appear in that top-K list.
  4. Metrics are averaged across a sample of users, for every weight combo.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'ml-32m')
MODEL_DIR = os.path.join(BASE_DIR, 'model')

TOP_K = 10
RELEVANCE_THRESHOLD = 4.0
MIN_RATINGS_PER_USER = 10
N_EVAL_USERS = 1500
RANDOM_STATE = 42
MAX_SEEDS_PER_USER = 5


def load_artifacts():
    with open(os.path.join(MODEL_DIR, 'similarity.pkl'), 'rb') as f:
        similarity = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'movies_data.pkl'), 'rb') as f:
        data = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'cf_embeddings.pkl'), 'rb') as f:
        cf_embeddings = pickle.load(f)
    return similarity, data, cf_embeddings


def build_train_test_split(data):
    print("Loading ratings for evaluation split...")
    ratings = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'))

    modeled_movie_ids = set(data['movieId'].tolist())
    ratings = ratings[ratings['movieId'].isin(modeled_movie_ids)]

    counts = ratings.groupby('userId').size()
    eligible_users = counts[counts >= MIN_RATINGS_PER_USER].index

    rng = np.random.RandomState(RANDOM_STATE)
    sampled_users = rng.choice(eligible_users, size=min(N_EVAL_USERS, len(eligible_users)), replace=False)

    ratings = ratings[ratings['userId'].isin(sampled_users)].sort_values(['userId', 'timestamp'])

    train_rows, test_rows = [], []
    for uid, group in ratings.groupby('userId'):
        n = len(group)
        split_at = int(n * 0.8)
        if split_at < 1 or split_at >= n:
            continue
        train_rows.append(group.iloc[:split_at])
        test_rows.append(group.iloc[split_at:])

    train_df = pd.concat(train_rows)
    test_df = pd.concat(test_rows)
    print(f"Eligible users sampled: {len(sampled_users)} | usable after split: {train_df['userId'].nunique()}")
    return train_df, test_df


def precision_recall_ndcg_at_k(recommended_ids, relevant_ids, k):
    recommended_ids = recommended_ids[:k]
    hits = [1 if mid in relevant_ids else 0 for mid in recommended_ids]

    precision = sum(hits) / k
    recall = sum(hits) / len(relevant_ids) if relevant_ids else 0.0

    dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
    idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(relevant_ids))))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return precision, recall, ndcg


def precompute_case_scores(eval_cases, similarity, cf_embeddings):
    """Content and item-CF raw score vectors depend only on the seed movie,
    not on (alpha, beta, gamma) - compute them once and reuse across the
    whole weight grid instead of recomputing per combination."""
    precomputed = []
    for seed_idx, seen_idx_set, relevant_movie_ids in eval_cases:
        content_scores = similarity[seed_idx]
        cf_scores = cosine_similarity(cf_embeddings[seed_idx:seed_idx + 1], cf_embeddings)[0]
        cf_scores = (cf_scores + 1) / 2
        precomputed.append((content_scores, cf_scores, seen_idx_set, relevant_movie_ids))
    return precomputed


def evaluate_weights(alpha, beta, gamma, precomputed_cases, popularity, idx_to_movieid):
    precisions, recalls, ndcgs = [], [], []

    for content_scores, cf_scores, seen_idx_set, relevant_movie_ids in precomputed_cases:
        combined = alpha * content_scores + beta * cf_scores + gamma * popularity

        ranked = np.argsort(-combined)
        recommended_ids = []
        for i in ranked:
            if i in seen_idx_set:
                continue
            recommended_ids.append(idx_to_movieid[i])
            if len(recommended_ids) >= TOP_K:
                break

        p, r, n = precision_recall_ndcg_at_k(recommended_ids, relevant_movie_ids, TOP_K)
        precisions.append(p)
        recalls.append(r)
        ndcgs.append(n)

    return np.mean(precisions), np.mean(recalls), np.mean(ndcgs)


def main():
    similarity, data, cf_embeddings = load_artifacts()
    popularity = data['weighted_rating_norm'].to_numpy()

    movieid_to_idx = {mid: i for i, mid in enumerate(data['movieId'].tolist())}
    idx_to_movieid = {i: mid for mid, i in movieid_to_idx.items()}

    train_df, test_df = build_train_test_split(data)

    print("Building per-user evaluation cases (multiple seeds/user, "
          "so a user's niche-but-liked movies get a chance to be evaluated too)...")
    eval_cases = []
    test_by_user = test_df.groupby('userId')
    rng = np.random.RandomState(RANDOM_STATE)

    seed_rating_counts = []
    for uid, train_group in train_df.groupby('userId'):
        if uid not in test_by_user.groups:
            continue
        test_group = test_by_user.get_group(uid)

        relevant = test_group[test_group['rating'] >= RELEVANCE_THRESHOLD]['movieId'].tolist()
        if not relevant:
            continue

        # Use every movie the user liked in train as a candidate seed, not just
        # their single highest-rated one - otherwise the eval set is
        # structurally biased toward whatever movie happens to be each user's
        # #1 favorite, which is almost always something well-known, starving
        # the niche/long-tail bucket of any real sample size.
        liked_train = train_group[train_group['rating'] >= RELEVANCE_THRESHOLD]
        if liked_train.empty:
            liked_train = train_group.sort_values('rating', ascending=False).iloc[[0]]

        liked_train = liked_train[liked_train['movieId'].isin(movieid_to_idx)]
        if liked_train.empty:
            continue

        if len(liked_train) > MAX_SEEDS_PER_USER:
            liked_train = liked_train.sample(n=MAX_SEEDS_PER_USER, random_state=rng)

        seen_idx_set = {movieid_to_idx[m] for m in train_group['movieId'].tolist() if m in movieid_to_idx}

        for seed_movie_id in liked_train['movieId'].tolist():
            seed_idx = movieid_to_idx[seed_movie_id]
            eval_cases.append((seed_idx, seen_idx_set, set(relevant)))
            seed_rating_counts.append(data['rating_count'].iloc[seed_idx])

    print(f"Evaluation cases: {len(eval_cases)}")

    print("Precomputing per-case content/CF score vectors (reused across the whole grid)...")
    precomputed_cases = precompute_case_scores(eval_cases, similarity, cf_embeddings)

    weight_grid = []
    step = 0.2
    vals = [round(v, 2) for v in np.arange(0, 1 + 1e-9, step)]
    for a in vals:
        for b in vals:
            g = round(1 - a - b, 2)
            if -1e-9 <= g <= 1 + 1e-9:
                weight_grid.append((a, b, max(g, 0.0)))

    print(f"Grid-searching {len(weight_grid)} (alpha, beta, gamma) combinations...\n")
    results = []
    for alpha, beta, gamma in weight_grid:
        p, r, n = evaluate_weights(alpha, beta, gamma, precomputed_cases, popularity, idx_to_movieid)
        results.append((alpha, beta, gamma, p, r, n))
        print(f"alpha={alpha:.1f} beta={beta:.1f} gamma={gamma:.1f} "
              f"| Precision@{TOP_K}={p:.4f} Recall@{TOP_K}={r:.4f} NDCG@{TOP_K}={n:.4f}")

    results.sort(key=lambda x: x[5], reverse=True)  # sort by NDCG@K
    print("\nTop 5 weight combinations by NDCG@K:")
    for alpha, beta, gamma, p, r, n in results[:5]:
        print(f"  alpha={alpha:.1f} beta={beta:.1f} gamma={gamma:.1f} "
              f"-> Precision={p:.4f} Recall={r:.4f} NDCG={n:.4f}")

    best = results[0]
    print(f"\nBest by NDCG@{TOP_K}: alpha={best[0]}, beta={best[1]}, gamma={best[2]}")

    # --- Stratified re-evaluation: popular vs. long-tail seed movies ---
    # A single blended score can hide whether the hybrid actually helps niche
    # movies, or whether it's just riding popularity bias on blockbusters.
    # Split eval cases by the seed movie's rating_count and grid-search each
    # bucket separately.
    seed_rating_counts = np.array(seed_rating_counts)
    q80 = np.quantile(data['rating_count'], 0.80)
    q50 = np.quantile(data['rating_count'], 0.50)
    print(f"\nStratification thresholds (over all {len(data)} modeled movies): "
          f"popular >= {q80:.0f} ratings (top 20%), niche <= {q50:.0f} ratings (bottom 50%)")

    popular_cases = [c for c, rc in zip(precomputed_cases, seed_rating_counts) if rc >= q80]
    niche_cases = [c for c, rc in zip(precomputed_cases, seed_rating_counts) if rc <= q50]
    print(f"Popular-seed eval cases: {len(popular_cases)} | Niche-seed eval cases: {len(niche_cases)}")

    for label, cases in [("POPULAR seeds", popular_cases), ("NICHE seeds", niche_cases)]:
        print(f"\n--- Grid search restricted to {label} ({len(cases)} cases) ---")
        strat_results = []
        for alpha, beta, gamma in weight_grid:
            p, r, n = evaluate_weights(alpha, beta, gamma, cases, popularity, idx_to_movieid)
            strat_results.append((alpha, beta, gamma, p, r, n))
        strat_results.sort(key=lambda x: x[5], reverse=True)
        print(f"Top 5 for {label} by NDCG@{TOP_K}:")
        for alpha, beta, gamma, p, r, n in strat_results[:5]:
            print(f"  alpha={alpha:.1f} beta={beta:.1f} gamma={gamma:.1f} "
                  f"-> Precision={p:.4f} Recall={r:.4f} NDCG={n:.4f}")

    with open(os.path.join(MODEL_DIR, 'eval_results.pkl'), 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    main()
