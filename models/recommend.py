
import os
import pickle

from sklearn.metrics.pairwise import cosine_similarity

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')

with open(os.path.join(MODEL_DIR, 'similarity.pkl'), 'rb') as f:
    similarity = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'movies_data.pkl'), 'rb') as f:
    data = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'cf_embeddings.pkl'), 'rb') as f:
    cf_embeddings = pickle.load(f)

_TITLES = data['title'].str.lower().str.strip()

# Hybrid blend weights: content-based, item-based collaborative filtering, popularity.
# Tuned via models/evaluate.py (temporal per-user train/test split, multiple
# liked-movie seeds per user, grid search against Precision@10/Recall@10/
# NDCG@10). A popularity/CF-stratified re-run (~7k eval cases, split into
# ~7,000 popular-seed and 46 niche-seed cases) found alpha=0 wins in *every*
# stratum, including niche - i.e. content weight isn't earning measured
# accuracy even for movies with few ratings, since "few" is still enough for
# item-CF to work. Alpha is kept small (not 0) purely as a hedge for genuine
# cold-start movies with *zero* ratings, which by definition can never appear
# in a ratings-history-based evaluation like this one - so its value is
# untestable here, not disproven.
ALPHA, BETA, GAMMA = 0.1, 0.4, 0.5


def movie_recommend(movie_title: str, top_n: int):

    movie_title = movie_title.lower().strip()

    if movie_title not in _TITLES.values:
        return {
            "query_movie": movie_title,
            "recommendations": [],
            "message": "Movie not found"
        }

    idx = _TITLES[_TITLES == movie_title].index[0]

    content_scores = similarity[idx]
    cf_scores = cosine_similarity(cf_embeddings[idx:idx + 1], cf_embeddings)[0]
    cf_scores = (cf_scores + 1) / 2  # centered-rating cosine can be negative; rescale to [0,1]
    popularity_scores = data['weighted_rating_norm'].to_numpy()

    combined = ALPHA * content_scores + BETA * cf_scores + GAMMA * popularity_scores

    ranked = [(i, s) for i, s in enumerate(combined) if i != idx]
    ranked.sort(key=lambda x: x[1], reverse=True)
    top_ranked = ranked[:top_n]
    movie_indices = [i for i, _ in top_ranked]

    recommended = data.iloc[movie_indices][['movieId', 'title', 'average_rating', 'rating_count']].copy()
    recommended['confidence'] = [score for _, score in top_ranked]

    return {
        "query_movie": movie_title,
        "recommendations": recommended.to_dict(orient="records")
    }

