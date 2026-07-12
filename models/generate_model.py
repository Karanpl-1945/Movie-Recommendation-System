"""
Standalone script version of model.ipynb.

Reproduces the notebook's EDA + preprocessing + model-building pipeline
against the MovieLens 32M dataset (task4/ml-32m/) and writes the deployed
artifacts (similarity.pkl, movies_data.pkl) to task4/model/.
"""

import os
import re
import pickle

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack, csr_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'ml-32m')
MODEL_DIR = os.path.join(BASE_DIR, 'model')

CF_COMPONENTS = 50  # latent dimensions for the item-based CF embeddings


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def build_item_cf_embeddings(rating_data, movie_ids, n_components=CF_COMPONENTS):
    """Item-based collaborative filtering signal.

    Builds a (movies x users) sparse matrix of *adjusted* ratings (each
    user's ratings centered on their own mean, so a user who rates
    everything 5 stars doesn't skew similarity), then reduces it with
    TruncatedSVD to a dense (movies x n_components) embedding.

    Storing embeddings instead of the full movie-movie similarity matrix
    keeps this ~15MB instead of ~2.8GB (same size class as similarity.pkl);
    the movie-movie similarity is computed on the fly per-request in
    recommend.py from these embeddings.
    """
    movie_id_set = set(movie_ids)
    r = rating_data[rating_data['movieId'].isin(movie_id_set)][['userId', 'movieId', 'rating']].copy()

    user_means = r.groupby('userId')['rating'].transform('mean')
    r['rating_centered'] = r['rating'] - user_means

    movie_id_to_idx = {mid: i for i, mid in enumerate(movie_ids)}
    unique_users = r['userId'].unique()
    user_id_to_idx = {uid: i for i, uid in enumerate(unique_users)}

    rows = r['movieId'].map(movie_id_to_idx).to_numpy()
    cols = r['userId'].map(user_id_to_idx).to_numpy()
    vals = r['rating_centered'].to_numpy()

    matrix = csr_matrix((vals, (rows, cols)), shape=(len(movie_ids), len(unique_users)))

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    embeddings = svd.fit_transform(matrix)
    print("CF matrix shape:", matrix.shape, "| embeddings shape:", embeddings.shape)
    print("explained variance ratio (sum):", svd.explained_variance_ratio_.sum())
    return embeddings


def weighted_rating(df, m, C):
    """IMDB-style Bayesian-shrinkage weighted rating.

    Movies with few ratings get pulled toward the global mean C; only as
    the number of ratings v grows does the movie's own average R dominate.
    """
    v = df['rating_count']
    R = df['average_rating']
    return (v / (v + m)) * R + (m / (v + m)) * C


def eda(name, df):
    print(f"shape of {name} dataset is", df.shape)
    print("-----------------------------------------")
    print("unique movieid no.", df['movieId'].nunique())
    print("-----------------------------------------")
    print(df.isnull().sum())
    print("-----------------------------------------")
    print(df.describe())
    print("-----------------------------------------")
    print(df.head())


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    section("Loading datasets")
    movie_data = pd.read_csv(os.path.join(DATA_DIR, 'movies.csv'))
    rating_data = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'))
    tag_data = pd.read_csv(os.path.join(DATA_DIR, 'tags.csv'))

    section("EDA: movies")
    eda("MOVIE", movie_data)

    section("EDA: ratings")
    eda("RATING", rating_data)

    section("EDA: tags")
    eda("TAG", tag_data)

    section("Dropping null tags")
    tag_data.dropna(inplace=True, axis=0)
    print("shape of tag data", tag_data.shape)

    section("Grouping tags per movie")
    tags_grouped = tag_data.groupby('movieId')['tag'].agg(
        tags=lambda x: " ".join(x),
        tag_count='count'
    ).reset_index()

    section("Aggregating ratings per movie")
    movie_rating = rating_data.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
    movie_rating.columns = ['movieId', 'average_rating', 'rating_count']
    print(movie_rating.head())

    section("Merging movie_data, movie_rating and tags_grouped")
    merged_data = pd.merge(movie_data, movie_rating, on='movieId', how='inner')
    final_dataset = pd.merge(merged_data, tags_grouped, on='movieId', how='inner')
    print(final_dataset.head())

    section("Feature engineering: year, cleaned title, popularity_score")
    final_dataset['year'] = final_dataset['title'].str.extract(r'\((\d{4})\)')
    final_dataset['title'] = final_dataset['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()
    final_dataset['popularity_score'] = final_dataset['rating_count'] * np.log(final_dataset['average_rating'])
    print(final_dataset.shape)

    section("Filtering: keep movies with rating_count > 30")
    final_dataset = final_dataset[final_dataset['rating_count'] > 30].reset_index(drop=True)
    print(final_dataset.shape)

    data = final_dataset

    section("Weighted rating (Bayesian shrinkage popularity)")
    # m = the 90th percentile of rating_count: a movie needs at least this many
    # ratings before its own average is trusted over the global mean C.
    m = data['rating_count'].quantile(0.90)
    C = data['average_rating'].mean()
    print(f"m (min votes threshold) = {m}, C (global mean rating) = {C:.4f}")
    data['weighted_rating'] = data.apply(weighted_rating, axis=1, m=m, C=C)
    data['weighted_rating_norm'] = (
        (data['weighted_rating'] - data['weighted_rating'].min())
        / (data['weighted_rating'].max() - data['weighted_rating'].min())
    )
    print(data[['title', 'average_rating', 'rating_count', 'weighted_rating']].sort_values(
        'weighted_rating', ascending=False).head(10))

    section("NLTK stopwords + stemmer setup")
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    def preprocess_tags(text):
        text = str(text).lower()
        text = re.sub('[^a-zA-Z ]', ' ', text)
        words = text.split()
        words = [w for w in words if w not in stop_words]
        words = [stemmer.stem(w) for w in words]
        return " ".join(words)

    section("Cleaning tags")
    data['tags_clean'] = data['tags'].fillna("").apply(preprocess_tags)

    section("TF-IDF vectorization of tags")
    tfidf = TfidfVectorizer(max_features=5000)
    tag_matrix = tfidf.fit_transform(data['tags_clean'])
    print("tag_matrix shape:", tag_matrix.shape)

    section("One-hot encoding genres")
    data['genres_list'] = data['genres'].apply(lambda x: str(x).split('|'))
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(data['genres_list'])
    print("genre_matrix shape:", genre_matrix.shape)

    section("Combining features + computing cosine similarity (content-based)")
    feature_matrix = hstack([tag_matrix, genre_matrix])
    similarity = cosine_similarity(feature_matrix)
    print("similarity matrix shape:", similarity.shape)

    section("Building item-based collaborative filtering embeddings")
    cf_embeddings = build_item_cf_embeddings(rating_data, data['movieId'].tolist())

    # See models/recommend.py for how these were tuned (models/evaluate.py grid
    # search, including a popularity-stratified re-run) and why alpha is kept
    # small rather than 0 despite the metric favoring 0 in every stratum tested.
    ALPHA, BETA, GAMMA = 0.1, 0.4, 0.5  # content, item-CF, popularity weights

    def hybrid_recommend(movie_title, top_n=10):
        if movie_title not in data['title'].values:
            return {"query_movie": movie_title, "recommendations": [], "message": "Movie not found"}

        idx = data[data['title'] == movie_title].index[0]

        content_scores = similarity[idx]
        cf_scores = cosine_similarity(cf_embeddings[idx:idx + 1], cf_embeddings)[0]
        cf_scores = (cf_scores + 1) / 2  # cosine on centered vectors can be negative; rescale to [0,1]
        popularity_scores = data['weighted_rating_norm'].to_numpy()

        combined = ALPHA * content_scores + BETA * cf_scores + GAMMA * popularity_scores

        ranked = [(i, s) for i, s in enumerate(combined) if i != idx]
        ranked.sort(key=lambda x: x[1], reverse=True)
        movie_indices = [i for i, _ in ranked[:top_n]]

        recommended = data.iloc[movie_indices][['movieId', 'title', 'average_rating', 'rating_count']]

        return {"query_movie": movie_title, "recommendations": recommended.to_dict(orient="records")}

    section("Sanity check recommendations (hybrid: content + item-CF + popularity)")
    print(hybrid_recommend("Toy Story", 5))
    print(hybrid_recommend("Waiting to Exhale", 5))

    section("Saving model artifacts")
    with open(os.path.join(MODEL_DIR, 'similarity.pkl'), 'wb') as f:
        pickle.dump(similarity, f)
    with open(os.path.join(MODEL_DIR, 'movies_data.pkl'), 'wb') as f:
        pickle.dump(data, f)
    with open(os.path.join(MODEL_DIR, 'cf_embeddings.pkl'), 'wb') as f:
        pickle.dump(cf_embeddings, f)

    print(f"Saved similarity.pkl, movies_data.pkl and cf_embeddings.pkl to {MODEL_DIR}")


if __name__ == '__main__':
    main()
