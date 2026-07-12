from fastapi import FastAPI, HTTPException

from schema.input import input_movie
from schema.response import ResponseModel
from models.recommend import movie_recommend


app = FastAPI(
    title="Movie Recommendation API",
    description="Content-based movie recommendation system",
    version="1.0"
)


@app.get('/')
def home():
    return {'message':'this the movie recommendation system'}

@app.get('/health_check')
def health_check():
  return {
    'status':'ok',
    #'model_loaded':model is not None

  }
   

@app.post("/recommend", response_model=ResponseModel)
def recommend_movies(movie_request: input_movie):

    movie_title = movie_request.movie

    # number of recommendations
    top_n = movie_request.no_recommendation or 5

    result = movie_recommend(movie_title, top_n)

    if result.get("message") == "Movie not found":
        raise HTTPException(status_code=404, detail="Movie not found")

    return result
      