from pydantic import BaseModel,Field
from typing import Annotated,List




class MovieRecommendation(BaseModel):
    movieId: Annotated[int, Field(..., description="Unique movie identifier")]
    title: Annotated[str , Field(..., description="Movie title")]
    average_rating: Annotated[float , Field(..., description="Average movie rating")]
    rating_count: Annotated[int ,Field(..., description="Number of ratings")]
    confidence: Annotated[float, Field(..., description="Hybrid recommendation score")]


class ResponseModel(BaseModel):
    query_movie: Annotated[str , Field(..., description="Movie used for recommendation")]
    recommendations: List[MovieRecommendation]