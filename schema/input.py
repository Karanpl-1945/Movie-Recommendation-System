from pydantic import BaseModel,Field
from typing import Annotated,Optional


class input_movie(BaseModel):
    movie: Annotated[str,Field(...,description='Enter the movie for which you want recommendation')]
    no_recommendation:Annotated[int,Field(description='number of  movie recommendation you want',default=5,lt=20,gt=0,)]
    

 