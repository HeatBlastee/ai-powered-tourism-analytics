from pydantic import BaseModel
from typing import List, Optional

class RecommendationRequest(BaseModel):
    place_name: str
    top_n: Optional[int] = 5

class PlaceResponse(BaseModel):
    Place_Id: int
    Place_Name: str
    Category: str
    City: str
    Price: int
    Rating: float
    Similarity: Optional[float] = None

class RecommendationResponse(BaseModel):
    recommendations: List[PlaceResponse]

class ClassificationResponse(BaseModel):
    predicted_class: str
    confidence: float
