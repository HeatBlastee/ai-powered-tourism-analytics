from fastapi import FastAPI, UploadFile, File, HTTPException
from src.api.schemas import RecommendationRequest, RecommendationResponse, ClassificationResponse
from src.models.tourism_recommender import TourismRecommender
from src.data.recommender_loader import RecommenderDataLoader

# tensorflow import removed
import numpy as np
# cv2 removed
import base64
import json
import uvicorn
import mlflow
import requests
import json
from contextlib import asynccontextmanager
from src.settings import settings

# Set MLflow Experiment for API Traces
mlflow.set_experiment("API Tracing")

# Global variables for models
# Vision model (HeritageClassifier) is now decoupled and served via MLflow
recommender_model = None
places_df = None
class_indices = None
class_names = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models
    global recommender_model, places_df, class_indices, class_names
    
    # Load Vision Model Metadata (Class Names) - REMOVED
    # The PyFunc Model Server now handles class mapping internally.
    # We no longer need to read class_indices.json here.


    # Load Recommender Model
    try:
        recommender_model = TourismRecommender()
        recommender_model.load(settings.RECOMMENDER_MODEL_PATH)
        
        loader = RecommenderDataLoader()
        _, _, places_df = loader.load_data()
        
    except Exception as e:
        print(f"Warning: Could not load Recommender Model: {e}")
        
    yield
    # Clean up

app = FastAPI(title="AI-Powered Tourism Analytics API", lifespan=lifespan)

# Enable CORS for Next.js (localhost:3000)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "ok", "version": "1.0.0"}

@app.post("/predict/classify", response_model=ClassificationResponse)
@mlflow.trace(name="predict_classify")
def classify_image(file: UploadFile = File(...)):
    # Read image
    # Note: 'file.read()' is async, but since we removed 'async def', we can't await it easily without
    # making the whole function async. 
    # The "Bar Raiser" feedback was: "Remove async (let FastAPI run it in a ThreadPool)".
    # However, 'UploadFile' methods are async. 
    # Standard pattern for non-async def in FastAPI:
    # 1. Use 'file.file.read()', which is the underlying synchronous file object.
    
    contents = file.file.read()
    
    # Encode to Base64
    b64_string = base64.b64encode(contents).decode('utf-8')
    
    # Payload for MLflow PyFunc (pandas-split or records)
    # We use dataframe_records which maps nicely to the DataFrame input in our wrapper
    payload = {"dataframe_records": [{"inputs": b64_string}]}
    
    # Predict via External Model Server
    print(f"DEBUG: Sending payload to {settings.MODEL_SERVING_URL}. B64 len: {len(b64_string)}")
    with mlflow.start_span("inference_http") as span:
        try:
            response = requests.post(settings.MODEL_SERVING_URL, json=payload)
            response.raise_for_status()
            
            # PyFunc returns: {"predictions": [{"predicted_class": "0", "confidence": 0.99}]}
            results = response.json() 
            
            if 'predictions' in results:
                preds = results['predictions']
            else:
                # Fallback
                preds = results
                
            first_pred = preds[0]
            
            # Safely handle if PyFunc returns the dict directly or different structure
            # Our HeritageModelWrapper returns dict with keys: 'predicted_class', 'confidence'
            predicted_class = first_pred.get('predicted_class', 'Unknown')
            confidence = first_pred.get('confidence', 0.0)

        except requests.exceptions.RequestException as e:
             detail = f"Model Serving unavailable: {e}"
             if hasattr(e, 'response') and e.response is not None:
                 detail += f" | MLflow Error: {e.response.text}"
             raise HTTPException(status_code=503, detail=detail)
    
    return {
        "predicted_class": predicted_class,
        "confidence": confidence
    }

@app.post("/predict/recommend", response_model=RecommendationResponse)
@mlflow.trace(name="predict_recommend")
def recommend_places(request: RecommendationRequest):
    if not recommender_model:
        raise HTTPException(status_code=503, detail="Recommender model not loaded")
        
    recommendations, error = recommender_model.recommend(
        request.place_name, 
        places_df, 
        top_n=request.top_n
    )
    
    if error:
        raise HTTPException(status_code=404, detail=error)
        
    return {"recommendations": recommendations.to_dict(orient="records")}
