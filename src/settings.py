import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Tuple

class Settings(BaseSettings):
    # Paths
    ROOT_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = ROOT_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    MODELS_DIR: Path = ROOT_DIR / "models"
    
    # Data sub-directories
    VISION_DATA_DIR: Path = RAW_DATA_DIR / "vision"
    RECOMMENDER_DATA_DIR: Path = RAW_DATA_DIR / "recommender"
    
    # Model Paths
    VISION_MODEL_PATH: Path = MODELS_DIR / "heritage_classifier.h5"
    RECOMMENDER_MODEL_PATH: Path = MODELS_DIR / "tourism_recommender.pkl"
    
    # Hyperparameters
    IMG_SIZE: Tuple[int, int] = (224, 224)
    BATCH_SIZE: int = 16
    EPOCHS: int = 20
    RANDOM_SEED: int = 42
    LEARNING_RATE: float = 0.001
    DROPOUT_RATE: float = 0.5
    
    # Deployment
    MODEL_SERVING_URL: str = "http://127.0.0.1:5001/invocations"
    
    class Config:
        env_file = ".env"

# Logic to create directories
def ensure_directories(settings: Settings):
    for d in [
        settings.DATA_DIR, 
        settings.RAW_DATA_DIR, 
        settings.PROCESSED_DATA_DIR, 
        settings.MODELS_DIR,
        settings.RAW_DATA_DIR / "vision",
        settings.RAW_DATA_DIR / "recommender"
    ]:
        d.mkdir(parents=True, exist_ok=True)

settings = Settings()
ensure_directories(settings)
