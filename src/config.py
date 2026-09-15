import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Sub-directories for specific tasks
VISION_DATA_DIR = RAW_DATA_DIR / "vision"
RECOMMENDER_DATA_DIR = RAW_DATA_DIR / "recommender"

# Models directory
MODELS_DIR = ROOT_DIR / "models"
VISION_MODEL_PATH = MODELS_DIR / "heritage_classifier.h5"
RECOMMENDER_MODEL_PATH = MODELS_DIR / "tourism_recommender.pkl"

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
RANDOM_SEED = 42
MODEL_SERVING_URL = "http://127.0.0.1:5001/invocations"
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.5

# Create directories if they don't exist
for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, VISION_DATA_DIR, RECOMMENDER_DATA_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
