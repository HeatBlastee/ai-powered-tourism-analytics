
import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
import tensorflow as tf
import numpy as np
import base64
import cv2
from pathlib import Path
import shutil
import pandas as pd
import json

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT_DIR / "models"
H5_PATH = MODELS_DIR / "heritage_classifier.h5"
CLASS_INDICES_PATH = MODELS_DIR / "class_indices.json"
ARTIFACT_PATH = MODELS_DIR / "heritage_classifier"

class HeritageModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        Load the model and class indices from the artifacts.
        """
        keras_model_path = context.artifacts["keras_model"]
        print(f"Loading Keras model from {keras_model_path}...")
        self.model = tf.keras.models.load_model(keras_model_path, compile=False)
        print("Model loaded successfully.")
        
        # Load class indices mapping
        self.class_labels = {}
        if "class_indices" in context.artifacts:
            class_indices_path = context.artifacts["class_indices"]
            try:
                with open(class_indices_path, "r") as f:
                    # JSON content: {"Label": Index, ...} -> We need {Index: "Label"}
                    indices_map = json.load(f)
                    # Invert the map: {0: "Taj Mahal", 1: "Eiffel Tower"}
                    self.class_labels = {v: k for k, v in indices_map.items()}
                print(f"Loaded {len(self.class_labels)} class labels.")
            except Exception as e:
                print(f"Warning: Failed to load class indices: {e}")

    def predict(self, context, model_input):
        """
        Predict method matching MLflow PyFunc signature.
        model_input: pandas.DataFrame with an 'inputs' column containing Base64 strings.
        """
        # Extract the base64 string from the input DataFrame
        # FastAPI sends: {"inputs": ["base64string"]} -> MLflow converts to DataFrame
        
        # Handle different input shapes/types just in case
        if isinstance(model_input, pd.DataFrame):
            # Take the first row's 'inputs' column
            input_b64 = model_input["inputs"].iloc[0] 
        elif isinstance(model_input, dict):
             input_b64 = model_input["inputs"][0]
        else:
            raise ValueError(f"Unsupported input type: {type(model_input)}")

        # Decode Base64 to Image
        try:
            image_data = base64.b64decode(input_b64)
            np_arr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            raise ValueError(f"Failed to decode base64 image: {e}")

        if img is None:
            raise ValueError("Decoded image is None. Check input base64 string.")

        # Preprocessing (Resize to 224x224, Convert BGR to RGB)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Expand dims to (1, 224, 224, 3)
        img_batch = np.expand_dims(img, axis=0)

        # Predict
        preds = self.model.predict(img_batch)
        
        # Post-processing
        # preds is shape (1, num_classes)
        predicted_idx = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))
        
        # Map index to label if available
        predicted_label = str(predicted_idx)
        if hasattr(self, 'class_labels') and predicted_idx in self.class_labels:
            predicted_label = self.class_labels[predicted_idx]

        # Return result
        # We return a list of dicts or a DataFrame to match PyFunc expectations
        return [{"predicted_class": predicted_label, "confidence": confidence}]

def convert_model():
    print(f"Preparing to wrap model from {H5_PATH}...")
    if not H5_PATH.exists():
        print(f"Error: {H5_PATH} does not exist.")
        return

    # Clean up existing artifact
    if ARTIFACT_PATH.exists():
        print(f"Removing existing artifact at {ARTIFACT_PATH}...")
        shutil.rmtree(ARTIFACT_PATH)

    # Define artifacts to bundle
    artifacts = {
        "keras_model": str(H5_PATH),
        "class_indices": str(CLASS_INDICES_PATH)
    }
    
    # Check if class indices exist
    if not CLASS_INDICES_PATH.exists():
        print(f"Warning: {CLASS_INDICES_PATH} not found. Prediction will return indices only.")
        del artifacts["class_indices"]

    # Save as PyFunc model
    print(f"Saving MLflow PyFunc model to {ARTIFACT_PATH}...")
    mlflow.pyfunc.save_model(
        path=str(ARTIFACT_PATH),
        python_model=HeritageModelWrapper(),
        artifacts=artifacts,
        # We can specify conda_env or pip_requirements to ensure the server has dependencies
        pip_requirements=[
            "tensorflow",
            "numpy",
            "opencv-python-headless", 
            "pandas"
        ]
    )
    print("Conversion complete!")

if __name__ == "__main__":
    convert_model()
