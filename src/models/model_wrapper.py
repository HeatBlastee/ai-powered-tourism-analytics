import mlflow
import tensorflow as tf
import json
import numpy as np
import cv2
from typing import List, Dict, Any

class HeritageModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        Loads the model and artifacts from the MLflow artifact location.
        """
        # Load Keras Model
        # output_graph_path refers to the path in the artifact dictionary
        self.model = tf.keras.models.load_model(context.artifacts["keras_model"])
        
        # Load Class Indices
        with open(context.artifacts["class_indices"], 'r') as f:
            self.class_indices = json.load(f)
            # Create reverse mapping: index -> class_name
            self.class_names = {v: k for k, v in self.class_indices.items()}
            
    def predict(self, context, model_input):
        """
        model_input: dict with key "inputs" containing list of base64 encoded strings
        Returns: List of dictionaries with class_name and confidence
        """
        import base64

        # Extract base64 strings
        if isinstance(model_input, dict) and "inputs" in model_input:
            inputs = model_input["inputs"]
        elif isinstance(model_input, (list, np.ndarray)):
            inputs = model_input
        else:
            inputs = model_input.values.flatten().tolist() if hasattr(model_input, 'values') else []
            
        processed_images = []
        
        for item in inputs:
            # Decode Base64
            # Handle potential Data URI scheme (e.g. data:image/jpeg;base64,...)
            if isinstance(item, str):
                if "," in item:
                    item = item.split(",")[1]
                img_bytes = base64.b64decode(item)
                nparr = np.frombuffer(img_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                # Assume it's already an array (backward compatibility or testing)
                image = np.array(item)

            if image is None:
                continue

            # Preprocessing (Resize & EfficientNet Preprocess)
            # HARDCODED SIZE (224, 224) to ensure model compatibility 
            # Ideally this comes from the artifact config, but efficientnet b0 is standard.
            image = cv2.resize(image, (224, 224))
            image = tf.keras.applications.efficientnet.preprocess_input(image)
            processed_images.append(image)
            
        if not processed_images:
            return []

        images_batch = np.array(processed_images)
        
        # Inference
        probs = self.model.predict(images_batch)
        
        # Postprocessing
        results = []
        for i, prob_vector in enumerate(probs):
            class_idx = np.argmax(prob_vector)
            confidence = float(prob_vector[class_idx])
            class_name = self.class_names.get(class_idx, "Unknown")
            
            results.append({
                "predictions": prob_vector.tolist(), # Raw probs for debugging
                "predicted_class": class_name,
                "confidence": confidence
            })
            
        return results
