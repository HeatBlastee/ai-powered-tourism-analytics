import mlflow
import tensorflow as tf
import logging
from src.data.vision_loader import VisionDataLoader
from src.models.heritage_classifier import HeritageClassifier
from src.settings import settings
import json
import os

# Configure Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_training_pipeline() -> None:
    """
    Executes the end-to-end vision model training pipeline.
    """
    logger.info("Starting Vision Model Training...")
    
    mlflow.set_experiment("Heritage Classification")
    
    # Enable Auto-Logging
    mlflow.tensorflow.autolog(log_models=False)
    
    with mlflow.start_run() as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")
        
        # Load Data
        loader = VisionDataLoader()
        train_ds, val_ds = loader.load_data()
        class_names = loader.get_class_names()
        
        if not class_names:
            logger.error("No classes found. Aborting training.")
            raise ValueError("No classes found in dataset.")
            
        mlflow.log_param("custom_num_classes", len(class_names))
        logger.info(f"Classes found ({len(class_names)}): {class_names}")
        
        # Build Model
        classifier = HeritageClassifier(num_classes=len(class_names))
        
        # Train
        classifier.train(
            train_ds, 
            val_ds, 
            epochs=settings.EPOCHS,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
            ]
        )
        
        # Save Model to disk first (so we can log it as artifact)
        classifier.save(str(settings.VISION_MODEL_PATH))
        
        # Save Metadata
        class_indices = {name: i for i, name in enumerate(class_names)}
        metadata_path = settings.VISION_MODEL_PATH.parent / "class_indices.json"
        with open(metadata_path, 'w') as f:
            json.dump(class_indices, f)

        # Log Custom PyFunc Model
        from src.models.model_wrapper import HeritageModelWrapper
        from mlflow.models import infer_signature
        
        # Define Artifacts mapping
        artifacts = {
            "keras_model": str(settings.VISION_MODEL_PATH),
            "class_indices": str(metadata_path)
        }
        
        # Create Input Example & Signature
        try:
            sample_batch = next(iter(train_ds.take(1)))
            sample_input_raw = sample_batch[0].numpy()
            
            # Predict outcome for signature
            # Note: We need to use the Wrapper's logic to get the right output schema
            wrapped_model = HeritageModelWrapper()
            class ContextStub:
                artifacts = artifacts
            wrapped_model.load_context(ContextStub())
            sample_output = wrapped_model.predict(None, sample_input_raw)
            
            signature = infer_signature(sample_input_raw, sample_output)
            input_example = sample_input_raw[:1] # Just one image for example
            
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=HeritageModelWrapper(),
                code_path=["src/models/model_wrapper.py"], # log source code
                artifacts=artifacts,
                registered_model_name="HeritageClassifier_PyFunc",
                signature=signature,
                input_example=input_example
            )
            logger.info("PyFunc Model logged successfully with signature and input_example.")
            
        except Exception as e:
            logger.error(f"Failed to log PyFunc model: {e}")
            raise

    logger.info("Training pipeline finished successfully.")

if __name__ == "__main__":
    run_training_pipeline()
