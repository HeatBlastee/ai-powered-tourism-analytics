import logging
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras import layers
from typing import Tuple, List, Optional
from src.settings import settings

# Configure structured logging
logger = logging.getLogger(__name__)

class HeritageClassifier:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.augmentation_model = self._build_augmentation()
        self.model = self._build_model()
        logger.info(f"Initialized HeritageClassifier with {num_classes} classes.")
        
    def _build_augmentation(self) -> tf.keras.Sequential:
        """Builds data augmentation pipeline."""
        return tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
        ], name="augmentation")

    def _build_model(self) -> Model:
        """
        Builds the model using EfficientNetB0 backbone.
        """
        inputs = Input(shape=(settings.IMG_SIZE[0], settings.IMG_SIZE[1], 3))
        
        # Apply augmentation
        x = self.augmentation_model(inputs)
        
        base_model = EfficientNetB0(
            weights='imagenet', 
            include_top=False, 
            input_tensor=x
        )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(settings.DROPOUT_RATE)(x)
        output = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=output)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=settings.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
    
    def unfreeze(self, learning_rate: float = 1e-5) -> None:
        """
        Unfreezes the top layers of the backbone for fine-tuning.
        """
        try:
            base_model = self.model.get_layer("efficientnetb0")
            base_model.trainable = True
            
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            )
            logger.info(f"Model unfrozen for fine-tuning with LR={learning_rate}")
        except ValueError as e:
            logger.error(f"Could not find efficientnetb0 layer to unfreeze: {e}")
            raise
        
    def train(self, 
              train_ds: tf.data.Dataset, 
              val_ds: tf.data.Dataset, 
              epochs: int = 10, 
              callbacks: Optional[List[tf.keras.callbacks.Callback]] = None) -> tf.keras.callbacks.History:
        
        if callbacks is None:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
            ]
            
        logger.info(f"Starting training for {epochs} epochs...")
        return self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks
        )
    
    def save(self, path: str) -> None:
        self.model.save(path)
        logger.info(f"Model saved to {path}")
        
    def load(self, path: str) -> None:
        self.model = tf.keras.models.load_model(path)
        logger.info(f"Model loaded from {path}")
        
    def predict(self, image: tf.Tensor) -> tf.Tensor:
        return self.model.predict(image)
