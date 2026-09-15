import tensorflow as tf
from pathlib import Path
import logging
from src.settings import settings
from src.data.mock_generator import MockDataGenerator

# Configure Logger
logger = logging.getLogger(__name__)

class VisionDataLoader:
    def __init__(self, data_dir: Path = settings.VISION_DATA_DIR):
        self.data_dir = data_dir
        
        # Check for user provided specific structure
        user_structure = self.data_dir / "dataset_hist_structures" / "Stuctures_Dataset"
        if user_structure.exists():
            logger.info(f"Found user dataset at {user_structure}")
            self.data_dir = user_structure
        
        # Check if data exists, if not generate mock
        # We check if the resolved data_dir is empty or key subdirs don't exist
        if not self.data_dir.exists() or not any(self.data_dir.iterdir()):
            logger.warning("Vision data directory empty. Generating mock data...")
            MockDataGenerator.generate_vision_data()
            # If we generated mock data, it went to VISION_DATA_DIR root
            self.data_dir = settings.VISION_DATA_DIR
            
    def load_data(self, validation_split=0.2):
        """
        Loads images using tf.keras.utils.image_dataset_from_directory
        Returns train_ds, val_ds
        """
        logger.info(f"Loading images from {self.data_dir}")
        
        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=validation_split,
            subset="training",
            seed=settings.RANDOM_SEED,
            image_size=settings.IMG_SIZE,
            batch_size=settings.BATCH_SIZE,
            label_mode='categorical'
        )
        
        val_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=validation_split,
            subset="validation",
            seed=settings.RANDOM_SEED,
            image_size=settings.IMG_SIZE,
            batch_size=settings.BATCH_SIZE,
            label_mode='categorical'
        )
        
        # Prefetch for performance
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        
        return train_ds, val_ds

    def get_class_names(self):
        # Helper to get class names from directory structure
        return sorted([item.name for item in self.data_dir.iterdir() if item.is_dir()])
