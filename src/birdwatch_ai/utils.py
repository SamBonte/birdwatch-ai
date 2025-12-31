"""
utils.py

Centralized utility functions for:
- Image loading and preprocessing
- Path operations
- Model loading
- Class mapping
- Directory management

No imports from train, evaluate, or predict modules.
"""

import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from birdwatch_ai.config import IMG_SIZE, MODEL_FILE

# -------------------------
# Setup Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -------------------------
# GPU Configuration
# -------------------------
def configure_gpu():
    """Configure GPU memory growth to avoid OOM errors."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPU enabled: {gpus}")
        except RuntimeError as e:
            logger.warning(f"GPU configuration error: {e}")
    else:
        logger.warning("No GPU detected, using CPU.")

# -------------------------
# Directory Management
# -------------------------
def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)
    return directory

def dir_exists(directory):
    """Check if directory exists and is not empty."""
    return os.path.exists(directory) and len(os.listdir(directory)) > 0

# -------------------------
# Image Operations
# -------------------------
def load_and_preprocess_image(img_path, target_size=None):
    """
    Load image from path and preprocess for model input.
    
    Args:
        img_path: Path to image file
        target_size: Tuple of (height, width). Defaults to config.IMG_SIZE
    
    Returns:
        Preprocessed image array ready for model prediction
    """
    if target_size is None:
        target_size = IMG_SIZE
    
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# -------------------------
# Model Operations
# -------------------------
def load_trained_model(model_path=None):
    """
    Load a trained Keras model from disk.
    
    Args:
        model_path: Path to model file. Defaults to config.MODEL_FILE
                   If MODEL_FILE doesn't exist, falls back to BEST_MODEL_FILE
    
    Returns:
        Loaded Keras model
    
    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    from birdwatch_ai.config import BEST_MODEL_FILE
    
    if model_path is None:
        model_path = MODEL_FILE
        # Fallback to best model if final model doesn't exist
        if not os.path.exists(model_path) or os.path.isdir(model_path):
            logger.info(f"Final model not found, using best model: {BEST_MODEL_FILE}")
            model_path = BEST_MODEL_FILE
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    logger.info(f"Loading model from {model_path}")
    return tf.keras.models.load_model(model_path)

# -------------------------
# Class Mapping
# -------------------------
def get_class_index_mapping(class_names):
    """
    Create bidirectional mapping between class names and indices.
    
    Args:
        class_names: List of class names
    
    Returns:
        Tuple of (name_to_idx, idx_to_name) dictionaries
    """
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idx_to_name = {idx: name for idx, name in enumerate(class_names)}
    return name_to_idx, idx_to_name

def normalize_class_name(class_name):
    """Normalize class name to lowercase for case-insensitive matching."""
    return class_name.lower()

# -------------------------
# Path Utilities
# -------------------------
def find_subdirs(root_dir, target_names):
    """
    Find subdirectories with specific names (case-insensitive).
    
    Args:
        root_dir: Root directory to search
        target_names: List of directory names to find
    
    Returns:
        Dictionary mapping target names to found paths
    """
    found = {}
    target_names_lower = [name.lower() for name in target_names]
    
    for root, dirs, _ in os.walk(root_dir):
        for d in dirs:
            if d.lower() in target_names_lower:
                idx = target_names_lower.index(d.lower())
                found[target_names[idx]] = os.path.join(root, d)
    
    return found

def get_train_test_paths(filtered_data_dir):
    """
    Get paths to Train and Test directories.
    
    Args:
        filtered_data_dir: Path to filtered dataset
    
    Returns:
        Tuple of (train_path, test_path)
    """
    train_path = os.path.join(filtered_data_dir, "Train")
    test_path = os.path.join(filtered_data_dir, "Test")
    return train_path, test_path

