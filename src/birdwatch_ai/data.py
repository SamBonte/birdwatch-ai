"""
data.py (formerly build_dataset.py)

Responsibilities:
- Download bird species dataset from Kaggle (idempotent)
- Automatically detect Train and Test folders
- Create filtered dataset with 25 selected bird classes
- Case-insensitive class matching for robustness
- Skip work if dataset already exists
"""

import os
import shutil
import logging
import kagglehub

from birdwatch_ai.config import (
    RAW_DATA_DIR, RAW_DATASET_NAME, FILTERED_DATA_DIR,
    SELECTED_CLASSES, KAGGLE_DATASET_ID
)
from birdwatch_ai.utils import ensure_dir, dir_exists, find_subdirs, normalize_class_name

# -------------------------
# Setup Logging
# -------------------------
logger = logging.getLogger(__name__)

# -------------------------
# Download Dataset
# -------------------------
def download_dataset():
    """
    Download dataset from Kaggle if not already present.
    Idempotent - skips download if data exists.
    
    Returns:
        Path to the raw dataset folder
    """
    final_folder = os.path.join(RAW_DATA_DIR, RAW_DATASET_NAME)
    
    # Check if dataset already exists
    if dir_exists(final_folder):
        logger.info(f"Dataset already exists at: {final_folder}")
        return final_folder
    
    logger.info("Downloading dataset from Kaggle...")
    ensure_dir(RAW_DATA_DIR)
    
    path = kagglehub.dataset_download(KAGGLE_DATASET_ID)
    logger.info(f"Dataset downloaded to: {path}")
    
    # Detect main folder inside downloaded path
    subfolders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    if len(subfolders) == 1:
        source_path = os.path.join(path, subfolders[0])
    else:
        source_path = path
    
    # Copy to final location
    if not os.path.exists(final_folder):
        shutil.copytree(source_path, final_folder)
    
    logger.info(f"Raw dataset prepared at: {final_folder}")
    return final_folder

# -------------------------
# Find Train and Test Folders
# -------------------------
def find_train_test_dirs(raw_dir):
    """
    Find Train and Test directories within raw dataset.
    
    Args:
        raw_dir: Path to raw dataset directory
    
    Returns:
        Tuple of (train_dir, test_dir)
    
    Raises:
        FileNotFoundError: If Train or Test folders not found
    """
    found_dirs = find_subdirs(raw_dir, ["Train", "Test"])
    
    if "Train" not in found_dirs or "Test" not in found_dirs:
        raise FileNotFoundError("Could not find Train and Test folders in dataset")
    
    train_dir = found_dirs["Train"]
    test_dir = found_dirs["Test"]
    
    logger.info(f"Found Train folder: {train_dir}")
    logger.info(f"Found Test folder: {test_dir}")
    
    return train_dir, test_dir

# -------------------------
# Create Filtered Dataset
# -------------------------
def create_filtered_dataset(train_dir, test_dir):
    """
    Create filtered dataset with only selected bird classes.
    Case-insensitive matching for robustness.
    
    Args:
        train_dir: Path to raw training data
        test_dir: Path to raw test data
    """
    train_filtered = os.path.join(FILTERED_DATA_DIR, "Train")
    test_filtered = os.path.join(FILTERED_DATA_DIR, "Test")
    
    ensure_dir(train_filtered)
    ensure_dir(test_filtered)
    
    # Create case-insensitive mapping of actual class folders
    actual_train_classes = {
        normalize_class_name(c): c 
        for c in os.listdir(train_dir) 
        if os.path.isdir(os.path.join(train_dir, c))
    }
    actual_test_classes = {
        normalize_class_name(c): c 
        for c in os.listdir(test_dir) 
        if os.path.isdir(os.path.join(test_dir, c))
    }
    
    # Copy selected classes
    for cls in SELECTED_CLASSES:
        cls_lower = normalize_class_name(cls)
        
        # Process Train folder
        if cls_lower in actual_train_classes:
            src_train = os.path.join(train_dir, actual_train_classes[cls_lower])
            dst_train = os.path.join(train_filtered, cls)
            if not os.path.exists(dst_train):
                shutil.copytree(src_train, dst_train)
        else:
            logger.warning(f"Train folder missing for {cls}")
        
        # Process Test folder
        if cls_lower in actual_test_classes:
            src_test = os.path.join(test_dir, actual_test_classes[cls_lower])
            dst_test = os.path.join(test_filtered, cls)
            if not os.path.exists(dst_test):
                shutil.copytree(src_test, dst_test)
        else:
            logger.warning(f"Test folder missing for {cls}")
    
    logger.info("Filtered dataset with 25 classes created successfully.")

# -------------------------
# Main Pipeline
# -------------------------
def prepare_dataset():
    """
    Main pipeline to prepare dataset.
    Idempotent - skips if filtered dataset already exists.
    
    Returns:
        Path to filtered dataset directory
    """
    # Check if filtered dataset already exists
    if dir_exists(FILTERED_DATA_DIR):
        logger.info(f"Filtered dataset already exists at: {FILTERED_DATA_DIR}")
        return FILTERED_DATA_DIR
    
    logger.info("Preparing dataset...")
    raw_folder = download_dataset()
    train_dir, test_dir = find_train_test_dirs(raw_folder)
    create_filtered_dataset(train_dir, test_dir)
    
    logger.info(f"Dataset preparation complete: {FILTERED_DATA_DIR}")
    return FILTERED_DATA_DIR

# -------------------------
# CLI Entry Point
# -------------------------
def main():
    """Command-line interface entry point."""
    prepare_dataset()

if __name__ == "__main__":
    main()