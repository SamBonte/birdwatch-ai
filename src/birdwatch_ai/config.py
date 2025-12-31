"""
config.py

Single source of truth for all configuration constants.
No imports from project modules, no logic - only constants.
"""

import os

# -------------------------
# Project Structure
# -------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# -------------------------
# Data Directories
# -------------------------
RAW_DATA_DIR = os.path.join(BASE_DATA_DIR, "bird-species-classification-220-categories")
RAW_DATASET_NAME = "Bird Species Classification 220 Dataset"
FILTERED_DATA_DIR = os.path.join(BASE_DATA_DIR, "birds_filtered_25")

# -------------------------
# Model Directories
# -------------------------
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "mobilenetv2_finetuned")
MODEL_FILE = os.path.join(MODEL_DIR, "mobilenetv2_final.keras")
BEST_MODEL_FILE = os.path.join(MODEL_DIR, "mobilenetv2_best_finetuned.keras")
HISTORY_FILE = os.path.join(MODEL_DIR, "training_history.json")
METRICS_FILE = os.path.join(MODEL_DIR, "metrics_summary.txt")

# -------------------------
# Model Hyperparameters
# -------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 25
VAL_SPLIT = 0.2
LEARNING_RATE = 1e-4
EPOCHS = 3  # Set to 2 for testing, default is 30
FINE_TUNE_LAYERS = 30  # Last N layers to fine-tune

# -------------------------
# Training Configuration
# -------------------------
RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-6

# -------------------------
# Data Augmentation
# -------------------------
ROTATION_RANGE = 20
WIDTH_SHIFT_RANGE = 0.2
HEIGHT_SHIFT_RANGE = 0.2
SHEAR_RANGE = 0.15
ZOOM_RANGE = 0.15
HORIZONTAL_FLIP = True
VERTICAL_FLIP = True
FILL_MODE = 'nearest'

# -------------------------
# Model Architecture
# -------------------------
DROPOUT_RATE = 0.5
BASE_MODEL_WEIGHTS = 'imagenet'

# -------------------------
# Selected Bird Classes (25)
# -------------------------
SELECTED_CLASSES = [
    "Mallard", "House_Sparrow", "Common_Raven", "Yellow_billed_Cuckoo",
    "Ring_billed_Gull", "Horned_Lark", "Bank_Swallow", "Cedar_Waxwing",
    "European_Goldfinch", "Gadwall", "Gray_Catbird", "Great_Grey_Shrike",
    "Herring_Gull", "Hooded_Merganser", "House_Wren", "Northern_Flicker",
    "Northern_Fulmar", "Painted_Bunting", "Pied_billed_Grebe",
    "Pileated_Woodpecker", "Purple_Finch", "Red_breasted_Merganser",
    "Red_winged_Blackbird", "Red_eyed_Vireo", "Summer_Tanager"
]

# -------------------------
# Kaggle Dataset
# -------------------------
KAGGLE_DATASET_ID = "kedarsai/bird-species-classification-220-categories"