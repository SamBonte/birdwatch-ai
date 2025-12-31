"""
run.py (formerly mobilenetv2_finetuned_model.py)

Responsibilities:
- High-level orchestration of the entire pipeline
- Ensure dataset exists (download if needed)
- Ensure model is trained (train if needed)
- Provide convenient prediction interface
- Example usage and sanity checks
"""

import os
import logging

from birdwatch_ai.config import FILTERED_DATA_DIR, MODEL_FILE
from birdwatch_ai.data import prepare_dataset
from birdwatch_ai.train import train_model
from birdwatch_ai.evaluate import evaluate_model
from birdwatch_ai.predict import predict_image, display_supported_classes
from birdwatch_ai.utils import dir_exists

# -------------------------
# Setup Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -------------------------
# Ensure Dataset
# -------------------------
def ensure_dataset():
    """
    Ensure dataset is available. Download and prepare if needed.
    Idempotent - safe to call multiple times.
    """
    if dir_exists(FILTERED_DATA_DIR):
        logger.info("Dataset already exists.")
        return True
    
    logger.info("Dataset not found. Preparing dataset...")
    prepare_dataset()
    logger.info("Dataset preparation complete.")
    return True

# -------------------------
# Ensure Model
# -------------------------
def ensure_model():
    """
    Ensure model is trained. Train if needed.
    Idempotent - safe to call multiple times.
    """
    from birdwatch_ai.config import BEST_MODEL_FILE
    
    # Check if model exists and is a file (not a directory)
    if os.path.exists(MODEL_FILE) and os.path.isfile(MODEL_FILE):
        logger.info("Trained model already exists.")
        return True
    
    # Also check best model as fallback
    if os.path.exists(BEST_MODEL_FILE) and os.path.isfile(BEST_MODEL_FILE):
        logger.info("Best model found (final model missing, but can use best model).")
        return True
    
    logger.info("Model not found. Training model...")
    train_model()
    logger.info("Model training complete.")
    
    logger.info("Running model evaluation...")
    evaluate_model()
    
    return True

# -------------------------
# Full Pipeline
# -------------------------
def run_full_pipeline():
    """
    Run the complete pipeline from dataset preparation to evaluation.
    """
    logger.info("="*60)
    logger.info("Starting Full Bird Classification Pipeline")
    logger.info("="*60)
    
    # Step 1: Dataset
    logger.info("Step 1: Dataset Preparation")
    ensure_dataset()
    
    # Step 2: Training
    logger.info("Step 2: Model Training")
    ensure_model()
    
    logger.info("="*60)
    logger.info("Pipeline Complete!")
    logger.info("="*60)

# -------------------------
# Predict with Pipeline Check
# -------------------------
def predict_with_checks(img_path, return_all_probs=False):
    """
    Make prediction with automatic dataset and model checks.
    
    Args:
        img_path: Path to image file
        return_all_probs: If True, return all class probabilities
    
    Returns:
        Prediction result (format depends on return_all_probs)
    """
    ensure_dataset()
    ensure_model()
    
    result = predict_image(img_path, return_all_probs=return_all_probs)
    
    if not return_all_probs:
        predicted_class, confidence = result
        logger.info(f"Prediction: {predicted_class} ({confidence:.2%} confidence)")
    
    return result

# -------------------------
# CLI Entry Point
# -------------------------
def main():
    """
    Command-line interface entry point.
    Provides menu-driven interface for different operations.
    """
    import sys
    
    if len(sys.argv) < 2:
        print("\nBird Species Classification - Main Runner")
        print("="*60)
        print("\nUsage:")
        print("  python run.py pipeline         - Run full pipeline")
        print("  python run.py prepare          - Prepare dataset only")
        print("  python run.py train            - Train model only")
        print("  python run.py evaluate         - Evaluate model only")
        print("  python run.py predict <image>  - Predict image")
        print("  python run.py classes          - Show supported classes")
        print("\nExample:")
        print("  python run.py predict tests/inferenceTest.jpg")
        print("="*60)
        sys.exit(0)
    
    command = sys.argv[1].lower()
    
    if command == "pipeline":
        run_full_pipeline()
    
    elif command == "prepare":
        logger.info("Preparing dataset...")
        prepare_dataset()
    
    elif command == "train":
        logger.info("Training model...")
        ensure_dataset()
        train_model()
    
    elif command == "evaluate":
        logger.info("Evaluating model...")
        evaluate_model()
    
    elif command == "predict":
        if len(sys.argv) < 3:
            logger.error("Please provide image path: python run.py predict <image_path>")
            sys.exit(1)
        img_path = sys.argv[2]
        predict_with_checks(img_path)
    
    elif command == "classes":
        display_supported_classes()
    
    else:
        logger.error(f"Unknown command: {command}")
        logger.info("Run 'python run.py' without arguments to see usage")
        sys.exit(1)

if __name__ == "__main__":
    main()