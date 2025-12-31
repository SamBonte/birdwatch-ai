"""
predict.py (formerly inference.py)

Responsibilities:
- Load trained model
- Predict bird species from a single image
- Support both single prediction and full probability distribution
- Warn users about supported classes
"""

import logging
import numpy as np

from birdwatch_ai.config import MODEL_FILE, SELECTED_CLASSES
from birdwatch_ai.model import create_data_generators, get_class_names_from_generator
from birdwatch_ai.utils import load_trained_model, load_and_preprocess_image, configure_gpu

# -------------------------
# Setup
# -------------------------
logger = logging.getLogger(__name__)
configure_gpu()

# -------------------------
# Predict Image
# -------------------------
def predict_image(img_path, return_all_probs=False, model=None):
    """
    Predict bird species from an image.
    
    Args:
        img_path: Path to image file
        return_all_probs: If True, return probabilities for all classes
        model: Pre-loaded model (optional, will load if None)
    
    Returns:
        If return_all_probs=False: Tuple of (predicted_class, confidence)
        If return_all_probs=True: Dictionary of {class_name: probability}
    """
    # Load model if not provided
    # load_trained_model() will automatically fall back to BEST_MODEL_FILE if MODEL_FILE doesn't exist
    if model is None:
        model = load_trained_model()
    
    # Get class names
    train_gen, _, _ = create_data_generators()
    class_names = get_class_names_from_generator(train_gen)
    
    # Load and preprocess image
    logger.info(f"Processing image: {img_path}")
    img_array = load_and_preprocess_image(img_path)

    # Make prediction
    predictions = model.predict(img_array, verbose=0)[0]
    
    # Return results
    if return_all_probs:
        prob_dict = dict(zip(class_names, predictions.tolist()))
        return prob_dict
    else:
        max_idx = np.argmax(predictions)
        predicted_class = class_names[max_idx]
        confidence = float(predictions[max_idx])
        
        logger.info(f"Prediction: {predicted_class} (confidence: {confidence:.2%})")
        return predicted_class, confidence

# -------------------------
# Batch Prediction
# -------------------------
def predict_images(img_paths, model=None):
    """
    Predict bird species for multiple images.
    
    Args:
        img_paths: List of image file paths
        model: Pre-loaded model (optional, will load if None)
    
    Returns:
        List of tuples [(predicted_class, confidence), ...]
    """
    # Load model once for all predictions
    # load_trained_model() will automatically fall back to BEST_MODEL_FILE if MODEL_FILE doesn't exist
    if model is None:
        model = load_trained_model()
    
    results = []
    for img_path in img_paths:
        result = predict_image(img_path, return_all_probs=False, model=model)
        results.append(result)
    
    return results

# -------------------------
# Display Supported Classes
# -------------------------
def display_supported_classes():
    """Display the 25 supported bird species."""
    logger.info("Supported bird species (25 classes):")
    for i, cls in enumerate(SELECTED_CLASSES, 1):
        logger.info(f"  {i:2d}. {cls}")

# -------------------------
# CLI Entry Point
# -------------------------
def main():
    """
    Command-line interface entry point.
    Usage: python predict.py <image_path>
    """
    import sys
    
    if len(sys.argv) < 2:
        logger.error("Usage: python predict.py <image_path>")
        logger.info("\nNote: Only the following 25 bird species are supported:")
        display_supported_classes()
        sys.exit(1)
    
    img_path = sys.argv[1]
    
    logger.warning("Only the 25 selected bird species are supported for prediction.")
    logger.info("Loading model and making prediction...")
    
    predicted_class, confidence = predict_image(img_path)
    
    print("\n" + "="*60)
    print(f"Predicted Species: {predicted_class}")
    print(f"Confidence:        {confidence:.2%}")
    print("="*60)

if __name__ == "__main__":
    main()