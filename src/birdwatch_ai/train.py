"""
train.py (formerly train_model.py)

Responsibilities:
- Train MobileNetV2 model
- Save trained model
- Save training history for later evaluation
- Use components from model.py
"""

import json
import logging
import os

from birdwatch_ai.config import MODEL_FILE, HISTORY_FILE, EPOCHS
from birdwatch_ai.model import build_training_components
from birdwatch_ai.utils import configure_gpu, ensure_dir

# -------------------------
# Setup
# -------------------------
logger = logging.getLogger(__name__)
configure_gpu()

# -------------------------
# Training Function
# -------------------------
def train_model(epochs=None):
    """
    Train the bird classification model.
    
    Args:
        epochs: Number of training epochs. Defaults to config.EPOCHS
    
    Returns:
        Tuple of (model, history)
    """
    if epochs is None:
        epochs = EPOCHS
    
    logger.info("Building training components...")
    model, train_gen, val_gen, test_gen, callbacks, class_names = build_training_components()
    
    logger.info(f"Starting training for {epochs} epochs...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model_dir = os.path.dirname(MODEL_FILE)
    ensure_dir(model_dir)
    logger.info(f"Saving trained model to {MODEL_FILE}")
    model.save(MODEL_FILE)
    
    # Save training history
    history_dict = history.history
    history_dict['epoch'] = list(range(1, len(history.history['accuracy']) + 1))
    
    with open(HISTORY_FILE, "w") as f:
        json.dump(history_dict, f, indent=2)
    logger.info(f"Training history saved to {HISTORY_FILE}")
    
    # Log final metrics
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    logger.info(f"Training complete - Final train accuracy: {final_train_acc:.4f}, val accuracy: {final_val_acc:.4f}")
    
    return model, history

# -------------------------
# CLI Entry Point
# -------------------------
def main():
    """Command-line interface entry point."""
    train_model()

if __name__ == "__main__":
    main()