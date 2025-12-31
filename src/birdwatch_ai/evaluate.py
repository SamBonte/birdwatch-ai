"""
evaluate.py (formerly model_metrics.py)

Responsibilities:
- Load trained model and history
- Evaluate model on test set
- Generate training curves
- Create classification report
- Plot confusion matrix
- Save metrics summary
"""

import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix

from birdwatch_ai.config import MODEL_FILE, HISTORY_FILE, METRICS_FILE
from birdwatch_ai.model import create_data_generators, get_class_names_from_generator
from birdwatch_ai.utils import load_trained_model, configure_gpu

# -------------------------
# Setup
# -------------------------
logger = logging.getLogger(__name__)
configure_gpu()

# -------------------------
# Load History
# -------------------------
def load_training_history(path=None):
    """
    Load training history from JSON file.
    
    Args:
        path: Path to history file. Defaults to config.HISTORY_FILE
    
    Returns:
        Dictionary containing training history
    """
    if path is None:
        path = HISTORY_FILE
    
    with open(path, "r") as f:
        history = json.load(f)
    logger.info(f"Training history loaded from {path}")
    return history

# -------------------------
# Plot Training Curves
# -------------------------
def plot_training_curves(history):
    """
    Plot training and validation accuracy/loss curves.
    
    Args:
        history: Training history dictionary
    """
    epochs = history["epoch"]
    
    plt.figure(figsize=(14, 6))
    
    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["accuracy"], label="Train Accuracy", marker='o')
    plt.plot(epochs, history["val_accuracy"], label="Val Accuracy", marker='s')
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["loss"], label="Train Loss", marker='o')
    plt.plot(epochs, history["val_loss"], label="Val Loss", marker='s')
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# -------------------------
# Plot Confusion Matrix
# -------------------------
def plot_confusion_matrix(cm, class_names):
    """
    Plot confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
    """
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - Bird Species Classification")
    plt.tight_layout()
    plt.show()

# -------------------------
# Evaluate Model
# -------------------------
def evaluate_model():
    """
    Comprehensive model evaluation pipeline.
    
    Returns:
        Dictionary containing all evaluation metrics
    """
    logger.info("Loading model and data...")
    model = load_trained_model(MODEL_FILE)
    history = load_training_history(HISTORY_FILE)
    
    # Get test data
    train_gen, val_gen, test_gen = create_data_generators()
    class_names = get_class_names_from_generator(train_gen)
    
    # Plot training curves
    logger.info("Plotting training curves...")
    plot_training_curves(history)
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_loss, test_acc, test_precision, test_recall = model.evaluate(test_gen, verbose=1)
    
    # Generate predictions
    logger.info("Generating predictions for confusion matrix...")
    y_true = test_gen.classes
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate metrics
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    macro_precision = precision_score(y_true, y_pred, average="macro")
    macro_recall = recall_score(y_true, y_pred, average="macro")
    
    # Classification report
    cls_report = classification_report(y_true, y_pred, target_names=class_names)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    logger.info("Plotting confusion matrix...")
    plot_confusion_matrix(cm, class_names)
    
    # Save metrics summary
    logger.info(f"Saving metrics summary to {METRICS_FILE}")
    with open(METRICS_FILE, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("MODEL EVALUATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test Loss:           {test_loss:.4f}\n")
        f.write(f"Test Accuracy:       {test_acc:.4f}\n")
        f.write(f"Precision (macro):   {macro_precision:.4f}\n")
        f.write(f"Recall (macro):      {macro_recall:.4f}\n")
        f.write(f"F1-score (macro):    {macro_f1:.4f}\n\n")
        f.write("=" * 60 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(cls_report + "\n")
    
    logger.info("Evaluation complete!")
    
    # Return metrics dictionary
    return {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'precision': macro_precision,
        'recall': macro_recall,
        'f1_score': macro_f1,
        'classification_report': cls_report
    }

# -------------------------
# CLI Entry Point
# -------------------------
def main():
    """Command-line interface entry point."""
    evaluate_model()

if __name__ == "__main__":
    main()