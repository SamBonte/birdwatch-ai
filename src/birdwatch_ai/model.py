"""
model.py (formerly build_model.py)

Responsibilities:
- Build MobileNetV2 model architecture
- Configure fine-tuning layers
- Add classifier head
- Compile model with optimizer and metrics
- Setup data augmentation and generators
- Create training callbacks
"""

import os
import logging
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from birdwatch_ai.config import (
    IMG_SIZE, BATCH_SIZE, NUM_CLASSES, VAL_SPLIT, LEARNING_RATE,
    DROPOUT_RATE, BASE_MODEL_WEIGHTS, FINE_TUNE_LAYERS,
    ROTATION_RANGE, WIDTH_SHIFT_RANGE, HEIGHT_SHIFT_RANGE,
    SHEAR_RANGE, ZOOM_RANGE, HORIZONTAL_FLIP, VERTICAL_FLIP,
    FILL_MODE, BEST_MODEL_FILE, EARLY_STOPPING_PATIENCE,
    REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR, MIN_LR, FILTERED_DATA_DIR
)
from birdwatch_ai.utils import configure_gpu, ensure_dir, get_train_test_paths

# -------------------------
# Setup
# -------------------------
logger = logging.getLogger(__name__)
configure_gpu()

# -------------------------
# Data Generators
# -------------------------
def create_data_generators():
    """
    Create training, validation, and test data generators.
    
    Returns:
        Tuple of (train_generator, val_generator, test_generator)
    """
    train_path, test_path = get_train_test_paths(FILTERED_DATA_DIR)
    
    # Training data with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=ROTATION_RANGE,
        width_shift_range=WIDTH_SHIFT_RANGE,
        height_shift_range=HEIGHT_SHIFT_RANGE,
        shear_range=SHEAR_RANGE,
        zoom_range=ZOOM_RANGE,
        horizontal_flip=HORIZONTAL_FLIP,
        vertical_flip=VERTICAL_FLIP,
        fill_mode=FILL_MODE,
        validation_split=VAL_SPLIT
    )
    
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        subset='training'
    )
    
    val_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        subset='validation'
    )
    
    # Test data without augmentation
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    logger.info(f"Data generators created: {len(train_generator.classes)} training samples")
    return train_generator, val_generator, test_generator

# -------------------------
# Build Model
# -------------------------
def build_mobilenet_model():
    """
    Build MobileNetV2 model with fine-tuning configuration.
    
    Returns:
        Compiled Keras model
    """
    # Load base model
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights=BASE_MODEL_WEIGHTS
    )
    
    # Freeze early layers, fine-tune last N layers
    for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False
    for layer in base_model.layers[-FINE_TUNE_LAYERS:]:
        layer.trainable = True
    
    # Add classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(DROPOUT_RATE)(x)
    output = Dense(NUM_CLASSES, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=base_model.input, outputs=output)
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    logger.info(f"Model built: {FINE_TUNE_LAYERS} layers trainable")
    return model

# -------------------------
# Training Callbacks
# -------------------------
def create_callbacks(model_save_path=None):
    """
    Create training callbacks for model checkpointing, early stopping, etc.
    
    Args:
        model_save_path: Path to save best model. Defaults to config.BEST_MODEL_FILE
    
    Returns:
        List of Keras callbacks
    """
    if model_save_path is None:
        model_save_path = BEST_MODEL_FILE
    
    ensure_dir(os.path.dirname(model_save_path))
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LR,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_save_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    logger.info("Training callbacks created")
    return callbacks

# -------------------------
# Get Class Names
# -------------------------
def get_class_names_from_generator(generator):
    """
    Extract class names from data generator.
    
    Args:
        generator: Keras ImageDataGenerator
    
    Returns:
        List of class names
    """
    return list(generator.class_indices.keys())

# -------------------------
# Main Builder Function
# -------------------------
def build_training_components():
    """
    Build all components needed for training.
    
    Returns:
        Tuple of (model, train_gen, val_gen, test_gen, callbacks, class_names)
    """
    train_gen, val_gen, test_gen = create_data_generators()
    model = build_mobilenet_model()
    callbacks = create_callbacks()
    class_names = get_class_names_from_generator(train_gen)
    
    return model, train_gen, val_gen, test_gen, callbacks, class_names

# -------------------------
# CLI Entry Point
# -------------------------
def main():
    """Command-line interface entry point - displays model summary."""
    model = build_mobilenet_model()
    model.summary()

if __name__ == "__main__":
    main()