"""
birdwatch_ai package

A production-ready bird species classification system using MobileNetV2.
Supports 25 bird species with fine-tuned deep learning model.
"""

__version__ = "1.0.0"
__author__ = "BirdWatch AI Team a.k.a. Sam Bonte"

# Main modules
from birdwatch_ai import config
from birdwatch_ai import utils
from birdwatch_ai import data
from birdwatch_ai import model
from birdwatch_ai import train
from birdwatch_ai import evaluate
from birdwatch_ai import predict
from birdwatch_ai import run

# Key functions for external use
from birdwatch_ai.data import prepare_dataset
from birdwatch_ai.train import train_model
from birdwatch_ai.evaluate import evaluate_model
from birdwatch_ai.predict import predict_image
from birdwatch_ai.run import run_full_pipeline

__all__ = [
    'config',
    'utils',
    'data',
    'model',
    'train',
    'evaluate',
    'predict',
    'run',
    'prepare_dataset',
    'train_model',
    'evaluate_model',
    'predict_image',
    'run_full_pipeline',
]