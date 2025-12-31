"""
serve.py

FastAPI serving layer for BirdWatch AI.
Responsibilities:
- Initialize the model once at startup
- Handle HTTP requests for inference
- Validate inputs
- Delegate logic to predict.py
"""

import os
import logging
import shutil
import tempfile
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager

from birdwatch_ai.config import SELECTED_CLASSES
from birdwatch_ai.utils import load_trained_model
from birdwatch_ai.predict import predict_image

# -------------------------
# Setup Logging
# -------------------------
# Basic Logging config already done in run.py
logger = logging.getLogger(__name__)

# -------------------------
# Global State
# -------------------------
ML_MODELS = {}

# -------------------------
# Lifespan Management
# -------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle startup and shutdown logic.
    Loads the model into memory exactly once.
    """
    try:
        logger.info("Initializing API service...")
        # Load model using existing utility
        # This handles fallback logic (MODEL_FILE vs BEST_MODEL_FILE) automatically
        ML_MODELS["mobilenet"] = load_trained_model()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model during startup: {e}")
        raise e
    
    yield
    
    # Clean up resources if necessary
    ML_MODELS.clear()
    logger.info("Shutting down API service...")

  
# -------------------------
# FastAPI App
# -------------------------
app = FastAPI(
    title="BirdWatch AI API",
    description="Classify bird species using MobileNetV2",
    version="1.0.0",
    lifespan=lifespan
)

# -------------------------
# Endpoints
# -------------------------
@app.get("/health")
async def health_check():
    """Health check endpoint to ensure service and model are ready."""
    if "mobilenet" not in ML_MODELS:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model_loaded": True}

@app.get("/classes")
async def get_classes():
    """Return the list of supported bird species."""
    # Source specific classes from config
    return {
        "count": len(SELECTED_CLASSES),
        "classes": SELECTED_CLASSES
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict bird species from an uploaded image.
    
    Process:
    1. Validate file type
    2. Save to temp file (to reuse existing path-based preprocessing)
    3. Run inference using pre-loaded model
    4. Cleanup temp file
    """
    # 1. Validate file type
    if file.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Only JPEG and PNG are supported."
        )

    temp_filename = None
    
    try:
        # 2. Save to temp file
        # We must use a temp file because utils.load_and_preprocess_image 
        # expects a file path, not bytes.
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_filename = tmp.name

        # 3. Inference
        # We pass the pre-loaded model to prevent reloading it per request
        predicted_class, confidence = predict_image(
            img_path=temp_filename, 
            return_all_probs=False, 
            model=ML_MODELS["mobilenet"]
        )

        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "filename": file.filename
        }

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal processing error")
        
    finally:
        # 4. Cleanup
        if temp_filename and os.path.exists(temp_filename):
            os.remove(temp_filename)

# -------------------------
# Execution
# -------------------------
if __name__ == "__main__":
    # Allows running directly via `python src/birdwatch_ai/serve.py`
    uvicorn.run(
        "birdwatch_ai.serve:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False # reload=False for production safety; enable only in local dev
    )