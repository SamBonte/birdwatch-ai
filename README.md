# BirdWatch AI

**BirdWatch AI** is a production-ready computer vision system for classifying **25 bird species** from images using a **fine-tuned MobileNetV2** model.
The project supports **local execution**, **FastAPI-based inference**, and **Dockerized deployment**.

---

## Features

* Image-based bird species classification (25 classes)
* Fine-tuned **MobileNetV2** for efficient inference
* Idempotent dataset download from Kaggle
* CLI, Python, FastAPI, and Docker support
* Production-oriented project structure
* Cross-platform (Windows / Linux / macOS)

---

## Project Structure (Simplified)

```
birdwatch-ai/
├── .git/
├── .venv/
├── notebooks/
│   └── exploration.ipynb # experimentation with different models
│
├── src/
│   └── birdwatch_ai/
│       ├── data/ # contains raw & processed dataset after running data.py
│       │
│       ├── models/ # contains folder with model & metrics after training
│       │   
│       │
│       ├── __init__.py      # birdwatch_ai package
│       ├── config.py        # Central configuration (paths, classes, hyperparameters)
│       ├── data.py          # Dataset download & filtering (Kaggle, idempotent)
│       ├── model.py         # MobileNetV2 architecture & fine-tuning
│       ├── train.py         # Training pipeline
│       ├── evaluate.py      # Evaluation & metrics generation
│       ├── predict.py       # CLI & API inference
│       ├── serve.py         # FastAPI application
│       ├── run.py           # High-level orchestration (CLI)
│       └── utils.py         # Shared helper functions
│
├── tests/                   # Contains a test image Common Raven
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
├── requirements.txt
├── run_birdwatch.py         # run birdwatch-ai commands from project root
├── setup.py                 # setup script for birdwatch-ai package.
└── vercel.json # possibility to work with vercel
```

---

## Requirements

* **Python 3.13.2** (local development)
* **Docker Desktop** (for containerized usage)
* Kaggle account (for dataset download)

---

## Dataset

* **Bird Species Classification – 200 Categories**
* Author: **Kedar**
* Source: Kaggle

The project automatically downloads and prepares a **filtered subset of 25 classes**.

Dataset download is **idempotent**:
If the dataset already exists locally, it will **not** be re-downloaded.

---

## Kaggle Setup (Required Once)

1. Create a Kaggle API token (`kaggle.json`)
2. Place it in one of the following locations:

   * `~/.kaggle/kaggle.json` (recommended)
3. Ensure permissions are correct:

   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

---

## Running the Project

You can run the project in **three ways**.

---

## 1 Local (Python / CLI)

### Install dependencies

```bash
pip install -e .
```

### Prepare dataset

```bash
python -m birdwatch_ai.run prepare
```

### Train the model

```bash
python -m birdwatch_ai.run train
```

### Evaluate the model

```bash
python -m birdwatch_ai.run evaluate
```

### Run full pipeline

```bash
python -m birdwatch_ai.run pipeline
```

### Predict on a single image

```bash
python -m birdwatch_ai.predict path/to/bird.jpg
```

> Supported formats: **.jpg / .jpeg / .png**
> Best performance with clear, centered bird images.

---

## 2 FastAPI (Local Inference Server)

### Start the API

```bash
uvicorn birdwatch_ai.serve:app --host 127.0.0.1 --port 8000
```

### Interactive API docs

Open in your browser:

```
http://127.0.0.1:8000/docs
```

### Health check

```bash
curl http://127.0.0.1:8000/health
```

### Predict via API

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -F "file=@path/to/bird.jpg"
```

Example response:

```json
{
  "predicted_class": "sparrow",
  "confidence": 0.87,
  "filename": "bird.jpg"
}
```

---

## 3 Docker (Recommended for Deployment)

### Build the image

```bash
docker build -t birdwatch-ai .
```

### Run the container

```bash
docker run -p 8000:8000 birdwatch-ai
```

### Access the API

* Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
* Health: [http://localhost:8000/health](http://localhost:8000/health)

> ℹThe Docker image runs the **FastAPI server** by default
> Python base image: **3.11-slim** (TensorFlow compatible)

---

## Limitations

* Only **25 bird species** supported
* Optimized for **.jpg / .jpeg / .png**
* Not a detection model (classification only)
* Requires Kaggle access for dataset download

---

## Citation

If you use this project or dataset, please cite:

> **Kedar**, *Bird Species Classification – 200 Categories*, Kaggle

---

## Author Notes

This project was designed with **production deployment in mind**, focusing on:

* Clean architecture
* Reproducibility
* Minimal assumptions
* Clear separation of concerns



