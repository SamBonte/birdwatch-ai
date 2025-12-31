# Use Python 3.11 for TensorFlow stability
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Ensure the app code is discoverable as a package
ENV PYTHONPATH=/app/src

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Option 2: Copy the source code AND the models from the internal path
# src/ contains birdwatch_ai/ which contains models/
COPY src/ ./src/

# Verify the structure for the user:
# Your code expects: /app/src/birdwatch_ai/models/mobilenetv2_finetuned/mobilenetv2_final.keras
# This COPY command ensures that path exists inside the container.

# Expose FastAPI port
EXPOSE 8000

# Run using the module path
CMD ["uvicorn", "birdwatch_ai.serve:app", "--host", "0.0.0.0", "--port", "8000"]