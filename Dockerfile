# Real-time fraud-scoring API
FROM python:3.11-slim

WORKDIR /app

# System deps for lightgbm
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir \
    numpy pandas scipy scikit-learn pyarrow lightgbm shap \
    fastapi uvicorn pydantic joblib

COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
