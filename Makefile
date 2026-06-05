.PHONY: help install data features train autoencoder gnn drift app api test lint clean all

PY ?= python

help:
	@echo "Targets:"
	@echo "  install     pip install -r requirements.txt"
	@echo "  data        download Sparkov dataset"
	@echo "  features    build engineered feature tables"
	@echo "  train       train + tune the production LightGBM (+ imbalance study)"
	@echo "  autoencoder train the unsupervised autoencoder baseline"
	@echo "  gnn         train the graph neural network"
	@echo "  drift       run the concept-drift (PSI) report"
	@echo "  app         launch the Gradio dashboard"
	@echo "  api         launch the FastAPI real-time scoring server"
	@echo "  test        run the unit tests"
	@echo "  lint        flake8"
	@echo "  all         data -> features -> train -> autoencoder -> gnn"

install:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) scripts/download_data.py

features:
	$(PY) scripts/run_features.py

train:
	$(PY) scripts/run_training.py

autoencoder:
	$(PY) scripts/run_autoencoder.py

gnn:
	$(PY) scripts/run_gnn.py

drift:
	$(PY) scripts/run_drift.py

app:
	$(PY) app/gradio_app.py

api:
	uvicorn api.main:app --reload --port 8000

test:
	$(PY) -m pytest tests/ -v

lint:
	flake8 src/ tests/ scripts/ app/ api/

all: data features train autoencoder gnn

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache mlruns coverage.xml
