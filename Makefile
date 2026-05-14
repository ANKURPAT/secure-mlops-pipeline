.PHONY: help install mlflow-server train test lint clean

# Use the venv's Python explicitly so `make` doesn't depend on shell aliases
# or whether you remembered to `source .venv/bin/activate`.
PYTHON := .venv/bin/python
MLFLOW := .venv/bin/mlflow

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install:  ## Install dev dependencies
	$(PYTHON) -m pip install -r requirements-dev.txt

mlflow-server:  ## Start the MLflow tracking + registry server
	$(MLFLOW) server \
		--host 127.0.0.1 --port 5000 \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlartifacts

train:  ## Run a training run against the local MLflow server
	MLFLOW_TRACKING_URI=http://127.0.0.1:5000 $(PYTHON) -m ml.train

test:  ## Run tests
	$(PYTHON) -m pytest -v --cov=ml --cov=app

lint:  ## Lint with ruff
	$(PYTHON) -m ruff check ml app tests
	$(PYTHON) -m ruff format --check ml app tests

clean:  ## Remove caches and local MLflow state
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov __pycache__
	find . -name __pycache__ -exec rm -rf {} +

.PHONY: serve serve-prod

serve:
	.venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

serve-prod:
	.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2

IMAGE_NAME  := secure-mlops-pipeline
IMAGE_TAG   := dev

.PHONY: docker-build docker-run docker-stop trivy-scan

docker-build:
	podman build --tag $(IMAGE_NAME):$(IMAGE_TAG) .

docker-run:
	podman run --rm \
		--name fraud-api \
		-p 8000:8000 \
		-e MLFLOW_TRACKING_URI=http://host.containers.internal:5000 \
		$(IMAGE_NAME):$(IMAGE_TAG)

docker-stop:
	podman stop fraud-api || true

trivy-scan:
	mkdir -p security/scans
	podman save $(IMAGE_NAME):$(IMAGE_TAG) -o /tmp/$(IMAGE_NAME)-$(IMAGE_TAG).tar
	trivy image \
		--input /tmp/$(IMAGE_NAME)-$(IMAGE_TAG).tar \
		--severity HIGH,CRITICAL \
		--format table \
		--output security/scans/trivy-$(IMAGE_TAG).txt
	cat security/scans/trivy-$(IMAGE_TAG).txt