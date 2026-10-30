.PHONY: help install install-train lint format test cov audit train sweep export benchmark \
        loadtest calibrate report fetch index serve docker docker-size up down clean all

PY ?= python3
MODEL_DIR ?= artifacts/run/model
ONNX_DIR ?= artifacts/onnx
IMAGE ?= medextract:local
API_URL ?= http://localhost:8000

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Runtime + dev dependencies
	$(PY) -m pip install -e ".[dev,serve]"

install-train:  ## Everything needed to fine-tune and export
	$(PY) -m pip install -e ".[dev,serve,train]"

lint:  ## ruff check + format check
	ruff check .
	ruff format --check .

format:  ## Apply ruff formatting and autofixes
	ruff check --fix .
	ruff format .

test:  ## Run the test suite
	pytest -q

cov:  ## Test suite with coverage
	pytest -q --cov=medextract --cov-report=term-missing

all:  ## The whole pipeline, unattended: sweep, export, benchmark, retrieval, serve, README
	$(PY) -m medextract.pipeline

train:  ## One fine-tuning run (see train.py for flags)
	$(PY) -m medextract.train

sweep:  ## The full comparison sweep
	$(PY) -m medextract.sweep

report:  ## Regenerate the MLflow comparison table
	$(PY) -m medextract.report --experiment medextract-ner > docs/sweep_table.md

export:  ## Export to ONNX and verify parity with PyTorch
	$(PY) -m medextract.export_onnx --model-dir $(MODEL_DIR) --output-dir $(ONNX_DIR)
	$(PY) -m medextract.retrieval.export_embedder --output-dir artifacts/embedder

benchmark:  ## PyTorch vs ONNX latency table
	$(PY) -m medextract.benchmark --torch-dir $(MODEL_DIR) --onnx-dir $(ONNX_DIR)

loadtest:  ## Latency under concurrency against a running API (make up first)
	$(PY) -m medextract.loadtest --url $(API_URL) --corpus data/pubmed.jsonl

calibrate:  ## Reliability of the entity score on the validation split
	$(PY) -m medextract.calibration --model-dir $(ONNX_DIR)

fetch:  ## Download PubMed abstracts
	$(PY) -m medextract.retrieval.pubmed --max 3000

index:  ## Embed and index the abstracts into Qdrant
	$(PY) -m medextract.retrieval.build_index

serve:  ## Run the API locally with reload
	uvicorn medextract.api.main:app --reload --port 8000

docker:  ## Build the serving image
	docker build -t $(IMAGE) .

docker-size:  ## Print the built image size in MB
	@docker image inspect $(IMAGE) --format '{{.Size}}' | awk '{printf "%d MB\n", $$1/1000000}'

up:  ## Bring up the API and Qdrant
	docker compose up --build -d

down:  ## Tear it down
	docker compose down

clean:  ## Remove caches and build output
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist src/*.egg-info
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
