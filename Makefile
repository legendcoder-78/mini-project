# ==============================================================================
# MLOps Weather Early Warning System - Automation Makefile
# ==============================================================================
# Resolves virtual environment automatically and handles Python pathing.
# Works from within mlops-weather-early-warning directory.

.PHONY: help prepare up status register init demo-normal demo-drift demo-transition open logs test down reset

# Detect python and pytest executables with automatic virtualenv resolution
PYTHON := $(shell if [ -f ../.venv/bin/python ]; then echo "../.venv/bin/python"; elif [ -f .venv/bin/python ]; then echo ".venv/bin/python"; elif [ -n "$$VIRTUAL_ENV" ]; then echo "$$VIRTUAL_ENV/bin/python"; else echo "python3"; fi)
PYTEST := $(shell if [ -f ../.venv/bin/pytest ]; then echo "../.venv/bin/pytest"; elif [ -f .venv/bin/pytest ]; then echo ".venv/bin/pytest"; elif [ -n "$$VIRTUAL_ENV" ]; then echo "$$VIRTUAL_ENV/bin/pytest"; else echo "pytest"; fi)

# Cross-platform browser launcher
OPEN_CMD := $(shell which open 2>/dev/null || which xdg-open 2>/dev/null || echo "echo Open browser at")

# Ensure repository root is always in PYTHONPATH
export PYTHONPATH := .

# Default Target
help:
	@echo "=============================================================================="
	@echo " MLOps Weather Early Warning System - Available Commands"
	@echo "=============================================================================="
	@echo " Setup & Infrastructure:"
	@echo "   make init             Full one-command first-time setup (prepare -> up -> register)"
	@echo "   make prepare          Generate datasets, encode zones, train Model v1 & stats"
	@echo "   make up               Launch Docker Compose stack (Kafka, Redis, MLflow, Airflow, API)"
	@echo "   make register         Register baseline Model v1 into MLflow Registry as Production"
	@echo "   make status           Inspect running Docker containers and health endpoints"
	@echo "   make open             Open GIS Monitoring Dashboard in default web browser"
	@echo ""
	@echo " Live Telemetry & Evaluation Demos:"
	@echo "   make demo-normal      Stream 100 normal baseline weather events (rate: 10/s)"
	@echo "   make demo-drift       Inject 200 severe monsoon drift events to trigger retrain"
	@echo "   make demo-transition  Stream 300 events transitioning from normal to monsoon"
	@echo ""
	@echo " Monitoring & Verification:"
	@echo "   make logs             Follow live logs for inference service and drift monitor"
	@echo "   make test             Run full 60-test pytest suite across all 8 phases"
	@echo ""
	@echo " Teardown & Reset:"
	@echo "   make down             Stop containers while preserving volumes and databases"
	@echo "   make reset            Full reset: stop containers and wipe all persistent volumes"
	@echo "=============================================================================="

prepare:
	@echo "--> Step 1/3: Fetching / generating raw telemetry..."
	$(PYTHON) -m ml_core.preprocessing.data_fetcher
	@echo "--> Step 2/3: Cleaning, geospatial encoding, and dataset splitting..."
	$(PYTHON) -m ml_core.preprocessing.cleaner
	@echo "--> Step 3/3: Training baseline model (Model v1) and profiling reference statistics..."
	$(PYTHON) -m ml_core.training.train_baseline

up:
	@echo "--> Starting Docker containerized infrastructure..."
	docker compose -f infrastructure/docker-compose.yml up -d
	@echo "--> Waiting for containers to initialize..."
	@docker compose -f infrastructure/docker-compose.yml ps

status:
	@echo "=== Container Status ==="
	@docker compose -f infrastructure/docker-compose.yml ps
	@echo ""
	@echo "=== Inference Service Health ==="
	@curl -s http://localhost:8000/health | $(PYTHON) -m json.tool 2>/dev/null || curl -s http://localhost:8000/health || echo "Inference service unreachable on port 8000"
	@echo ""
	@echo "=== Metrics Summary ==="
	@curl -s http://localhost:8000/metrics/summary | $(PYTHON) -m json.tool 2>/dev/null || curl -s http://localhost:8000/metrics/summary || echo "Metrics unavailable"

register:
	@echo "--> Registering Model v1 into MLflow Registry under 'Production' alias..."
	$(PYTHON) infrastructure/scripts/register_baseline_mlflow.py
	@echo "--> Triggering hot-reload on Inference Service..."
	@curl -s -X POST http://localhost:8000/reload-model || true

init: prepare up
	@echo "--> Waiting for MLflow and Inference services to become healthy..."
	@for i in $$(seq 1 30); do \
		if curl -s http://localhost:5001/health >/dev/null 2>&1 && curl -s http://localhost:8000/health >/dev/null 2>&1; then \
			echo "Infrastructure is healthy and responding!"; \
			break; \
		fi; \
		echo "Waiting for services... ($$i/30)"; \
		sleep 2; \
	done
	@$(MAKE) register
	@echo ""
	@echo "=============================================================================="
	@echo " MLOps Weather Early Warning System is initialized and running!"
	@echo " Access the live GIS Dashboard at: http://localhost:8000/"
	@echo " Run 'make open' to launch the dashboard in your default browser."
	@echo " Run 'make demo-normal' or 'make demo-drift' to stream events."
	@echo "=============================================================================="

demo-normal:
	@echo "--> Streaming 100 normal pre-monsoon events into Kafka 'raw-telemetry'..."
	$(PYTHON) services/ingestion/telemetry_producer.py --rate 10.0 --mode normal --max-events 100

demo-drift:
	@echo "--> Injecting 200 severe monsoon drift events (drift-multiplier 1.5)..."
	$(PYTHON) services/ingestion/telemetry_producer.py --rate 20.0 --mode monsoon --drift-multiplier 1.5 --max-events 200

demo-transition:
	@echo "--> Streaming 300 transition events (normal to monsoon shift)..."
	$(PYTHON) services/ingestion/telemetry_producer.py --rate 10.0 --mode transition --max-events 300

open:
	@echo "--> Opening GIS Dashboard at http://localhost:8000/ ..."
	@$(OPEN_CMD) http://localhost:8000/ || echo "Please navigate to http://localhost:8000/ in your browser."

logs:
	@echo "--> Following logs for inference-service and drift-monitor (Ctrl+C to stop)..."
	docker compose -f infrastructure/docker-compose.yml logs -f inference-service drift-monitor

test:
	@echo "--> Running full test suite across all 8 phases..."
	$(PYTEST) tests/ -v

down:
	@echo "--> Stopping Docker containers (volumes and databases preserved)..."
	docker compose -f infrastructure/docker-compose.yml down

reset:
	@echo "--> Full reset: stopping containers and removing all named volumes..."
	docker compose -f infrastructure/docker-compose.yml down -v
