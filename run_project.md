# Run Project Guide: MLOps Weather Early Warning System

This guide provides instructions on how to start, execute, test, and demonstrate the **End-to-End MLOps Pipeline for Real-Time Model Monitoring, Drift Detection, and Automated Retraining**.

The project includes an automation **`Makefile`** that eliminates multi-step manual commands, auto-detects your virtual environment, and guarantees correct Python module pathing.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Quick-Start & 5-Minute Live Demo Workflow (Recommended)](#2-quick-start--5-minute-live-demo-workflow-recommended)
3. [Makefile Command Reference Table](#3-makefile-command-reference-table)
4. [Step-by-Step Manual Command Equivalents](#4-step-by-step-manual-command-equivalents)
   - [Step 1: Data Preparation](#step-1-data-preparation)
   - [Step 2: Train Offline Baseline Model](#step-2-train-offline-baseline-model)
   - [Step 3: Docker Infrastructure](#step-3-docker-infrastructure)
   - [Step 4: Register Model in MLflow](#step-4-register-model-in-mlflow)
   - [Step 5: Access the GIS Dashboard](#step-5-access-the-gis-dashboard)
   - [Step 6: Stream Telemetry & Induce Drift](#step-6-stream-telemetry--induce-drift)
   - [Step 7: Run Automated Tests](#step-7-run-automated-tests)
   - [Step 8: Teardown & Reset](#step-8-teardown--reset)
5. [Service Architecture & Port Reference](#5-service-architecture--port-reference)
6. [Troubleshooting & FAQs](#6-troubleshooting--faqs)

---

## 1. Prerequisites

Before running the commands, ensure the following are installed and running:
- **Python 3.11+** with virtual environment created (`.venv`)
- **Docker Desktop** (Open Docker Desktop and ensure the engine status is green/running)
- Available local ports: `8000` (FastAPI / Dashboard), `6379` (Redis), `29092` (Kafka), `5001` (MLflow), `8080` (Airflow)

---

## 2. Quick-Start & 5-Minute Live Demo Workflow (Recommended)

Run all commands from inside the `mlops-weather-early-warning/` directory:

```bash
cd mlops-weather-early-warning
```

### Step 1: Initialize Everything (One-Command Setup)
```bash
make init
```
- **What this does:** Fetches raw data, cleans and encodes spatial zones, trains Model v1 (XGBoost), profiles reference baseline distributions into `baseline_stats.json`, launches the full Docker Compose stack (Kafka, Redis, MLflow, Airflow, Inference Service, Drift Monitor), waits until services are healthy, and registers Model v1 under MLflow's `"Production"` alias.

### Step 2: Open the GIS Monitoring Dashboard
```bash
make open
```
- **What this does:** Opens [http://localhost:8000/](http://localhost:8000/) in your default browser.
- **Visual Checkpoints:**
  - **Left Fixed Navigation Bar**: Displays live system badges (WebSocket stream, Active model version, spatial drift status, 4-city drift gate indicator) and quick-scroll navigation links (`Overview & Vitals`, `Current Model`, `Model Performances`, `MLOps Audit Stream`).
  - **Current Model Section**: Shows currently deployed model version, deployment/training timestamps, model serving status (`HEALTHY & ACTIVE`), hazard risk scores, Monitored Cities & Risk Scores table, and interactive Leaflet GIS map with 6 Western Ghats zones.
  - **Model Performances Section**: Shows Multi-City Drift Alert Banner, active model evaluation metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC), separate Rolling PSI drift chart (`PSI = 0.20`), Champion vs Challenger performance comparison chart, City Drift & Threshold Matrix, and Model Evaluation History ledger.
  - **MLOps Audit Stream Section**: Live chronological log of prediction, drift, retrain, and hot-reload events.

### Step 3: Stream Normal Baseline Telemetry
```bash
make demo-normal
```
- **What this does:** Streams 100 normal pre-monsoon weather events at 10 events/sec into Kafka topic `raw-telemetry`.
- **What to observe:** Green circle markers (`< 0.4` risk) update on the map; event count increments; PSI on the drift chart remains low (`< 0.10`).

### Step 4: Inject Severe Monsoon Drift & Trigger CT Retraining
```bash
make demo-drift
```
- **What this does:** Injects 200 severe monsoon events (1.5x amplified rainfall/moisture) to induce spatial-temporal drift.
- **What to observe:**
  1. Amber (`0.4-0.7`) and Red (`>= 0.7`) convective hazard pins appear on the map.
  2. Rolling PSI crosses the dashed threshold line (`PSI > 0.20`).
  3. Affected spatial zone outline turns into a **pulsing, dashed red border**.
  4. Header drift badge changes to `Drift Detected`.
  5. Audit log reports `[DRIFT_ALERT]` and automated Airflow retraining trigger `[RETRAIN]`.
  6. Airflow trains Challenger Model v2, validates improvement (+1% F1), promotes it in MLflow, and dispatches zero-downtime hot-reload.
  7. Top model badge flips to green: **`Model: v2`** without service downtime.

### Step 5: Check Infrastructure Health & Metrics
```bash
make status
```
- **What this does:** Prints `docker compose ps` status table alongside formatted JSON from `/health` and `/metrics/summary`.

### Step 6: Run Full Test Suite
```bash
make test
```
- **What this does:** Executes all 60 tests across unit and integration suites (Phases 1-8).

### Step 7: Teardown
```bash
make down
```
- **What this does:** Stops containers while preserving databases, volumes, and registered models. (Use `make reset` to wipe all data for a fresh start).

---

## 3. Makefile Command Reference Table

All targets automatically detect `../.venv/bin/python`, `.venv/bin/python`, or `$VIRTUAL_ENV`, and set `PYTHONPATH=.`.

| Target | Command Executed Under the Hood | Purpose |
| :--- | :--- | :--- |
| `make help` | Formatted recipe help echo | Lists all available make targets and descriptions |
| `make init` | `make prepare` -> `make up` -> wait healthy -> `make register` | Full one-command first-time setup and service registration |
| `make prepare` | `python -m ml_core.preprocessing.data_fetcher`<br>`python -m ml_core.preprocessing.cleaner`<br>`python -m ml_core.training.train_baseline` | Generates datasets, encodes zones, trains Model v1, and profiles baseline stats |
| `make up` | `docker compose -f infrastructure/docker-compose.yml up -d` | Starts Kafka, Redis, MLflow, Airflow, API, and Drift Monitor |
| `make status` | `docker compose ps`<br>`curl http://localhost:8000/health`<br>`curl http://localhost:8000/metrics/summary` | Inspects container status and prints formatted service health metrics |
| `make register` | `python infrastructure/scripts/register_baseline_mlflow.py`<br>`curl -X POST http://localhost:8000/reload-model` | Registers Model v1 in MLflow as `"Production"` and triggers API reload |
| `make open` | `open http://localhost:8000/` (or `xdg-open`) | Opens GIS Monitoring Dashboard in default web browser |
| `make demo-normal` | `python services/ingestion/telemetry_producer.py --rate 10.0 --mode normal --max-events 100` | Streams 100 baseline events to Kafka `raw-telemetry` |
| `make demo-drift` | `python services/ingestion/telemetry_producer.py --rate 20.0 --mode monsoon --drift-multiplier 1.5 --max-events 200` | Injects 200 severe monsoon events to trigger drift and CT retraining |
| `make demo-transition` | `python services/ingestion/telemetry_producer.py --rate 10.0 --mode transition --max-events 300` | Replays gradual seasonal transition from baseline to monsoon drift |
| `make logs` | `docker compose -f infrastructure/docker-compose.yml logs -f inference-service drift-monitor` | Follows live logs for inference service and background drift detector |
| `make test` | `pytest tests/ -v` | Executes the complete 60-test regression suite across all 8 phases |
| `make down` | `docker compose -f infrastructure/docker-compose.yml down` | Stops containers while preserving volumes and models |
| `make reset` | `docker compose -f infrastructure/docker-compose.yml down -v` | Full reset: stops containers and deletes all persistent volumes |

---

## 4. Step-by-Step Manual Command Equivalents

If you prefer executing the individual Python commands directly without `make`, use the following commands with `PYTHONPATH=.`:

### Step 1: Data Preparation
```bash
python -m ml_core.preprocessing.data_fetcher
python -m ml_core.preprocessing.cleaner
```
- Fetches 12,000+ hourly weather records and splits them into `data/processed/baseline_train.csv` and `data/processed/stream_simulation.csv`.

### Step 2: Train Offline Baseline Model
```bash
python -m ml_core.training.train_baseline
```
- Trains Model v1 (`ml_core/models/model_v1.joblib`) and computes baseline quantiles (`data/reference/baseline_stats.json`).

### Step 3: Docker Infrastructure
```bash
cd infrastructure
docker compose up -d
docker compose ps
cd ..
```
- Starts Kafka (KRaft), Redis, MLflow, Airflow, Inference Service, and Drift Monitor on network `mlops-net`.

### Step 4: Register Model in MLflow
```bash
python infrastructure/scripts/register_baseline_mlflow.py
```
- Registers `model_v1.joblib` in MLflow under the `"Production"` alias.

### Step 5: Access the GIS Dashboard
Open your browser at **[http://localhost:8000/](http://localhost:8000/)**.

### Step 6: Stream Telemetry & Induce Drift
```bash
# Normal baseline streaming
python services/ingestion/telemetry_producer.py --rate 10.0 --mode normal --max-events 100

# Monsoon drift injection
python services/ingestion/telemetry_producer.py --rate 20.0 --mode monsoon --drift-multiplier 1.5 --max-events 200
```

### Step 7: Run Automated Tests
```bash
pytest tests/ -v
```

### Step 8: Teardown & Reset
```bash
cd infrastructure
docker compose down      # Preserve data
docker compose down -v   # Full wipe
```

---

## 5. Service Architecture & Port Reference

| Service | Host Port | Internal Port | URL / Interface | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **GIS Dashboard & Inference** | `8000` | `8000` | [http://localhost:8000/](http://localhost:8000/) | Live Leaflet map, Chart.js PSI monitor, scoring API, and WebSocket server |
| **Apache Airflow** | `8080` | `8080` | [http://localhost:8080/](http://localhost:8080/) | DAG scheduler & monitoring UI (Credentials: `airflow` / `airflow`) |
| **MLflow Server** | `5001` | `5000` | [http://localhost:5001/](http://localhost:5001/) | Experiment tracking, model metrics, and Model Registry |
| **Redis** | `6379` | `6379` | `localhost:6379` | Spatial sliding window feature store and debounce locks |
| **Apache Kafka** | `29092` | `9092` | `localhost:29092` | KRaft event streaming broker for telemetry, predictions, and alerts |

---

## 6. Troubleshooting & FAQs

### Q1: `ModuleNotFoundError: No module named 'ml_core'`
- **Why this occurred:** Executing Python scripts via path syntax (`python dir/script.py`) populates `sys.path[0]` with `dir/` rather than the repository root.
- **Resolution:** Use `make <target>` or `python -m <module>` or set `export PYTHONPATH=.`. In addition, all scripts now include automatic `sys.path` bootstrapping.

### Q2: `failed to connect to the docker API at unix:///Users/.../.docker/run/docker.sock`
- **Why this occurs:** Docker Desktop is not running.
- **Resolution:** Open Docker Desktop and wait until the whale icon shows "Engine running" before executing `make up` or `make init`.

### Q3: What are the credentials for the Airflow Webserver?
- **Username:** `airflow`
- **Password:** `airflow`
- **URL:** [http://localhost:8080](http://localhost:8080)
