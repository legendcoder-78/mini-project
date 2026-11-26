# MLOps Weather Early Warning System

## End-to-End MLOps Pipeline for Real-Time Model Monitoring, Drift Detection, and Automated Retraining

This repository implements a modular, containerized, closed-loop **Continuous Training / Continuous Deployment (CT/CD)** weather intelligence platform. The system is designed to predict extreme convective weather events (thunderstorms, cloudbursts, severe flash flood triggers) across the Western Ghats and Coastal Karnataka regions, detect spatial-temporal drift under seasonal shifts, and automatically retrain and deploy updated models with zero downtime.

---

## Repository Structure

```text
mlops-weather-early-warning/
├── Makefile                           # Automation targets (init, up, open, demo-normal, demo-drift, test)
├── config/
│   ├── settings.yaml                  # Global features, drift limits, Kafka/Redis topics, MLflow URIs
│   └── spatial_zones.json             # Bounding boxes for Karnataka / Western Ghats zones
├── data/
│   ├── raw/                           # Raw weather telemetry CSVs (weather_telemetry_raw.csv)
│   ├── processed/                     # Cleaned datasets: baseline_train.csv & stream_simulation.csv
│   └── reference/                     # Reference baseline statistical profiles (baseline_stats.json)
├── frontend/
│   ├── index.html                     # Lightweight dark-mode GIS & MLOps dashboard (Leaflet + Chart.js via CDN)
│   ├── css/
│   │   └── styles.css                 # Responsive CSS Grid layout, status badges & audit log styling
│   └── js/
│       ├── map_controller.js          # Leaflet.js map, spatial zone boundary polygons & color-coded hazard pins
│       ├── chart_controller.js        # Chart.js live rolling PSI drift curves with threshold line (PSI = 0.20)
│       └── ws_client.js               # Native WebSocket client with auto-reconnect & UI state dispatcher
├── ml_core/
│   ├── models/                        # Local model artifacts (model_v1.joblib)
│   ├── evaluation/
│   │   ├── __init__.py                # Evaluation package exports
│   │   └── comparator.py              # Champion-Challenger validation gate (F1, Precision, Recall, ROC-AUC)
│   ├── preprocessing/
│   │   ├── __init__.py                # Preprocessing package exports
│   │   ├── geo_encoder.py             # Deterministic & vectorized geospatial zone encoder
│   │   ├── data_fetcher.py            # Open-Meteo Archive API client & synthetic telemetry generator
│   │   └── cleaner.py                 # Schema enforcement, anomaly clamping & temporal splitter
│   └── training/
│       ├── __init__.py                # Training package exports
│       ├── profiler.py                # Statistical distribution & histogram profiler (PSI & KS)
│       ├── train_baseline.py          # Offline training & evaluation script for Model v1
│       └── retrain_pipeline.py        # Automated retraining pipeline (blends buffer with baseline)
├── infrastructure/
│   ├── docker-compose.yml             # Full distributed stack (Kafka, Redis, MLflow, Airflow, Inference, Drift Monitor)
│   ├── Dockerfile.airflow             # Custom Airflow image pre-installed with ML dependencies
│   ├── Dockerfile.service             # Inference and monitoring container image
│   └── scripts/
│       ├── init_kafka_topics.sh       # Creates raw-telemetry, predictions-log, drift-alerts (4 partitions)
│       └── register_baseline_mlflow.py# Registers model_v1.joblib into MLflow Registry as Production
├── orchestration/
│   └── airflow/
│       └── dags/
│           └── automated_retrain_dag.py # 4-task production CT DAG (extract -> train -> gate -> promote & reload)
├── services/
│   ├── __init__.py
│   ├── common/
│   │   ├── __init__.py
│   │   ├── schemas.py                 # Strict Pydantic v2 event contracts (Telemetry, Prediction, DriftAlert)
│   │   └── redis_buffer.py            # Bounded spatial-temporal sliding window manager in Redis
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── telemetry_producer.py      # Streams simulation CSV to Kafka with CLI replay speed & drift controls
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── model_loader.py            # Thread-safe MLflow Production model loader & atomic hot-reload manager
│   │   └── app.py                     # FastAPI server, WebSocket broadcaster, background Kafka consumers & GIS host
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── drift_detector.py          # Compatibility wrapper forwarding to drift_monitor
│   │   └── drift_worker.py            # Compatibility CLI wrapper forwarding to drift_monitor
│   └── drift_monitor/
│       ├── __init__.py
│       ├── detector.py                # Spatial-temporal drift detector (PSI + KS test with auto-reloading baselines)
│       └── worker.py                  # Polling worker, Kafka drift publisher, Redis debounce lock & Airflow dispatcher
├── tests/
│   ├── unit/
│   │   ├── test_phase1.py             # Unit tests for Phase 1 (Encoding, cleaning, split, drift)
│   │   ├── test_phase2.py             # Unit tests for Phase 2 (Model v1, inference, baseline stats)
│   │   ├── test_phase6_detector.py    # Unit tests for Phase 6 (PSI, KS calculation, baseline auto-reload)
│   │   └── test_phase7_comparator.py  # Unit tests for Phase 7 (Champion-Challenger promotion gate)
│   └── integration/
│       ├── test_phase3_infra.py       # Integration tests for Kafka, Redis, MLflow, and Airflow
│       ├── test_phase4_ingestion.py   # Integration tests for schemas, Redis buffer cap, and Kafka producer
│       ├── test_phase5_inference.py   # Integration tests for health, latency (<50ms), E2E streaming, and hot-reload
│       ├── test_phase6_worker.py      # Integration tests for drift polling, debounce locks, and Airflow dispatch
│       ├── test_phase7_retrain_dag.py # Integration tests for Airflow DAG, MLflow promotion, and zero-downtime reload
│       └── test_phase8_e2e.py         # Full closed-loop tests: dashboard hosting, `/api/zones`, and WebSocket broadcasts
├── requirements.txt                   # Pinned dependencies (fastapi, uvicorn, pydantic v2, kafka-python-ng, redis, mlflow, websockets)
├── run_project.md                     # Step-by-step execution guide and Makefile reference
└── README.md                          # Project documentation & runbook
```

---

## Spatial Zones (Karnataka & Western Ghats)

Configured in `config/spatial_zones.json` across 6 distinct climate corridors:

1. **`ZONE_KODAGU`**: Kodagu (Coorg Highlands) — High elevation, steep orographic cloudburst risk.
2. **`ZONE_DAKSHINA_KANNADA`**: Coastal Dakshina Kannada & Mangaluru — Coastal tropical convective storms.
3. **`ZONE_CHIKMAGALUR`**: Chikmagalur Malnad Ridge — High altitude coffee belt with heavy monsoon runoff.
4. **`ZONE_SHIMOGA_AGUMBE`**: Shivamogga & Agumbe Rainforest — One of the highest annual rainfall belts in India.
5. **`ZONE_UTTARA_KANNADA`**: Uttara Kannada Coastal & Foothills — Coastal/estuarine flood vulnerability.
6. **`ZONE_WAYANAD_NILGIRIS`**: Wayanad & Nilgiris Border Corridor — Landslide-sensitive southern Western Ghats transition.

Points outside these bounding boxes map deterministically to `OUT_OF_BOUNDS`.

---

## Infrastructure Stack

The distributed runtime runs on a single bridge network (`mlops-net`) optimized for local laptop resource budgets:

| Service | Container Name | Image / Base | Internal Port | Host Port | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Apache Kafka** | `mlops-kafka` | `apache/kafka:3.7.0` | `9092` | `29092` | Event streaming broker in single-node **KRaft** mode (no Zookeeper). |
| **Kafka Init** | `mlops-kafka-init` | `apache/kafka:3.7.0` | — | — | Automatically provisions `raw-telemetry`, `predictions-log`, `drift-alerts`. |
| **Redis** | `mlops-redis` | `redis:7-alpine` | `6379` | `6379` | In-memory feature store for sliding windows & debounce distributed locks. |
| **MLflow Server** | `mlops-mlflow` | `ghcr.io/mlflow/mlflow:v2.13.0` | `5000` | `5001` | Tracking & Model Registry with SQLite and artifact proxy (`--serve-artifacts`). |
| **PostgreSQL** | `mlops-airflow-postgres` | `postgres:15-alpine` | `5432` | — | Dedicated metadata database for Airflow `LocalExecutor`. |
| **Airflow Standalone**| `mlops-airflow` | `infrastructure-airflow` | `8080` | `8080` | Webserver & scheduler executing CT retraining DAGs. |
| **Inference Service** | `mlops-inference-service` | `infrastructure-service` | `8000` | `8000` | FastAPI scoring service, WebSocket hub, and static GIS dashboard host. |
| **Drift Monitor** | `mlops-drift-monitor` | `infrastructure-service` | — | — | Background worker computing PSI/KS and triggering Airflow DAG. |

---

## GIS Monitoring Dashboard & Frontend UI Components

The `frontend/` directory contains a zero-build (vanilla JS, Leaflet.js, and Chart.js via CDN), dark-mode mission control dashboard served directly by FastAPI at `http://localhost:8000/`.

Below is a complete guide to each visual component, what it indicates, and its underlying frontend architecture.

### 1. Frontend Files & Architectural Roles

| File | Technical Stack | Role & Responsibility |
| :--- | :--- | :--- |
| [index.html](file:///Users/abhi/Documents/mini-project/mlops-weather-early-warning/frontend/index.html) | Semantic HTML5 | Defines the modern command view layout with a fixed, scrollable left-side navigation bar (`.sidebar-nav`) and organized sections for Overview, Current Model, Model Performances, and MLOps Audit Log. Loads Leaflet and Chart.js from CDN (zero npm/build steps). |
| [styles.css](file:///Users/abhi/Documents/mini-project/mlops-weather-early-warning/frontend/css/styles.css) | Modern CSS Grid & Variables | Dark-mode palette (`#0a0e17` slate theme), custom color variables (`--green`, `--amber`, `--red`, `--blue`, `--purple`), fixed sidebar positioning (`270px`), pulsing badge animations, scrollable tables, and responsive cards. |
| [map_controller.js](file:///Users/abhi/Documents/mini-project/mlops-weather-early-warning/frontend/js/map_controller.js) | Leaflet.js (v1.9.4) | CartoDB DarkMatter basemap tiles, dynamically renders spatial bounding rectangles from `GET /api/zones`, plots and updates color-coded sensor markers, and activates dashed red borders on drifted zones. |
| [chart_controller.js](file:///Users/abhi/Documents/mini-project/mlops-weather-early-warning/frontend/js/chart_controller.js) | Chart.js (v4.4.1) | Manages two separate visualizations: (1) `ChartController` maintaining live rolling PSI curves per spatial zone against the configured reference threshold (`PSI = 0.20`), and (2) `PerformanceChartController` rendering Champion vs. Challenger model metric comparisons (Accuracy, Precision, Recall, F1-Score, ROC-AUC). |
| [ws_client.js](file:///Users/abhi/Documents/mini-project/mlops-weather-early-warning/frontend/js/ws_client.js) | Native WebSocket API | Connects to `ws://localhost:8000/ws/telemetry` with exponential-backoff auto-reconnect, hydrates live model metadata and metrics from `GET /api/model/performance`, synchronizes navigation scrollspy, updates tables, and handles multi-city drift alerts. |

---

### 2. Fixed Left-Side Navigation Bar (`.sidebar-nav`)
A scrollable, fixed navigation drawer on the left side of the screen that remains accessible at all times:
- **Brand Header**: Displays project title (`Early Warning EWS: Western Ghats & Karnataka`).
- **Section Navigation Links**: Quick smooth-scrolling links with active section highlighting:
  - 📊 **Overview & Vitals** (`#section-overview`)
  - 🎯 **Current Model** (`#section-current-model`)
  - 📈 **Model Performances** (`#section-model-performance`)
  - 🛡️ **MLOps Audit Stream** (`#section-audit-log`)
- **Live System Status Badges**:
  - **Telemetry Stream** (`#ws-badge`): 🟢 `Live Stream` (connected) / 🟡 `Connecting...` / 🔴 `Reconnecting`.
  - **Active Model** (`#model-badge`): 🟣 `Model: v1` / 🟢 `Model: v2+` (indicates automated promotion).
  - **Spatial Drift** (`#drift-badge`): 🟢 `Drift: Nominal` / 🟡 `Drift: ZONE` / 🟣 `Retrain Triggered`.
  - **Multi-City Status** (`#multi-city-badge`): 🟢 `4-City Gate: Normal` / 🔴 `Multi-City Alert (X/6)`.
- **Real-Time Counters**:
  - **Events Processed** (`#event-counter`): Total count of scored weather records.
  - **Avg Inference Latency** (`#avg-latency`): Rolling average latency in ms (sub-50ms SLA).

---

### 3. Current Model Section (`#section-current-model`)
Displays complete operational and serving details for the model currently serving production traffic:
1. **Model Identifier & Registry Card**:
   - **Model Version** (`#cm-version`): Current production version (`v1`, `v2`, etc.).
   - **Registry Source** (`#cm-source`): Origin path in MLflow Model Registry (`weather-hazard-classifier@Production`).
   - **Serving Status** (`#cm-status-badge`): Displays `HEALTHY & ACTIVE`.
2. **Deployment Timestamps Card**:
   - **Deployed / Reloaded At** (`#cm-loaded-at`): Exact timestamp when the model was loaded into inference memory.
   - **Training Timestamp** (`#cm-trained-at`): Creation timestamp recorded during model training.
   - **Inference Features**: 5 atmospheric features (`rainfall_mm`, `soil_moisture`, `temperature_c`, `humidity_pct`, `wind_speed_kmh`).
3. **Current Risk Telemetry Card**:
   - **Max Hazard Score** (`#cm-max-risk`): Peak risk probability across all monitored cities.
   - **High Risk Cities** (`#cm-high-risk-count`): Count of cities currently at or above severe threshold (&ge; 70%).
4. **Interactive Leaflet GIS Map (`#gis-map`)**:
   - **Micro-Climate Zones**: 6 bounding polygons loaded dynamically from `GET /api/zones`.
   - **Color-Coded Sensor Pins**: 🟢 Low Risk (`< 0.40`), 🟡 Moderate Hazard (`0.40 - 0.70`), 🔴 Severe Hazard (`≥ 0.70`).
   - **Drifted Zone Borders**: Dashed red outline (`dashArray: 6, 6`) indicates active distribution drift (`PSI > 0.20`).
5. **Monitored Cities & Predicted Hazard Severity Table (`#city-risk-table`)**:
   - Lists all monitored cities (Kodagu, Dakshina Kannada, Chikmagalur, Shimoga / Agumbe, Uttara Kannada, Wayanad / Nilgiris), coordinates, current hazard probability (%), severity badge, and scoring latency.

---

### 4. Model Performances Section (`#section-model-performance`)
Dedicated to continuous model evaluation, drift monitoring, and Champion vs. Challenger comparisons:
1. **Multi-City Drift Alert Banner (`#multi-city-drift-banner`)**:
   - **Trigger Condition**: Displayed when **4 or more monitored cities** simultaneously exceed `PSI > 0.20`.
   - **MLOps Safety Guarantee**: Notifies operators that automated Airflow retraining has been dispatched while confirming that the existing Champion model remains in active production until a Challenger passes the evaluation gate (+1.00% F1 improvement).
2. **Active Model Quality Metrics Cards**:
   - Real-time metrics sourced directly from verified MLflow evaluation artifacts:
     - **Accuracy** (`#perf-accuracy`)
     - **Precision** (`#perf-precision`)
     - **Recall** (`#perf-recall`)
     - **F1-Score** (`#perf-f1`) — Primary promotion gate
     - **ROC-AUC** (`#perf-roc-auc`)
   - Distinguishes unrecorded metrics with `"N/A"` (never fabricates zero values).
3. **Two Strictly Separated Visualizations**:
   - **Live Spatial Drift Monitor (`#drift-chart`)**: Rolling 30-point Chart.js line graph of real-time PSI per zone against a dashed red threshold line (`PSI = 0.20`).
   - **Champion vs. Challenger Comparison (`#performance-chart`)**: Grouped Chart.js bar chart comparing Champion vs. Challenger models across Accuracy, Precision, Recall, F1, and ROC-AUC from the latest evaluation.
4. **Monitored Cities Drift & Configured Threshold Matrix (`#city-drift-table`)**:
   - Real-time matrix of all 6 cities showing City Name, Zone ID, Current PSI, Configured Threshold (`0.20`), Drift Status (`NOMINAL` vs `DRIFTED`), and Retraining State.
5. **Model Evaluation History Table (`#eval-history-table`)**:
   - Full ledger of automated CT evaluation runs showing Timestamp, Champion Version, Challenger Version, F1 Delta (e.g. `+2.45%`), Promotion Decision (`PROMOTED` vs `REJECTED`), and Gating Rationale.

---

### 5. Multi-City Drift Detection & Promotion Gate Logic
- **Single-City Drift**: Localized micro-climate shifts are logged and tracked without prematurely triggering full pipeline retraining.
- **Multi-City Threshold**: When **$\ge 4$ monitored cities** exceed `PSI > 0.20`, a `multi_city_drift_alert` is generated and published to Kafka and WebSockets.
- **Debounced CT Dispatch**: Drift monitor acquires distributed Redis lock `retrain_in_progress` (TTL: 300s) and triggers Airflow DAG `automated_retrain_dag`.
- **Promotion Safety**: Challenger model is promoted to `"Production"` alias in MLflow and hot-reloaded **only if** `challenger_f1 - champion_f1 >= 0.01` (+1.00% improvement). If the challenger fails to improve performance, the Champion is retained.

---

### 6. MLOps Audit & Event Stream (`#section-audit-log`)
Real-time chronological ledger capturing telemetry and orchestration events:
- `HAZARD`: Severe convective hazard detected (&ge; 70%).
- `DRIFT`: Spatial distribution shift detected in a specific zone.
- `RETRAIN`: Automated Airflow DAG triggered via REST API.
- `RELOAD`: Zero-downtime hot-reload completed without dropped requests.

---

## Quickstart & Runbook

### Recommended One-Command Automation via Makefile

You can control the entire platform using the [Makefile](file:///Users/abhi/Documents/mini-project/mlops-weather-early-warning/Makefile):

```bash
cd mlops-weather-early-warning

# 1. Complete one-command first-time setup (prepare -> docker up -> register model):
make init

# 2. Launch the GIS Dashboard in your browser:
make open

# 3. Stream normal baseline weather telemetry:
make demo-normal

# 4. Inject severe monsoon drift (triggers drift alert, Airflow retrain & v2 hot-reload):
make demo-drift

# 5. Check infrastructure health and scoring metrics:
make status

# 6. Run the complete 60-test test suite across all 8 phases:
make test

# 7. Stop infrastructure when finished:
make down
```

---

### Step-by-Step Manual Runbook

If you prefer executing the individual Python commands directly without `make`:

```bash
# 1. Set Up Virtual Environment & Dependencies
source ../.venv/bin/activate
pip install -r requirements.txt

# 2. Prepare Data (Phase 1)
python -m ml_core.preprocessing.data_fetcher
python -m ml_core.preprocessing.cleaner

# 3. Train Model v1 & Profile Baselines (Phase 2)
python -m ml_core.training.train_baseline

# 4. Launch Containerized Infrastructure (Phase 3)
cd infrastructure && docker compose up -d && cd ..

# 5. Register Baseline Model into MLflow Registry (Phase 3)
python infrastructure/scripts/register_baseline_mlflow.py

# 6. Stream Simulated Weather Telemetry (Phase 4)
python services/ingestion/telemetry_producer.py --rate 10.0 --mode normal --max-events 100

# 7. Inject Monsoon Drift (Phases 6 & 7)
python services/ingestion/telemetry_producer.py --rate 20.0 --mode monsoon --drift-multiplier 1.5 --max-events 200

# 8. Run Full Test Suite (60/60 Tests Passing)
pytest tests/ -v
```

---

## 5-Minute Live Evaluation Demo Script

Follow this step-by-step walkthrough to demonstrate the closed-loop MLOps pipeline live to evaluators:

### Step 1: Launch Infrastructure Stack
Start all containerized services:
```bash
make up
```
*Expected Output:* `mlops-kafka`, `mlops-redis`, `mlops-mlflow`, `mlops-airflow`, `mlops-inference-service`, and `mlops-drift-monitor` in running/healthy state.

---

### Step 2: Open the GIS Monitoring Dashboard
```bash
make open
```
Or open your browser to **[http://localhost:8000/](http://localhost:8000/)**.
*Visual Checkpoints:*
1. **Header Bar:** Shows `WebSocket: Live` (green pulsing badge), `Model: v1` (purple badge), `Drift: Nominal` (green badge), and latency metrics.
2. **Left Panel (GIS Map):** Renders Karnataka and Western Ghats map with 6 labeled spatial zone boundary polygons.
3. **Right Top Card (Drift Monitor):** Shows rolling PSI chart initialized with flat dashed reference line at `PSI = 0.20`.
4. **Right Bottom Card (Audit Log):** Logs `[SYSTEM] Connected to real-time telemetry stream`.

---

### Step 3: Stream Real-Time Telemetry (Normal Conditions)
Stream normal telemetry:
```bash
make demo-normal
```
*Visual Checkpoints on Dashboard:*
- Color-coded hazard pins appear on the GIS map at sensor coordinates (Green circle markers for low convective risk `< 0.4`).
- Clicking any sensor marker reveals an interactive popup with `zone_id`, `(lat, lon)`, `hazard_probability`, and active `model_version: v1`.
- Total Events counter increments smoothly in the header bar; latency stays strictly `< 15ms`.
- PSI values on the drift chart remain well below the dashed red line (`PSI < 0.10`).

---

### Step 4: Inject Seasonal Monsoon Drift
Run the telemetry producer in monsoon drift mode:
```bash
make demo-drift
```
*Visual Checkpoints on Dashboard:*
1. **Hazard Pin Color Shift:** High-risk Amber (`0.4-0.7`) and Red (`>= 0.7`) circular pins populate affected Western Ghats zones (e.g. `ZONE_KODAGU`, `ZONE_SHIMOGA_AGUMBE`).
2. **PSI Threshold Breach:** The rolling PSI line on the drift chart climbs steeply, crossing the horizontal dashed red reference line (`PSI > 0.20`).
3. **Spatial Zone Highlight:** The drifted spatial polygon border immediately turns into a **pulsing, dashed bright red outline** indicating active spatial drift.
4. **Header Status:** Overall Drift status badge changes to Amber/Red `Drift Detected`.
5. **Audit Log:** An orange warning appears: `[DRIFT_ALERT] Spatial drift detected in ZONE_... (PSI: 0.284 > 0.20)`.

---

### Step 5: Observe Closed-Loop Retraining & Zero-Downtime Hot-Reload
Within 15-30 seconds of drift detection:
1. **Airflow CT Trigger:** The background drift monitor acquires the Redis debounce lock and dispatches a trigger to Airflow's REST API.
2. **Audit Log Notification:** `[RETRAIN] Automated Airflow retraining DAG triggered for ZONE_...`
3. **Champion-Challenger Evaluation:** Airflow blends drifted buffer data with baseline training data, trains Model v2, passes the validation gate (+1% F1 improvement), promotes Version 2 in MLflow Model Registry to `"Production"`, and calls `POST /reload-model`.
4. **Zero-Downtime Hot-Reload Broadcast:**
   - Active Model badge in the dashboard header instantly flips to bright green: **`Model: v2`**.
   - Audit Log posts a high-priority green entry: `[MODEL_RELOAD] Model upgraded from v1 to v2 without service interruption`.
   - New sensor pins on the map reflect `model_version: v2` with zero dropped requests.

---

## Phase Completion Status

- [x] **Phase 1: Project Scaffolding & Geospatial Encoding** (8/8 tests passing)
- [x] **Phase 2: Offline Baseline Modeling & Reference Distribution Profiling** (6/6 tests passing)
- [x] **Phase 3: Containerized Infrastructure & Registry Setup** (5/5 integration tests passing)
  - [x] Lightweight Docker Compose stack in single-node KRaft mode (no Zookeeper).
  - [x] Custom Airflow image with Scikit-Learn, XGBoost, MLflow, and Redis installed.
  - [x] Automated Kafka topic initialization (`raw-telemetry`, `predictions-log`, `drift-alerts`).
  - [x] Phase 2 Model v1 registered in MLflow Model Registry under `Production` alias.
  - [x] Verified round-trip Kafka produce/consume, Redis sliding windows, and Airflow health.
- [x] **Phase 4: Real-Time Telemetry Ingestion & Spatial-Temporal Feature Buffering** (14/14 integration tests passing)
  - [x] Shared Pydantic v2 schemas: `TelemetryEvent`, `PredictionEvent`, `DriftAlertEvent`.
  - [x] Bounded `SpatialTemporalBuffer` in Redis with `LPUSH` + `LTRIM` atomic sliding window cap enforcement.
  - [x] CLI-configurable Kafka telemetry producer (`services/ingestion/telemetry_producer.py`) with `--rate`, `--mode`, `--drift-multiplier`, and `--max-events`.
  - [x] Verified schema coordinate/feature validation, Redis bounded memory guarantees, and end-to-end Kafka ingestion round-trip.
- [x] **Phase 5: Real-Time Predictive Inference & Hot-Reload Service** (4/4 integration tests passing)
  - [x] `ModelManager` with `threading.RLock()` protection for concurrent scoring and atomic MLflow hot-reload.
  - [x] Sub-50ms inference latency (< 5ms observed) and strict feature ordering enforcement.
  - [x] FastAPI service with background Kafka consumer subscribing to `raw-telemetry`, pushing to Redis, and emitting to `predictions-log`.
  - [x] REST endpoints: `/health`, `/metrics/summary`, `/predict`, and `/reload-model`.
  - [x] `infrastructure/Dockerfile.service` and updated `docker-compose.yml` with `inference-service` container.
- [x] **Phase 6: Spatial-Temporal Drift Detection Engine & Retrain Trigger** (10/10 tests passing)
  - [x] `SpatialTemporalDriftDetector` computing Population Stability Index (PSI) with outer-edge bin clamping and two-sample Kolmogorov-Smirnov (`scipy.stats.ks_2samp`) test.
  - [x] Hot-reload of `data/reference/baseline_stats.json` tracking file `mtime` changes upon model promotion.
  - [x] Scheduled `DriftMonitorWorker` polling active spatial micro-climate windows from Redis.
  - [x] Continuous publishing of `DriftAlertEvent` telemetry to Kafka `drift-alerts` topic.
  - [x] Distributed Redis debounce lock (`SET retrain_in_progress 1 NX EX 300`) preventing overlapping retraining runs across zones and polling ticks.
  - [x] Automated trigger dispatch to Airflow REST API (`POST /api/v1/dags/automated_retrain_dag/dagRuns`) with error-recovery lock release.
  - [x] Docker Compose updated with `drift-monitor` container mounting shared reference distributions.
- [x] **Phase 7: Automated Retraining, Champion-Challenger Evaluation & Zero-Downtime Deployment** (7/7 tests passing)
  - [x] Modular retraining pipeline (`ml_core/training/retrain_pipeline.py`) blending recent labeled Redis buffer telemetry with stratified historical baseline to prevent catastrophic forgetting.
  - [x] Automated Champion vs. Challenger evaluation gate (`ml_core/evaluation/comparator.py`) computing F1, Precision, Recall, and ROC-AUC on holdout validation data with configurable improvement threshold (+1% F1) and degraded recovery gate.
  - [x] Production 4-task Airflow DAG (`orchestration/airflow/dags/automated_retrain_dag.py`) triggered via REST API: `extract_drifted_window` -> `train_challenger_model` -> `evaluate_and_gate` -> `promote_and_hot_reload`.
  - [x] Automated promotion in MLflow Model Registry via `"Production"` alias.
  - [x] Atomic update of `data/reference/baseline_stats.json` with temporary file swapping (`os.replace`).
  - [x] Zero-downtime hot-reload dispatch (`POST http://inference-service:8000/reload-model`) updating `/health` without service restarts.
  - [x] Reliable Redis debounce lock (`retrain_in_progress`) release under `all_done` trigger rule.
- [x] **Phase 8: GIS Monitoring Dashboard, Evaluation History & Real-Time Alert Lifecycle** (8/8 tests passing, **62/62 full test suite passing**)
  - [x] Async `WebSocketConnectionManager` in `services/inference/app.py` broadcasting typed message envelopes (`prediction`, `drift_alert`, `multi_city_drift_alert`, `retrain_outcome`, `model_reload`) without blocking Kafka consumer threads.
  - [x] Dynamic endpoints `GET /api/zones`, `GET /api/drift/events`, and `POST /api/retrain/outcome`.
  - [x] Dark-mode Command Dashboard (`frontend/index.html` & `frontend/css/styles.css`) using Leaflet.js and Chart.js via CDN (zero npm/build tools).
  - [x] Fixed left sidebar navigation with scrollspy synchronization across Overview, Current Model, Model Performances, and Audit Stream.
  - [x] Separate **Current Model** section with model version, MLflow registry alias, deployment timestamps, hazard severity table, and GIS map.
  - [x] Separate **Model Performances** section with active metrics, rolling PSI drift chart, Champion vs. Challenger comparison chart, city threshold matrix, and evaluation ledger.
  - [x] **Significant Multi-City Drift Event History Table:** Persistent chronological ledger tracking trigger timestamps, affected cities, PSI values at trigger, model at trigger, challenger version, Champion–Challenger decision, and final deployed model & deployment timestamps.
  - [x] **Automatic 60-Second Alert Reset:** Post-retraining 60-second visual countdown timer returning temporary red alert banners and status badges to nominal without clearing persistent drift data, retraining records, or evaluation history. Handles concurrent and subsequent events independently.
  - [x] Interactive Leaflet map controller (`frontend/js/map_controller.js`) with color-coded hazard markers (`<0.4` green, `0.4-0.7` amber, `>=0.7` red) and red dashed borders for drifted zones.
  - [x] Rolling 30-point Chart.js line controller (`frontend/js/chart_controller.js`) with static dashed reference line (`PSI = 0.20`).
  - [x] Native WebSocket client (`frontend/js/ws_client.js`) with exponential backoff auto-reconnect, REST state hydration, and MLOps audit log.
  - [x] Production containerization (`infrastructure/Dockerfile.service` & `docker-compose.yml`) with mounted `frontend/` static assets.
  - [x] Automated integration test suite (`tests/integration/test_phase8_e2e.py`) validating HTML delivery, zone config API, WebSocket ping-pong, real-time broadcasts, drift event history endpoint, and approval/rejection outcome webhooks.
  - [x] Concise **5-Minute Live Evaluation Demo Script** walking evaluators through normal inference -> spatial drift detection -> automated Airflow retraining -> Champion-Challenger gating -> zero-downtime model hot-reload -> 60-second visual alert reset.
