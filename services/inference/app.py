"""FastAPI Real-Time Inference & Streaming Hazard Prediction Service.

Subscribes to incoming 'raw-telemetry' stream via Kafka consumer group, maintains
zone-partitioned sliding windows in Redis, executes model inference via thread-safe
ModelManager, and emits hazard predictions to 'predictions-log'.
Provides REST endpoints for health checks, live scoring, telemetry summaries, hot-reload,
spatial zones metadata, static dashboard hosting, and a high-performance WebSocket broadcaster.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from kafka import KafkaConsumer, KafkaProducer
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.common.redis_buffer import SpatialTemporalBuffer
from services.common.schemas import PredictionEvent, TelemetryEvent
from services.inference.model_loader import ModelManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("inference_service")


def _load_app_settings() -> Dict[str, Any]:
    """Load settings from config/settings.yaml if available."""
    config_path = PROJECT_ROOT / "config" / "settings.yaml"
    if config_path.is_file():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning("Could not read settings from %s: %s", config_path, e)
    return {}


class InferenceMetricsTracker:
    """Thread-safe rolling metrics tracker for inference requests and zone risks."""

    def __init__(self):
        self._lock = threading.Lock()
        self.total_predictions = 0
        self.total_latency_ms = 0.0
        self.per_zone_status: Dict[str, Dict[str, Any]] = {}

    def record_prediction(
        self,
        zone_id: str,
        predicted_label: int,
        hazard_probability: float,
        latency_ms: float,
        timestamp: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ):
        """Record an inference event."""
        with self._lock:
            self.total_predictions += 1
            self.total_latency_ms += latency_ms
            self.per_zone_status[zone_id] = {
                "latest_timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
                "hazard_probability": round(hazard_probability, 4),
                "predicted_label": predicted_label,
                "lat": lat,
                "lon": lon,
                "last_latency_ms": round(latency_ms, 2),
            }

    def get_summary(self) -> Dict[str, Any]:
        """Return snapshot summary of metrics."""
        with self._lock:
            count = self.total_predictions
            avg_latency = (self.total_latency_ms / count) if count > 0 else 0.0
            return {
                "total_predictions": count,
                "avg_latency_ms": round(avg_latency, 3),
                "per_zone_risk_status": dict(self.per_zone_status),
            }


class WebSocketConnectionManager:
    """Manages active WebSocket browser clients and bridges synchronous events into asyncio."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = threading.Lock()

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        with self._lock:
            self.active_connections.append(websocket)
        logger.info("WebSocket client connected. Active clients: %d", len(self.active_connections))

    def disconnect(self, websocket: WebSocket):
        with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info("WebSocket client disconnected. Active clients: %d", len(self.active_connections))

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast payload to all connected WebSocket clients."""
        payload_text = json.dumps(message)
        dead_conns = []
        with self._lock:
            targets = list(self.active_connections)

        for connection in targets:
            try:
                await connection.send_text(payload_text)
            except Exception:
                dead_conns.append(connection)

        if dead_conns:
            with self._lock:
                for dead in dead_conns:
                    if dead in self.active_connections:
                        self.active_connections.remove(dead)

    def broadcast_sync(self, message: Dict[str, Any]):
        """Bridge call from synchronous threads into the FastAPI asyncio event loop."""
        if self.loop is not None and self.loop.is_running() and self.active_connections:
            try:
                asyncio.run_coroutine_threadsafe(self.broadcast(message), self.loop)
            except Exception as exc:
                logger.debug("Failed to dispatch async broadcast: %s", exc)


# Service Singletons
settings = _load_app_settings()
kafka_cfg = settings.get("streaming", {}).get("kafka", {})
topics_cfg = kafka_cfg.get("topics", {})

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS") or kafka_cfg.get(
    "bootstrap_servers", "localhost:29092"
)
RAW_TOPIC = topics_cfg.get("raw_telemetry", "raw-telemetry")
PRED_TOPIC = topics_cfg.get("predictions_log", "predictions-log")
DRIFT_TOPIC = topics_cfg.get("drift_alerts", "drift-alerts")

model_manager = ModelManager()
redis_buffer = SpatialTemporalBuffer()
metrics_tracker = InferenceMetricsTracker()
ws_manager = WebSocketConnectionManager()

kafka_producer: Optional[KafkaProducer] = None
consumer_thread: Optional[threading.Thread] = None
drift_consumer_thread: Optional[threading.Thread] = None
stop_event = threading.Event()
consumer_running = False
consumer_ready = threading.Event()


def _get_kafka_producer() -> Optional[KafkaProducer]:
    """Initialize or return KafkaProducer."""
    global kafka_producer
    if kafka_producer is None:
        try:
            kafka_producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if isinstance(k, str) else k,
                acks=1,
                retries=2,
                api_version=(3, 7, 0),
            )
        except Exception as e:
            logger.warning("Could not initialize KafkaProducer for predictions (%s): %s", KAFKA_BOOTSTRAP, e)
    return kafka_producer


def run_kafka_consumer_loop(
    stop_sig: threading.Event,
    bootstrap_servers: str,
    topic: str,
    pred_topic: str,
):
    """Background consumer loop reading raw telemetry, buffering, and scoring."""
    global consumer_running, consumer_ready
    logger.info("Starting background Kafka consumer on topic '%s'...", topic)
    consumer = None
    producer = None

    group_id = os.environ.get("KAFKA_GROUP_ID", "inference-service-group")

    while not stop_sig.is_set():
        try:
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=bootstrap_servers,
                group_id=group_id,
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                session_timeout_ms=10000,
                heartbeat_interval_ms=3000,
                consumer_timeout_ms=1000,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                key_deserializer=lambda k: k.decode("utf-8") if k else None,
                api_version=(3, 7, 0),
            )
            consumer_running = True
            logger.info("KafkaConsumer connected to %s (group=%s). Awaiting telemetry...", bootstrap_servers, group_id)
            break
        except Exception as e:
            logger.warning("Kafka consumer connection attempt failed: %s. Retrying in 1s...", e)
            time.sleep(1.0)

    try:
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if isinstance(k, str) else k,
            acks=1,
            retries=2,
            api_version=(3, 7, 0),
        )
    except Exception as exc:
        logger.error("Could not initialize KafkaProducer for prediction events: %s", exc)

    try:
        while not stop_sig.is_set() and consumer is not None:
            raw_batches = consumer.poll(timeout_ms=500)
            if not consumer_ready.is_set() and consumer.assignment():
                consumer_ready.set()
                logger.info("Kafka consumer ready with partition assignment: %s", consumer.assignment())

            if not raw_batches:
                continue

            for tp, records in raw_batches.items():
                for record in records:
                    if stop_sig.is_set():
                        break
                    try:
                        # 1. Deserialize to TelemetryEvent
                        telemetry_event = TelemetryEvent.model_validate(record.value)

                        # 2. Buffer in Redis bounded sliding window
                        redis_buffer.push_event(telemetry_event)

                        # 3. Model Inference with latency measurement
                        t0 = time.perf_counter()
                        pred_label, hazard_prob = model_manager.predict(telemetry_event.features)
                        latency_ms = (time.perf_counter() - t0) * 1000.0

                        # 4. Construct PredictionEvent
                        pred_event = PredictionEvent(
                            event_id=telemetry_event.event_id,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            lat=telemetry_event.lat,
                            lon=telemetry_event.lon,
                            zone_id=telemetry_event.zone_id,
                            predicted_label=pred_label,
                            hazard_probability=hazard_prob,
                            model_version=model_manager.model_version,
                            inference_latency_ms=round(latency_ms, 3),
                        )

                        # 5. Emit to predictions-log Kafka topic
                        if producer is not None:
                            producer.send(
                                topic=pred_topic,
                                key=pred_event.zone_id,
                                value=pred_event.to_dict(),
                            )

                        # 6. Track rolling telemetry metrics
                        metrics_tracker.record_prediction(
                            zone_id=pred_event.zone_id,
                            predicted_label=pred_label,
                            hazard_probability=hazard_prob,
                            latency_ms=latency_ms,
                            timestamp=pred_event.timestamp,
                            lat=pred_event.lat,
                            lon=pred_event.lon,
                        )

                        # 7. WebSocket Live Broadcast: Envelope 1 (prediction)
                        ws_manager.broadcast_sync({
                            "type": "prediction",
                            "data": pred_event.to_dict(),
                        })

                    except Exception as err:
                        logger.error("Error processing streaming record: %s", err)

            # Flush producer after processing batch
            if producer is not None:
                producer.flush(timeout=2)

    except Exception as exc:
        logger.error("Consumer loop exception: %s", exc)
    finally:
        consumer_running = False
        consumer_ready.clear()
        if consumer is not None:
            try:
                consumer.close(timeout=3)
            except Exception:
                pass
        if producer is not None:
            try:
                producer.flush(timeout=3)
                producer.close(timeout=2)
            except Exception:
                pass
        logger.info("Background Kafka consumer loop terminated.")


def run_drift_alerts_consumer_loop(
    stop_sig: threading.Event,
    bootstrap_servers: str,
    alerts_topic: str,
):
    """Background listener consuming drift alerts from Kafka and forwarding to WebSockets."""
    logger.info("Starting background drift-alerts listener on topic '%s'...", alerts_topic)
    group_id = os.environ.get("KAFKA_DRIFT_GROUP_ID", "dashboard-drift-group")
    consumer = None

    while not stop_sig.is_set():
        try:
            consumer = KafkaConsumer(
                alerts_topic,
                bootstrap_servers=bootstrap_servers,
                group_id=group_id,
                auto_offset_reset="latest",
                enable_auto_commit=True,
                session_timeout_ms=10000,
                consumer_timeout_ms=1000,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                key_deserializer=lambda k: k.decode("utf-8") if k else None,
                api_version=(3, 7, 0),
            )
            logger.info("Connected to drift-alerts topic on %s (group=%s)", bootstrap_servers, group_id)
            break
        except Exception as exc:
            logger.debug("Drift listener waiting for Kafka (%s): %s", bootstrap_servers, exc)
            time.sleep(2.0)

    try:
        while not stop_sig.is_set() and consumer is not None:
            batch = consumer.poll(timeout_ms=500)
            if not batch:
                continue
            for tp, records in batch.items():
                for record in records:
                    if stop_sig.is_set():
                        break
                    alert_data = record.value
                    if isinstance(alert_data, dict) and alert_data.get("type") == "multi_city_drift_alert":
                        # Multi-City Significant Drift Alert Envelope
                        ws_manager.broadcast_sync({
                            "type": "multi_city_drift_alert",
                            "data": alert_data,
                        })
                    else:
                        # WebSocket Live Broadcast: Envelope 2 (drift_alert)
                        ws_manager.broadcast_sync({
                            "type": "drift_alert",
                            "data": alert_data,
                        })
    except Exception as exc:
        logger.error("Drift alerts consumer loop error: %s", exc)
    finally:
        if consumer is not None:
            try:
                consumer.close(timeout=2)
            except Exception:
                pass
        logger.info("Background drift-alerts consumer loop terminated.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager controlling startup and shutdown procedures."""
    global consumer_thread, drift_consumer_thread, stop_event

    logger.info("Initializing Inference Service...")
    # Initialize WebSocket loop bridge
    loop = asyncio.get_running_loop()
    ws_manager.set_loop(loop)

    # Load Production Model
    model_version = model_manager.load_production_model()
    logger.info("Active model loaded: version=%s", model_version)

    # Initialize Redis Buffer & Ping
    redis_alive = redis_buffer.ping()
    logger.info("Redis buffer status: connected=%s", redis_alive)

    # Start Background Consumer Threads
    stop_event.clear()
    consumer_ready.clear()

    consumer_thread = threading.Thread(
        target=run_kafka_consumer_loop,
        args=(stop_event, KAFKA_BOOTSTRAP, RAW_TOPIC, PRED_TOPIC),
        name="KafkaInferenceConsumer",
        daemon=True,
    )
    consumer_thread.start()

    drift_consumer_thread = threading.Thread(
        target=run_drift_alerts_consumer_loop,
        args=(stop_event, KAFKA_BOOTSTRAP, DRIFT_TOPIC),
        name="KafkaDriftConsumer",
        daemon=True,
    )
    drift_consumer_thread.start()

    yield

    # Shutdown procedure
    logger.info("Shutting down Inference Service...")
    stop_event.set()
    if consumer_thread and consumer_thread.is_alive():
        consumer_thread.join(timeout=3.0)
    if drift_consumer_thread and drift_consumer_thread.is_alive():
        drift_consumer_thread.join(timeout=3.0)

    logger.info("Inference Service shutdown complete.")


# Initialize FastAPI application
app = FastAPI(
    title="MLOps Real-Time Weather Hazard Inference & GIS Dashboard",
    description="Live scoring, Kafka streaming, Redis spatial-temporal buffering, WebSocket broadcaster, and GIS command center",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount static frontend directories if present
FRONTEND_DIR = PROJECT_ROOT / "frontend"
if (FRONTEND_DIR / "css").is_dir():
    app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
if (FRONTEND_DIR / "js").is_dir():
    app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")


@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """Serve the Command GIS Dashboard single-page application."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.is_file():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse(
        "<html><body><h1>MLOps Weather GIS Dashboard</h1><p>frontend/index.html not found</p></body></html>"
    )


@app.get("/api/zones", status_code=status.HTTP_200_OK)
def get_spatial_zones():
    """Expose spatial zone coordinates and drift configuration for GIS dashboard."""
    zones_file = PROJECT_ROOT / "config" / "spatial_zones.json"
    zones_data = []
    if zones_file.is_file():
        try:
            with open(zones_file, "r", encoding="utf-8") as f:
                parsed = json.load(f)
                zones_data = parsed.get("zones", [])
        except Exception as e:
            logger.error("Error reading spatial zones JSON: %s", e)

    cfg = _load_app_settings()
    drift_cfg = cfg.get("drift_thresholds", {})
    psi_threshold = drift_cfg.get("psi_threshold", 0.20)
    multi_city_threshold = drift_cfg.get("multi_city_drift_threshold", 4)
    features = cfg.get("features", {}).get(
        "numeric_features",
        ["rainfall_mm", "soil_moisture", "temperature_c", "humidity_pct", "wind_speed_kmh"],
    )

    return {
        "zones": zones_data,
        "psi_threshold": psi_threshold,
        "multi_city_drift_threshold": multi_city_threshold,
        "features": features,
    }


@app.get("/api/model/performance", status_code=status.HTTP_200_OK)
def get_model_performance():
    """Expose active model metadata, actual evaluation metrics, drift thresholds, and comparison history."""
    cfg = _load_app_settings()
    drift_cfg = cfg.get("drift_thresholds", {})
    psi_threshold = float(drift_cfg.get("psi_threshold", 0.20))
    multi_city_threshold = int(drift_cfg.get("multi_city_drift_threshold", 4))

    # Load actual evaluation history
    eval_history = []
    history_file = PROJECT_ROOT / "data" / "reference" / "evaluation_history.json"
    if history_file.is_file():
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                eval_history = json.load(f)
        except Exception as e:
            logger.error("Error reading evaluation history: %s", e)

    # Load persistent multi-city drift event history
    drift_history = []
    drift_file = PROJECT_ROOT / "data" / "reference" / "drift_event_history.json"
    if drift_file.is_file():
        try:
            with open(drift_file, "r", encoding="utf-8") as f:
                drift_history = json.load(f)
        except Exception as e:
            logger.error("Error reading drift event history: %s", e)

    metadata = model_manager.get_metadata()
    metrics = model_manager.get_metrics()
    summary = metrics_tracker.get_summary()

    return {
        "active_model_version": model_manager.model_version,
        "model_metadata": metadata,
        "active_metrics": metrics,
        "psi_threshold": psi_threshold,
        "multi_city_drift_threshold": multi_city_threshold,
        "latest_zone_status": summary.get("per_zone_risk_status", {}),
        "evaluation_history": eval_history,
        "drift_event_history": drift_history,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/drift/events", status_code=status.HTTP_200_OK)
def get_drift_events():
    """Expose persistent multi-city drift event history."""
    drift_file = PROJECT_ROOT / "data" / "reference" / "drift_event_history.json"
    if drift_file.is_file():
        try:
            with open(drift_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Error reading drift events: %s", e)
    return []


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """Real-time streaming WebSocket endpoint for GIS dashboard live telemetry."""
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(
                    json.dumps({
                        "type": "pong",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                )
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)


@app.get("/health", status_code=status.HTTP_200_OK)
def health():
    """Health check endpoint exposing service components and active model version."""
    return {
        "status": "healthy",
        "service": "inference-service",
        "model_version": model_manager.model_version,
        "model_metadata": model_manager.get_metadata(),
        "kafka_consumer": "running" if consumer_running else "idle",
        "redis_connected": redis_buffer.ping(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/metrics/summary", status_code=status.HTTP_200_OK)
def metrics_summary():
    """Summary of total predictions served, average latency, and per-zone risk states."""
    summary = metrics_tracker.get_summary()
    return {
        "service": "inference-service",
        "active_model_version": model_manager.model_version,
        "total_predictions": summary["total_predictions"],
        "avg_latency_ms": summary["avg_latency_ms"],
        "latest_zone_status": summary["per_zone_risk_status"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/predict", response_model=PredictionEvent, status_code=status.HTTP_200_OK)
def predict(event: TelemetryEvent) -> PredictionEvent:
    """Synchronous single-event scoring endpoint for latency benchmarking and direct API queries."""
    t0 = time.perf_counter()
    pred_label, hazard_prob = model_manager.predict(event.features)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    pred_event = PredictionEvent(
        event_id=event.event_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        lat=event.lat,
        lon=event.lon,
        zone_id=event.zone_id,
        predicted_label=pred_label,
        hazard_probability=round(hazard_prob, 4),
        model_version=model_manager.model_version,
        inference_latency_ms=round(latency_ms, 3),
    )

    # Record metrics
    metrics_tracker.record_prediction(
        zone_id=event.zone_id,
        predicted_label=pred_label,
        hazard_probability=hazard_prob,
        latency_ms=latency_ms,
        timestamp=pred_event.timestamp,
        lat=event.lat,
        lon=event.lon,
    )

    # Broadcast to WebSockets
    ws_manager.broadcast_sync({
        "type": "prediction",
        "data": pred_event.to_dict(),
    })

    return pred_event


@app.post("/reload-model", status_code=status.HTTP_200_OK)
def reload_model():
    """Trigger atomic hot-reload of active model from MLflow Registry without dropping requests."""
    prev_version = model_manager.model_version
    try:
        reload_result = model_manager.reload()
        new_version = model_manager.model_version

        # WebSocket Live Broadcast: Envelope 3 (model_reload)
        ws_manager.broadcast_sync({
            "type": "model_reload",
            "data": {
                "previous_version": prev_version,
                "active_version": new_version,
                "reloaded_at": datetime.now(timezone.utc).isoformat(),
                "source": reload_result.get("source"),
            },
        })

        return reload_result
    except Exception as exc:
        logger.error("Hot-reload failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hot-reload failed: {exc}",
        )


@app.post("/api/retrain/outcome", status_code=status.HTTP_200_OK)
def record_retrain_outcome(payload: Dict[str, Any]):
    """Receive Champion-Challenger evaluation & deployment outcome from Airflow, broadcast over WebSocket."""
    decision = payload.get("decision", "rejected")
    promoted = payload.get("promoted", False)
    final_model = payload.get("final_production_model_version", "v1")
    event_id = payload.get("event_id")

    logger.info(
        "Received Champion-Challenger retraining outcome: event=%s, decision=%s, promoted=%s, final_model=%s",
        event_id,
        decision,
        promoted,
        final_model,
    )

    # If promoted and model is not yet hot-reloaded, trigger hot-reload
    if promoted and model_manager.model_version != final_model:
        try:
            model_manager.reload()
        except Exception as e:
            logger.warning("Could not auto-reload model inside outcome handler: %s", e)

    # Broadcast retrain_outcome event to all connected dashboard WebSocket clients
    ws_manager.broadcast_sync({
        "type": "retrain_outcome",
        "data": payload,
    })

    return {
        "status": "success",
        "decision": decision,
        "final_production_model_version": final_model,
        "event_id": event_id,
    }
