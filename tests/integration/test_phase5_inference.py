"""Integration Tests for Phase 5: Real-Time Predictive Inference & Hot-Reload Service.

Verifies:
1. GET /health returns HTTP 200 with active model_version and component states.
2. POST /predict scores TelemetryEvent with strict schema validation in < 50ms.
3. End-to-end streaming roundtrip: 20 TelemetryEvent records published to 'raw-telemetry'
   result in 20 PredictionEvent records on 'predictions-log' and buffered in Redis window:{zone_id}.
4. Concurrent hot-reload: POST /reload-model during concurrent POST /predict traffic
   succeeds with HTTP 200 and zero failed requests under threading.RLock protection.
"""

import concurrent.futures
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Dict, List
import uuid
from fastapi.testclient import TestClient
from kafka import KafkaConsumer, KafkaProducer, TopicPartition
import pytest
import redis

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.common.redis_buffer import SpatialTemporalBuffer
from services.common.schemas import PredictionEvent, TelemetryEvent
from services.inference.app import app, consumer_ready
from services.inference.model_loader import ModelManager


@pytest.fixture(scope="module")
def sample_telemetry_payload() -> Dict:
    """Standard valid TelemetryEvent dictionary fixture."""
    return {
        "lat": 12.25,
        "lon": 75.82,
        "zone_id": "ZONE_KODAGU",
        "features": {
            "rainfall_mm": 48.5,
            "soil_moisture": 0.82,
            "temperature_c": 21.0,
            "humidity_pct": 97.0,
            "wind_speed_kmh": 26.0,
        },
        "delayed_label": 1,
    }


@pytest.fixture(scope="module")
def redis_client():
    """Redis client connected to local running container."""
    client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    try:
        client.ping()
    except Exception as exc:
        pytest.skip(f"Redis not reachable on localhost:6379: {exc}")
    yield client
    client.close()


@pytest.fixture(scope="module")
def kafka_broker():
    """Verify Kafka broker is reachable before running streaming integration tests."""
    bootstrap_server = "localhost:29092"
    try:
        consumer = KafkaConsumer(
            bootstrap_servers=bootstrap_server,
            consumer_timeout_ms=3000,
            api_version=(3, 7, 0),
        )
        partitions = consumer.partitions_for_topic("raw-telemetry")
        consumer.close()
        if partitions is None:
            pytest.skip("Kafka topic 'raw-telemetry' not found or broker offline")
    except Exception as exc:
        pytest.skip(f"Kafka broker offline on {bootstrap_server}: {exc}")
    return bootstrap_server


# =========================================================================
# 1. Health & Status Tests
# =========================================================================

def test_health_endpoint():
    """Verify GET /health returns HTTP 200 with active model_version and component states."""
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "inference-service"
        assert "model_version" in data and len(data["model_version"]) > 0
        assert data["redis_connected"] is True

        metadata = data.get("model_metadata", {})
        assert metadata.get("is_loaded") is True
        assert "feature_cols" in metadata
        assert "rainfall_mm" in metadata["feature_cols"]


# =========================================================================
# 2. Predictive Inference Latency & Contract Tests (< 50ms requirement)
# =========================================================================

def test_predict_endpoint_latency_and_schema(sample_telemetry_payload):
    """Verify POST /predict scores TelemetryEvent in < 50 ms and returns valid PredictionEvent."""
    with TestClient(app) as client:
        # Warmup call
        warmup_resp = client.post("/predict", json=sample_telemetry_payload)
        assert warmup_resp.status_code == 200

        # Execute timed scoring runs
        latencies_ms = []
        for _ in range(15):
            t0 = time.perf_counter()
            resp = client.post("/predict", json=sample_telemetry_payload)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)

            assert resp.status_code == 200
            data = resp.json()

            # Validate against strict Pydantic contract
            pred_event = PredictionEvent.model_validate(data)
            assert pred_event.predicted_label in {0, 1}
            assert 0.0 <= pred_event.hazard_probability <= 1.0
            assert pred_event.zone_id == sample_telemetry_payload["zone_id"]
            assert pred_event.lat == sample_telemetry_payload["lat"]
            assert pred_event.lon == sample_telemetry_payload["lon"]
            assert pred_event.inference_latency_ms < 50.0

        avg_latency = sum(latencies_ms) / len(latencies_ms)
        p95_latency = sorted(latencies_ms)[int(len(latencies_ms) * 0.95)]
        assert p95_latency < 50.0, f"p95 latency exceeded 50ms: {p95_latency:.2f}ms"
        assert avg_latency < 25.0, f"Average latency exceeded 25ms: {avg_latency:.2f}ms"

        # Verify summary metrics endpoint
        summary_resp = client.get("/metrics/summary")
        assert summary_resp.status_code == 200
        summary_data = summary_resp.json()
        assert summary_data["total_predictions"] >= 15
        assert summary_data["avg_latency_ms"] > 0.0
        assert sample_telemetry_payload["zone_id"] in summary_data["latest_zone_status"]


# =========================================================================
# 3. End-to-End Streaming Roundtrip Test (20 events)
# =========================================================================

def test_e2e_streaming_raw_telemetry_to_predictions_log_and_redis(kafka_broker, redis_client):
    """Verify 20 TelemetryEvents stream through Kafka -> App -> Redis window & predictions-log."""
    test_zone = f"ZONE_E2E_{uuid.uuid4().hex[:8]}"
    num_events = 20

    redis_buffer = SpatialTemporalBuffer(client=redis_client)
    redis_buffer.clear_zone_window(test_zone)

    # 1. Establish initial partition end offsets for predictions-log
    consumer_init = KafkaConsumer(
        bootstrap_servers=kafka_broker,
        enable_auto_commit=False,
        api_version=(3, 7, 0),
    )
    tps_pred = [TopicPartition("predictions-log", p) for p in consumer_init.partitions_for_topic("predictions-log")]
    start_offsets_pred = consumer_init.end_offsets(tps_pred)
    consumer_init.close()

    producer = KafkaProducer(
        bootstrap_servers=kafka_broker,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8"),
        api_version=(3, 7, 0),
    )

    with TestClient(app) as client:
        # Wait until background consumer has joined group and partition assignment is active
        consumer_ready.wait(timeout=10.0)
        assert consumer_ready.is_set(), "Kafka background consumer did not establish partition assignment"

        # 2. Publish 20 TelemetryEvent records to raw-telemetry
        for i in range(num_events):
            event = TelemetryEvent(
                lat=12.25,
                lon=75.82,
                zone_id=test_zone,
                features={
                    "rainfall_mm": float(10 + i * 2),
                    "soil_moisture": min(1.0, 0.4 + i * 0.02),
                    "temperature_c": 22.5,
                    "humidity_pct": 92.0,
                    "wind_speed_kmh": 16.0,
                },
                delayed_label=1 if i >= 10 else 0,
            )
            producer.send("raw-telemetry", key=test_zone, value=event.to_dict())
        producer.flush()

        # 3. Consume newly emitted records from predictions-log starting from start_offsets_pred
        pred_consumer = KafkaConsumer(
            bootstrap_servers=kafka_broker,
            enable_auto_commit=False,
            consumer_timeout_ms=10000,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            api_version=(3, 7, 0),
        )
        pred_consumer.assign(tps_pred)
        for tp in tps_pred:
            pred_consumer.seek(tp, start_offsets_pred[tp])

        received_predictions = []
        deadline = time.time() + 15.0
        while time.time() < deadline and len(received_predictions) < num_events:
            for msg in pred_consumer:
                val = msg.value
                if val.get("zone_id") == test_zone:
                    received_predictions.append(val)
                    if len(received_predictions) >= num_events:
                        break

        pred_consumer.close()

    producer.close()

    # Verify exactly 20 PredictionEvent payloads received on predictions-log
    assert len(received_predictions) == num_events, (
        f"Expected {num_events} messages on predictions-log, received {len(received_predictions)}"
    )

    for item in received_predictions:
        parsed_pred = PredictionEvent.model_validate(item)
        assert parsed_pred.zone_id == test_zone
        assert parsed_pred.predicted_label in {0, 1}
        assert 0.0 <= parsed_pred.hazard_probability <= 1.0

    # 4. Verify Redis bounded sliding window contains the 20 buffered events
    buffered_count = redis_buffer.get_window_size(test_zone)
    assert buffered_count == num_events, (
        f"Expected {num_events} buffered in Redis for {test_zone}, got {buffered_count}"
    )

    window_records = redis_buffer.get_zone_window(test_zone)
    assert len(window_records) == num_events
    # Oldest event should have rainfall_mm = 10.0, newest = 48.0
    assert window_records[0]["features"]["rainfall_mm"] == 10.0
    assert window_records[-1]["features"]["rainfall_mm"] == 48.0

    # Cleanup
    redis_buffer.clear_zone_window(test_zone)


# =========================================================================
# 4. Concurrent Hot-Reload Under Active Prediction Traffic Tests
# =========================================================================

def test_concurrent_hot_reload_under_traffic(sample_telemetry_payload):
    """Verify POST /reload-model succeeds with HTTP 200 and zero dropped requests under concurrency."""
    with TestClient(app) as client:
        num_predict_requests = 30
        failed_requests = 0

        def send_predict(idx: int) -> int:
            payload = dict(sample_telemetry_payload)
            payload["features"]["rainfall_mm"] = 10.0 + (idx % 10)
            resp = client.post("/predict", json=payload)
            return resp.status_code

        def send_reload() -> Dict:
            # Add micro-delay to interleave reload squarely inside active prediction requests
            time.sleep(0.01)
            resp = client.post("/reload-model")
            assert resp.status_code == 200, f"Reload failed: {resp.text}"
            return resp.json()

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            # Schedule predict requests and hot-reload concurrently
            f_reload = executor.submit(send_reload)
            f_preds = [executor.submit(send_predict, i) for i in range(num_predict_requests)]

            reload_body = f_reload.result()
            pred_statuses = [f.result() for f in f_preds]

        # Verify hot reload succeeded
        assert reload_body["status"] == "reloaded"
        assert "active_version" in reload_body
        assert "reloaded_at" in reload_body

        # Verify zero dropped requests during reload
        failed_count = sum(1 for s in pred_statuses if s != 200)
        assert failed_count == 0, f"Encountered {failed_count} failed predict requests during reload"
        assert len(pred_statuses) == num_predict_requests
