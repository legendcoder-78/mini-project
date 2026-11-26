"""Integration & Smoke Tests for Phase 3: Containerized Infrastructure & Registry Setup.

Verifies connectivity and operational contracts for:
1. Redis: In-memory sliding window operations (PING, LPUSH, LRANGE, DEL).
2. Apache Kafka: KRaft broker health, application topics presence, produce-consume round-trip.
3. MLflow: REST health endpoint, Model Registry verification, and Production alias assignment.
4. Apache Airflow: REST health endpoint and metadata connectivity.
"""

import json
import time
from typing import Any, Dict
import pytest
import requests
import yaml
from pathlib import Path


@pytest.fixture(scope="session")
def project_settings():
    """Load settings.yaml configuration."""
    settings_path = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yaml"
    with open(settings_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =========================================================================
# 1. Redis Integration Tests
# =========================================================================

def test_redis_connection_and_operations(project_settings):
    """Verify Redis is reachable, responds to PING, and supports sliding window operations."""
    import redis

    redis_cfg = project_settings.get("streaming", {}).get("redis", {})
    host = redis_cfg.get("host", "localhost")
    port = int(redis_cfg.get("port", 6379))
    db = int(redis_cfg.get("db", 0))

    r = redis.Redis(host=host, port=port, db=db, socket_timeout=3.0, decode_responses=True)

    # 1. Ping
    assert r.ping() is True, f"Redis at {host}:{port} did not respond to PING"

    # 2. Sliding window buffer emulation (LPUSH + LRANGE)
    test_key = "test:window:ZONE_KODAGU"
    r.delete(test_key)

    sample_record_1 = {"event_id": "e1", "rainfall_mm": 5.2, "soil_moisture": 0.45}
    sample_record_2 = {"event_id": "e2", "rainfall_mm": 18.4, "soil_moisture": 0.62}

    r.lpush(test_key, json.dumps(sample_record_1))
    r.lpush(test_key, json.dumps(sample_record_2))

    # Fetch window
    window = r.lrange(test_key, 0, -1)
    assert len(window) == 2, f"Expected 2 records in Redis list, got {len(window)}"

    parsed_latest = json.loads(window[0])
    assert parsed_latest["event_id"] == "e2", "Latest record should be at index 0 (LPUSH order)"

    # Clean up
    r.delete(test_key)
    assert r.exists(test_key) == 0


# =========================================================================
# 2. Apache Kafka Integration Tests
# =========================================================================

def test_kafka_topics_presence(project_settings):
    """Verify Kafka broker is reachable and application topics exist with valid partitions."""
    from kafka import KafkaAdminClient

    bootstrap_servers = project_settings.get("streaming", {}).get("kafka", {}).get(
        "bootstrap_servers", "localhost:29092"
    )

    admin = KafkaAdminClient(bootstrap_servers=bootstrap_servers, request_timeout_ms=8000)
    existing_topics = set(admin.list_topics())

    required_topics = {"raw-telemetry", "predictions-log", "drift-alerts"}
    assert required_topics.issubset(existing_topics), (
        f"Missing required Kafka topics. Expected {required_topics}, found: {existing_topics}"
    )

    # Inspect partition count for raw-telemetry
    topic_metadata = admin.describe_topics(["raw-telemetry"])
    assert len(topic_metadata) == 1
    assert len(topic_metadata[0]["partitions"]) >= 1, "raw-telemetry must have at least 1 partition"
    admin.close()


def test_kafka_produce_consume_roundtrip(project_settings):
    """Verify producing a JSON telemetry message to raw-telemetry and consuming it back round-trip."""
    from kafka import KafkaConsumer, KafkaProducer

    bootstrap_servers = project_settings.get("streaming", {}).get("kafka", {}).get(
        "bootstrap_servers", "localhost:29092"
    )
    topic = "raw-telemetry"
    test_event_id = f"test-ping-{int(time.time() * 1000)}"
    test_payload = {
        "event_id": test_event_id,
        "timestamp": "2023-07-15T14:00:00Z",
        "latitude": 12.25,
        "longitude": 75.82,
        "zone_id": "ZONE_KODAGU",
        "rainfall_mm": 42.5,
        "soil_moisture": 0.81,
        "temperature_c": 21.2,
        "humidity_pct": 98.0,
        "wind_speed_kmh": 35.0,
    }

    # 1. Produce message
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        request_timeout_ms=5000,
    )
    future = producer.send(topic, value=test_payload)
    record_metadata = future.get(timeout=5)
    producer.flush()
    producer.close()

    assert record_metadata.topic == topic, "Message not delivered to expected topic"

    # 2. Consume message
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        consumer_timeout_ms=6000,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )

    found = False
    for msg in consumer:
        if isinstance(msg.value, dict) and msg.value.get("event_id") == test_event_id:
            found = True
            assert msg.value["zone_id"] == "ZONE_KODAGU"
            assert msg.value["rainfall_mm"] == 42.5
            break

    consumer.close()
    assert found is True, f"Failed to retrieve published test message with event_id '{test_event_id}' from Kafka"


# =========================================================================
# 3. MLflow Tracking & Model Registry Tests
# =========================================================================

def test_mlflow_health_and_model_registry(project_settings):
    """Verify MLflow REST server is healthy and Model v1 is registered under Production alias."""
    from mlflow.tracking import MlflowClient

    mlflow_cfg = project_settings.get("orchestration", {}).get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "http://localhost:5000")
    model_name = mlflow_cfg.get("model_name", "weather-hazard-classifier")

    # 1. Check REST Health
    health_resp = requests.get(f"{tracking_uri.rstrip('/')}/health", timeout=3.0)
    assert health_resp.status_code == 200, f"MLflow health check failed: {health_resp.text}"

    # 2. Check Model Registry
    client = MlflowClient(tracking_uri=tracking_uri)
    model = client.get_registered_model(model_name)
    assert model.name == model_name, f"Expected registered model name '{model_name}', got '{model.name}'"

    # 3. Check Production alias
    aliases = model.aliases
    assert "Production" in aliases, f"Production alias not found on model '{model_name}'. Aliases: {aliases}"
    assert int(aliases["Production"]) >= 1, f"Production alias points to invalid version: {aliases['Production']}"


# =========================================================================
# 4. Apache Airflow Integration Tests
# =========================================================================

def test_airflow_health(project_settings):
    """Verify Apache Airflow Webserver and REST API are healthy."""
    airflow_cfg = project_settings.get("orchestration", {}).get("airflow", {})
    webserver_url = airflow_cfg.get("webserver_url", "http://localhost:8080")

    health_url = f"{webserver_url.rstrip('/')}/health"
    resp = requests.get(health_url, timeout=5.0)

    assert resp.status_code == 200, f"Airflow health check failed with status {resp.status_code}: {resp.text}"
    data = resp.json()

    assert "metadatabase" in data, f"Unexpected Airflow health response: {data}"
    assert data["metadatabase"].get("status") == "healthy", (
        f"Airflow metadatabase reports unhealthy status: {data['metadatabase']}"
    )
