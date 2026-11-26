"""Integration Tests for Phase 4: Real-Time Telemetry Ingestion & Feature Buffering.

Verifies:
1. Pydantic v2 schemas reject malformed coordinates or missing feature keys.
2. SpatialTemporalBuffer strictly enforces sliding window caps (e.g. N and N+50) in Redis
   and maintains chronological order.
3. TelemetryProducer CLI execution (--max-events 25 --rate 50) publishes 25 valid JSON
   records to Kafka 'raw-telemetry' that deserialize cleanly into TelemetryEvent objects.
"""

import json
from pathlib import Path
import subprocess
import sys
from typing import List
from kafka import KafkaConsumer, TopicPartition
import pandas as pd
from pydantic import ValidationError
import pytest
import redis

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.common.redis_buffer import SpatialTemporalBuffer
from services.common.schemas import (
    DriftAlertEvent,
    PredictionEvent,
    REQUIRED_FEATURE_KEYS,
    TelemetryEvent,
)


@pytest.fixture(scope="module")
def valid_feature_dict():
    """Standard valid meteorological features fixture."""
    return {
        "rainfall_mm": 18.5,
        "soil_moisture": 0.55,
        "temperature_c": 22.4,
        "humidity_pct": 94.0,
        "wind_speed_kmh": 14.2,
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
    """Verify Kafka broker is available before running streaming integration tests."""
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
# 1. Pydantic v2 Schema Validation Tests
# =========================================================================

def test_telemetry_event_valid_and_roundtrip(valid_feature_dict):
    """Verify valid TelemetryEvent creation, alias handling, and JSON roundtrip."""
    event = TelemetryEvent(
        latitude=12.25,
        longitude=75.82,
        zone_id="ZONE_KODAGU",
        features=valid_feature_dict,
        delayed_label=1,
    )

    assert event.event_id is not None
    assert len(event.event_id) > 10
    assert event.lat == 12.25
    assert event.lon == 75.82
    assert event.zone_id == "ZONE_KODAGU"
    assert event.features["rainfall_mm"] == 18.5
    assert event.delayed_label == 1

    # Test JSON serialization & deserialization roundtrip
    json_str = event.to_json()
    reconstructed = TelemetryEvent.from_json(json_str)
    assert reconstructed.event_id == event.event_id
    assert reconstructed.lat == event.lat
    assert reconstructed.lon == event.lon
    assert reconstructed.features == event.features
    assert reconstructed.delayed_label == event.delayed_label


@pytest.mark.parametrize(
    "bad_lat, bad_lon",
    [
        (90.1, 75.0),    # Latitude > 90
        (-90.1, 75.0),   # Latitude < -90
        (12.0, 180.1),   # Longitude > 180
        (12.0, -180.1),  # Longitude < -180
        ("invalid", 75.0), # Non-numeric latitude
        (12.0, "invalid"), # Non-numeric longitude
    ],
)
def test_telemetry_event_rejects_malformed_coordinates(bad_lat, bad_lon, valid_feature_dict):
    """Verify TelemetryEvent rejects out-of-bound or malformed coordinate inputs."""
    with pytest.raises(ValidationError):
        TelemetryEvent(
            lat=bad_lat,
            lon=bad_lon,
            zone_id="ZONE_KODAGU",
            features=valid_feature_dict,
        )


def test_telemetry_event_rejects_missing_feature_keys():
    """Verify TelemetryEvent rejects payload if any required feature key is missing."""
    # Missing wind_speed_kmh
    incomplete_features = {
        "rainfall_mm": 10.0,
        "soil_moisture": 0.4,
        "temperature_c": 25.0,
        "humidity_pct": 80.0,
    }
    with pytest.raises(ValidationError) as excinfo:
        TelemetryEvent(
            lat=12.25,
            lon=75.82,
            zone_id="ZONE_KODAGU",
            features=incomplete_features,
        )
    assert "Missing required feature keys" in str(excinfo.value)

    # Completely empty features
    with pytest.raises(ValidationError) as excinfo_empty:
        TelemetryEvent(
            lat=12.25,
            lon=75.82,
            zone_id="ZONE_KODAGU",
            features={},
        )
    assert "Missing required feature keys" in str(excinfo_empty.value)


def test_telemetry_event_rejects_invalid_feature_values():
    """Verify TelemetryEvent rejects non-numeric or None feature values."""
    corrupted_features = {
        "rainfall_mm": "heavy_downpour",
        "soil_moisture": 0.4,
        "temperature_c": 25.0,
        "humidity_pct": 80.0,
        "wind_speed_kmh": 10.0,
    }
    with pytest.raises(ValidationError) as excinfo:
        TelemetryEvent(
            lat=12.25,
            lon=75.82,
            zone_id="ZONE_KODAGU",
            features=corrupted_features,
        )
    err_msg = str(excinfo.value).lower()
    assert (
        "unable to parse string as a number" in err_msg
        or "valid number" in err_msg
        or "numerical float" in err_msg
    )


def test_telemetry_event_rejects_empty_zone_and_invalid_delayed_label(valid_feature_dict):
    """Verify zone_id cannot be blank and delayed_label must be 0 or 1."""
    with pytest.raises(ValidationError):
        TelemetryEvent(
            lat=12.25,
            lon=75.82,
            zone_id="   ",
            features=valid_feature_dict,
        )

    with pytest.raises(ValidationError):
        TelemetryEvent(
            lat=12.25,
            lon=75.82,
            zone_id="ZONE_KODAGU",
            features=valid_feature_dict,
            delayed_label=5,  # Out of bounds
        )


def test_prediction_and_drift_alert_schemas(valid_feature_dict):
    """Verify PredictionEvent and DriftAlertEvent schemas."""
    pred = PredictionEvent(
        lat=12.25,
        lon=75.82,
        zone_id="ZONE_KODAGU",
        predicted_label=1,
        hazard_probability=0.88,
        model_version="v1",
        inference_latency_ms=1.45,
    )
    assert pred.predicted_label == 1
    assert pred.hazard_probability == 0.88

    # Prediction probability > 1.0 must fail
    with pytest.raises(ValidationError):
        PredictionEvent(
            lat=12.25,
            lon=75.82,
            zone_id="ZONE_KODAGU",
            predicted_label=1,
            hazard_probability=1.5,
            model_version="v1",
            inference_latency_ms=1.0,
        )

    # Drift alert schema validation
    alert = DriftAlertEvent(
        zone_id="ZONE_KODAGU",
        psi_scores={"rainfall_mm": 0.35, "soil_moisture": 0.42},
        ks_pvalues={"rainfall_mm": 0.001, "soil_moisture": 0.002},
        max_psi=0.42,
        min_pvalue=0.001,
        drift_detected=True,
        retrain_triggered=True,
    )
    assert alert.drift_detected is True
    assert alert.retrain_triggered is True
    assert DriftAlertEvent.from_json(alert.to_json()) == alert


# =========================================================================
# 2. Redis Sliding Window Cap Enforcement & Chronological Order Tests
# =========================================================================

def test_redis_buffer_sliding_window_cap_enforcement(redis_client, valid_feature_dict):
    """Verify SpatialTemporalBuffer strictly enforces cap when pushing W + 50 events."""
    window_cap = 50
    test_zone = "ZONE_TEST_CAP_50"

    buffer = SpatialTemporalBuffer(
        client=redis_client,
        sliding_window_size=window_cap,
    )
    buffer.clear_zone_window(test_zone)

    # Push window_cap + 50 (50 + 50 = 100) events with incrementing rainfall
    total_events = window_cap + 50
    for i in range(1, total_events + 1):
        feat = dict(valid_feature_dict)
        feat["rainfall_mm"] = float(i)
        ev = TelemetryEvent(
            lat=12.25,
            lon=75.82,
            zone_id=test_zone,
            features=feat,
            delayed_label=0,
        )
        current_len = buffer.push_event(ev)

    # 1. Verify buffer strictly capped at window_cap
    final_size = buffer.get_window_size(test_zone)
    assert final_size == window_cap, f"Expected size {window_cap}, got {final_size}"

    records = buffer.get_zone_window(test_zone)
    assert len(records) == window_cap

    # 2. Verify chronological order: oldest event is 51.0, newest is 100.0
    rain_sequence = [r["features"]["rainfall_mm"] for r in records]
    expected_sequence = [float(x) for x in range(51, 101)]
    assert rain_sequence == expected_sequence, (
        f"Chronological sequence mismatch. Expected 51.0..100.0, got first={rain_sequence[0]}, last={rain_sequence[-1]}"
    )

    # 3. Verify limit parameter retrieves the most recent N items chronologically
    recent_10 = buffer.get_zone_window(test_zone, limit=10)
    assert len(recent_10) == 10
    recent_10_rain = [r["features"]["rainfall_mm"] for r in recent_10]
    assert recent_10_rain == [float(x) for x in range(91, 101)]

    # 4. Verify DataFrame representation
    df = buffer.get_zone_dataframe(test_zone)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == window_cap
    assert list(df["rainfall_mm"]) == expected_sequence

    # 5. Verify active zones scan
    active_zones = buffer.get_all_active_zones()
    assert test_zone in active_zones

    # Cleanup
    buffer.clear_zone_window(test_zone)
    assert buffer.get_window_size(test_zone) == 0


def test_redis_buffer_large_sliding_window_500(redis_client, valid_feature_dict):
    """Verify sliding_window_size cap of 500 when pushing 550 events."""
    window_cap = 500
    test_zone = "ZONE_TEST_CAP_500"

    buffer = SpatialTemporalBuffer(
        client=redis_client,
        sliding_window_size=window_cap,
    )
    buffer.clear_zone_window(test_zone)

    # Push 550 events
    for i in range(1, 551):
        feat = dict(valid_feature_dict)
        feat["rainfall_mm"] = float(i)
        ev = TelemetryEvent(
            lat=12.25,
            lon=75.82,
            zone_id=test_zone,
            features=feat,
        )
        buffer.push_event(ev)

    assert buffer.get_window_size(test_zone) == 500
    records = buffer.get_zone_window(test_zone)
    assert len(records) == 500
    assert records[0]["features"]["rainfall_mm"] == 51.0
    assert records[-1]["features"]["rainfall_mm"] == 550.0

    buffer.clear_zone_window(test_zone)


# =========================================================================
# 3. Kafka Telemetry Producer End-to-End Ingestion Round-Trip
# =========================================================================

def test_telemetry_producer_publishes_25_events_to_kafka(kafka_broker):
    """Verify telemetry_producer.py CLI emits 25 valid JSON records into raw-telemetry."""
    topic = "raw-telemetry"
    num_events = 25
    rate = 50

    # 1. Establish initial partition end offsets prior to producer execution
    consumer_init = KafkaConsumer(
        bootstrap_servers=kafka_broker,
        enable_auto_commit=False,
        consumer_timeout_ms=5000,
        api_version=(3, 7, 0),
    )
    partitions = consumer_init.partitions_for_topic(topic)
    assert partitions is not None and len(partitions) > 0, f"Topic '{topic}' has no partitions"
    topic_partitions = [TopicPartition(topic, p) for p in sorted(list(partitions))]
    start_offsets = consumer_init.end_offsets(topic_partitions)
    consumer_init.close()

    # 2. Run telemetry_producer.py CLI as a subprocess
    producer_script = PROJECT_ROOT / "services" / "ingestion" / "telemetry_producer.py"
    csv_path = PROJECT_ROOT / "data" / "processed" / "stream_simulation.csv"
    assert producer_script.is_file(), f"Missing producer script at {producer_script}"
    assert csv_path.is_file(), f"Missing simulation CSV at {csv_path}"

    cmd = [
        sys.executable,
        str(producer_script),
        "--max-events", str(num_events),
        "--rate", str(rate),
        "--bootstrap-servers", kafka_broker,
        "--topic", topic,
        "--csv-path", str(csv_path),
        "--mode", "transition",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    output = result.stdout + "\n" + result.stderr
    assert result.returncode == 0, (
        f"telemetry_producer failed with code {result.returncode}:\nOUTPUT:\n{output}"
    )
    assert f"Total emitted: {num_events} events" in output

    # 3. Read newly produced messages from start_offsets
    consumer = KafkaConsumer(
        bootstrap_servers=kafka_broker,
        enable_auto_commit=False,
        consumer_timeout_ms=10000,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        key_deserializer=lambda k: k.decode("utf-8") if k else None,
        api_version=(3, 7, 0),
    )
    consumer.assign(topic_partitions)
    for tp in topic_partitions:
        consumer.seek(tp, start_offsets[tp])

    consumed_messages = []
    for msg in consumer:
        consumed_messages.append(msg)
        if len(consumed_messages) >= num_events:
            break
    consumer.close()

    assert len(consumed_messages) == num_events, (
        f"Expected {num_events} messages, consumed {len(consumed_messages)}"
    )

    # 4. Validate that all messages deserialize cleanly into TelemetryEvent objects
    known_zones = {
        "ZONE_KODAGU",
        "ZONE_DAKSHINA_KANNADA",
        "ZONE_CHIKMAGALUR",
        "ZONE_SHIMOGA_AGUMBE",
        "ZONE_UTTARA_KANNADA",
        "ZONE_WAYANAD_NILGIRIS",
    }

    for msg in consumed_messages:
        # Verify key
        assert msg.key in known_zones, f"Message key '{msg.key}' not a known zone"

        # Deserialize into TelemetryEvent
        telemetry_event = TelemetryEvent.model_validate(msg.value)

        # Assert schema fields
        assert telemetry_event.event_id is not None
        assert telemetry_event.zone_id == msg.key
        assert -90.0 <= telemetry_event.lat <= 90.0
        assert -180.0 <= telemetry_event.lon <= 180.0
        assert REQUIRED_FEATURE_KEYS.issubset(set(telemetry_event.features.keys()))

        # Verify numerical properties
        for feat_name, feat_val in telemetry_event.features.items():
            assert isinstance(feat_val, (int, float))
            assert not pd.isna(feat_val)

        if telemetry_event.delayed_label is not None:
            assert telemetry_event.delayed_label in {0, 1}
