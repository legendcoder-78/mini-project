"""Integration Tests for Phase 6: Spatial-Temporal Drift Monitor Worker.

Verifies:
1. Pushing heavily drifted micro-climate records into Redis for two distinct zones triggers
   exactly ONE Airflow REST API call via the distributed Redis debounce lock (SET retrain_in_progress 1 NX EX 300).
2. Both zones publish valid DriftAlertEvent JSON messages to Kafka's 'drift-alerts' topic.
3. An active debounce lock suppresses duplicate Airflow triggers across subsequent polling cycles.
4. Transient Airflow dispatch errors cleanly release the debounce lock to permit retries.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Dict, List
import uuid
from kafka import KafkaConsumer, TopicPartition
import pytest
import redis
import requests

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.common.redis_buffer import SpatialTemporalBuffer
from services.common.schemas import DriftAlertEvent, TelemetryEvent
from services.drift_monitor.detector import SpatialTemporalDriftDetector
from services.drift_monitor.worker import DriftMonitorWorker


@pytest.fixture(scope="module")
def redis_client():
    """Redis client connected to the local container."""
    client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    try:
        client.ping()
    except Exception as exc:
        pytest.skip(f"Redis not accessible on localhost:6379: {exc}")
    return client


@pytest.fixture(scope="module")
def redis_buffer(redis_client):
    """SpatialTemporalBuffer instance using the shared Redis container."""
    return SpatialTemporalBuffer(client=redis_client)


@pytest.fixture
def clean_redis_state(redis_client, redis_buffer):
    """Ensure test keys and debounce locks are purged before and after each test."""
    redis_client.delete(DriftMonitorWorker.LOCK_KEY)
    redis_buffer.clear_all_windows()

    yield

    redis_client.delete(DriftMonitorWorker.LOCK_KEY)
    redis_buffer.clear_all_windows()


def _generate_drifted_telemetry(zone_id: str, count: int = 40) -> List[TelemetryEvent]:
    """Generate heavily drifted monsoon telemetry events for a given zone."""
    events = []
    lat = 12.25 if zone_id == "ZONE_KODAGU" else 13.32
    lon = 75.82 if zone_id == "ZONE_KODAGU" else 75.77

    for _ in range(count):
        ev = TelemetryEvent(
            lat=lat,
            lon=lon,
            zone_id=zone_id,
            features={
                "rainfall_mm": 68.5,      # Normal baseline ~0.0-5.0mm
                "soil_moisture": 0.94,     # Normal baseline ~0.3-0.55
                "temperature_c": 17.5,     # Normal baseline ~24-28C
                "humidity_pct": 99.0,      # Normal baseline ~70-85%
                "wind_speed_kmh": 36.0,    # Normal baseline ~8-15 km/h
            },
            delayed_label=1,
        )
        events.append(ev)
    return events


# =========================================================================
# 1. Debounce Lock & Single Airflow Trigger Verification
# =========================================================================

def test_drift_worker_debounce_lock_and_single_airflow_trigger(
    clean_redis_state,
    redis_client,
    redis_buffer,
):
    """Verify two drifted zones trigger only ONE Airflow DAG run and publish alerts to Kafka."""
    zone_a = "ZONE_KODAGU"
    zone_b = "ZONE_CHIKMAGALUR"

    # 1. Inspect existing partition end offsets for 'drift-alerts'
    try:
        probe_consumer = KafkaConsumer(
            bootstrap_servers="localhost:29092",
            enable_auto_commit=False,
            api_version=(3, 7, 0),
        )
        partitions = probe_consumer.partitions_for_topic("drift-alerts") or {0}
        tps = [TopicPartition("drift-alerts", p) for p in partitions]
        start_offsets = probe_consumer.end_offsets(tps)
        probe_consumer.close()
    except Exception as exc:
        pytest.skip(f"Kafka not accessible: {exc}")

    # 2. Push 40 drifted events into Redis for both zones
    for ev in _generate_drifted_telemetry(zone_a, count=40):
        redis_buffer.push_event(ev)
    for ev in _generate_drifted_telemetry(zone_b, count=40):
        redis_buffer.push_event(ev)

    assert redis_buffer.get_window_size(zone_a) == 40
    assert redis_buffer.get_window_size(zone_b) == 40

    # 3. Instantiate worker and spy on trigger_airflow_retraining
    worker = DriftMonitorWorker(
        poll_interval_sec=1.0,
        kafka_bootstrap="localhost:29092",
        alerts_topic="drift-alerts",
        redis_buffer=redis_buffer,
        airflow_url="http://localhost:8080",
    )

    airflow_triggered_zones: List[str] = []
    original_trigger = worker.trigger_airflow_retraining

    def spy_trigger(zone_id: str, alert_event: DriftAlertEvent) -> bool:
        airflow_triggered_zones.append(zone_id)
        return original_trigger(zone_id, alert_event)

    worker.trigger_airflow_retraining = spy_trigger

    # 4. Execute evaluation across active zones
    alert_events = worker.evaluate_all_active_zones()
    assert len(alert_events) >= 2, "Expected at least 2 evaluated zone events"

    # Filter for our test zones
    test_events = {e.zone_id: e for e in alert_events if e.zone_id in [zone_a, zone_b]}
    assert zone_a in test_events and zone_b in test_events

    # Both zones must detect drift
    event_a = test_events[zone_a]
    event_b = test_events[zone_b]
    assert event_a.drift_detected is True
    assert event_b.drift_detected is True
    assert event_a.max_psi > 0.20
    assert event_b.max_psi > 0.20
    assert event_a.min_pvalue < 0.05
    assert event_b.min_pvalue < 0.05

    # 5. Verify the debounce lock: exactly ONE Airflow trigger execution
    assert len(airflow_triggered_zones) == 1, (
        f"Expected exactly 1 Airflow trigger, got {len(airflow_triggered_zones)}: {airflow_triggered_zones}"
    )
    first_triggered_zone = airflow_triggered_zones[0]
    second_zone = zone_b if first_triggered_zone == zone_a else zone_a

    # Check retrain_triggered flags on the emitted events
    assert test_events[first_triggered_zone].retrain_triggered is True
    assert test_events[second_zone].retrain_triggered is False

    # 6. Verify Redis retrain_in_progress lock exists with valid TTL
    lock_val = redis_client.get(DriftMonitorWorker.LOCK_KEY)
    ttl = redis_client.ttl(DriftMonitorWorker.LOCK_KEY)
    assert lock_val is not None, "Redis retrain_in_progress lock was not set"
    assert f"zone={first_triggered_zone}" in lock_val
    assert 0 < ttl <= DriftMonitorWorker.LOCK_TIMEOUT_SEC

    # 7. Verify both DriftAlertEvents were published to Kafka 'drift-alerts' topic
    reader_consumer = KafkaConsumer(
        bootstrap_servers="localhost:29092",
        enable_auto_commit=False,
        consumer_timeout_ms=5000,
        api_version=(3, 7, 0),
    )
    reader_consumer.assign(tps)
    for tp, off in start_offsets.items():
        reader_consumer.seek(tp, off)

    consumed_records = []
    start_time = time.time()
    while time.time() - start_time < 5.0 and len(consumed_records) < 2:
        msg_pack = reader_consumer.poll(timeout_ms=1000)
        for tp, messages in msg_pack.items():
            for msg in messages:
                payload = json.loads(msg.value.decode("utf-8"))
                if payload.get("zone_id") in [zone_a, zone_b]:
                    consumed_records.append(payload)

    reader_consumer.close()

    assert len(consumed_records) >= 2, f"Expected 2 DriftAlertEvents in Kafka, got {len(consumed_records)}"

    # Validate schema of consumed messages
    kafka_zones = set()
    for rec in consumed_records:
        parsed = DriftAlertEvent.model_validate(rec)
        kafka_zones.add(parsed.zone_id)
        assert parsed.drift_detected is True
        assert parsed.max_psi > 0.20
        assert parsed.min_pvalue < 0.05

    assert zone_a in kafka_zones
    assert zone_b in kafka_zones


# =========================================================================
# 2. Debounce Lock Continuity Across Consecutive Cycles
# =========================================================================

def test_drift_worker_debounce_lock_continuity(
    clean_redis_state,
    redis_client,
    redis_buffer,
):
    """Verify that an active debounce lock prevents Airflow calls in consecutive polling cycles."""
    zone_id = "ZONE_KODAGU"

    for ev in _generate_drifted_telemetry(zone_id, count=35):
        redis_buffer.push_event(ev)

    worker = DriftMonitorWorker(
        poll_interval_sec=1.0,
        kafka_bootstrap="localhost:29092",
        redis_buffer=redis_buffer,
        airflow_url="http://localhost:8080",
    )

    trigger_calls = []
    worker.trigger_airflow_retraining = lambda zid, ev: trigger_calls.append(zid) or True

    # Cycle 1: Lock not held -> should acquire lock and trigger Airflow
    events_c1 = worker.evaluate_all_active_zones()
    z_ev1 = next(e for e in events_c1 if e.zone_id == zone_id)
    assert z_ev1.drift_detected is True
    assert z_ev1.retrain_triggered is True
    assert len(trigger_calls) == 1

    # Cycle 2: Lock already active in Redis -> should skip Airflow trigger
    events_c2 = worker.evaluate_all_active_zones()
    z_ev2 = next(e for e in events_c2 if e.zone_id == zone_id)
    assert z_ev2.drift_detected is True
    assert z_ev2.retrain_triggered is False
    assert len(trigger_calls) == 1, "Subsequent cycle triggered Airflow despite active debounce lock"


# =========================================================================
# 3. Lock Release On Downstream Airflow Failure
# =========================================================================

def test_drift_worker_releases_lock_on_airflow_dispatch_failure(
    clean_redis_state,
    redis_client,
    redis_buffer,
    monkeypatch,
):
    """Verify worker releases debounce lock if Airflow endpoint fails or is unreachable."""
    zone_id = "ZONE_KODAGU"

    for ev in _generate_drifted_telemetry(zone_id, count=35):
        redis_buffer.push_event(ev)

    # Simulate downstream connection failure on requests.post
    def mock_post_failure(*args, **kwargs):
        raise requests.ConnectionError("Airflow webserver unreachable")

    monkeypatch.setattr(requests, "post", mock_post_failure)

    worker = DriftMonitorWorker(
        poll_interval_sec=1.0,
        kafka_bootstrap="localhost:29092",
        redis_buffer=redis_buffer,
        airflow_url="http://localhost:8080",
    )

    events = worker.evaluate_all_active_zones()
    z_ev = next(e for e in events if e.zone_id == zone_id)

    assert z_ev.drift_detected is True
    assert z_ev.retrain_triggered is False

    # Debounce lock must be released so next cycle can retry
    lock_val = redis_client.get(DriftMonitorWorker.LOCK_KEY)
    assert lock_val is None, f"Expected lock to be released on dispatch failure, found {lock_val}"
