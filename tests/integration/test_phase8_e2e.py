"""End-to-End Integration Tests for Phase 8: GIS Dashboard & Real-Time MLOps Closed-Loop.

Verifies:
1. Static hosting: GET / returns valid HTML5 dashboard and GET /api/zones returns valid
   Karnataka/Western Ghats bounding polygons and PSI thresholds.
2. Real-time WebSocket streaming: A WebSocket client connected to ws://localhost:8000/ws/telemetry
   receives live 'prediction' envelopes as telemetry is scored.
3. Live drift alerting: WebSocket client receives 'drift_alert' envelopes when spatial drift
   telemetry is published to Kafka 'drift-alerts'.
4. Zero-downtime hot-reload: WebSocket client receives 'model_reload' envelopes when the
   inference service reloads the promoted model version.
5. Full closed-loop verification: stream ingestion -> prediction event -> drift detection -> hot-reload.
"""

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Dict
from kafka import KafkaProducer
import pytest
import requests
import websockets

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BASE_HTTP_URL = "http://localhost:8000"
BASE_WS_URL = "ws://localhost:8000/ws/telemetry"


@pytest.fixture(scope="module")
def ensure_service_ready():
    """Verify inference service is healthy before running tests."""
    for _ in range(15):
        try:
            r = requests.get(f"{BASE_HTTP_URL}/health", timeout=1.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    pytest.skip(f"Inference service not reachable at {BASE_HTTP_URL}")


@pytest.fixture(scope="module")
def kafka_producer():
    """Kafka producer for injecting synthetic test events."""
    try:
        producer = KafkaProducer(
            bootstrap_servers="localhost:29092",
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if isinstance(k, str) else k,
            acks=1,
            api_version=(3, 7, 0),
        )
        return producer
    except Exception as exc:
        pytest.skip(f"Kafka not accessible on localhost:29092: {exc}")


# =========================================================================
# 1. Dashboard Static Serving & Zones Metadata Tests
# =========================================================================

def test_dashboard_static_hosting_and_index_html(ensure_service_ready):
    """Verify GET / serves the GIS command dashboard and assets are accessible."""
    resp = requests.get(f"{BASE_HTTP_URL}/", timeout=3.0)
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")

    html_text = resp.text
    assert "Western Ghats & Karnataka Early Warning System" in html_text
    assert 'id="gis-map"' in html_text
    assert 'id="drift-chart"' in html_text
    assert 'id="audit-log-body"' in html_text
    assert "leaflet.js" in html_text
    assert "chart.umd.min.js" in html_text

    # Verify CSS & JS static asset delivery
    css_resp = requests.get(f"{BASE_HTTP_URL}/css/styles.css", timeout=2.0)
    assert css_resp.status_code == 200
    assert "dashboard-grid" in css_resp.text

    map_js_resp = requests.get(f"{BASE_HTTP_URL}/js/map_controller.js", timeout=2.0)
    assert map_js_resp.status_code == 200
    assert "MapController" in map_js_resp.text


def test_api_spatial_zones_endpoint(ensure_service_ready):
    """Verify GET /api/zones returns valid Karnataka spatial polygons and PSI threshold."""
    resp = requests.get(f"{BASE_HTTP_URL}/api/zones", timeout=3.0)
    assert resp.status_code == 200

    data = resp.json()
    assert "zones" in data
    assert "psi_threshold" in data
    assert "features" in data

    zones = data["zones"]
    assert len(zones) >= 6

    zone_ids = [z["zone_id"] for z in zones]
    assert "ZONE_KODAGU" in zone_ids
    assert "ZONE_CHIKMAGALUR" in zone_ids
    assert "ZONE_SHIMOGA_AGUMBE" in zone_ids

    # Validate coordinate bounds
    for z in zones:
        assert z["min_lat"] < z["max_lat"]
        assert z["min_lon"] < z["max_lon"]

    assert data["psi_threshold"] > 0.0


# =========================================================================
# 2. Real-Time WebSocket Streaming & Event Envelopes
# =========================================================================

def test_websocket_ping_pong(ensure_service_ready):
    """Verify WebSocket connectivity and heartbeat ping-pong."""
    async def run():
        async with websockets.connect(BASE_WS_URL) as ws:
            await ws.send("ping")
            raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
            msg = json.loads(raw)
            assert msg.get("type") == "pong"
            assert "timestamp" in msg

    asyncio.run(run())


def test_websocket_prediction_broadcast(ensure_service_ready):
    """Verify connected WebSocket receives 'prediction' messages when events are scored."""
    async def run():
        async with websockets.connect(BASE_WS_URL) as ws:
            # Trigger prediction via POST /predict
            payload = {
                "lat": 12.25,
                "lon": 75.82,
                "zone_id": "ZONE_KODAGU",
                "features": {
                    "rainfall_mm": 55.0,
                    "soil_moisture": 0.85,
                    "temperature_c": 21.0,
                    "humidity_pct": 98.0,
                    "wind_speed_kmh": 28.0,
                },
                "delayed_label": 1,
            }
            p_resp = requests.post(f"{BASE_HTTP_URL}/predict", json=payload, timeout=3.0)
            assert p_resp.status_code == 200

            # Await broadcast envelope
            raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
            msg = json.loads(raw)

            assert msg.get("type") == "prediction"
            data = msg.get("data", {})
            assert data.get("zone_id") == "ZONE_KODAGU"
            assert data.get("predicted_label") in [0, 1]
            assert 0.0 <= data.get("hazard_probability") <= 1.0
            assert "model_version" in data
            assert data.get("inference_latency_ms") >= 0.0

    asyncio.run(run())


def test_websocket_drift_alert_broadcast(ensure_service_ready, kafka_producer):
    """Verify WebSocket clients receive 'drift_alert' messages emitted to Kafka."""
    async def run():
        async with websockets.connect(BASE_WS_URL) as ws:
            alert_id = f"test-drift-{int(time.time())}"
            alert_payload = {
                "alert_id": alert_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "zone_id": "ZONE_CHIKMAGALUR",
                "psi_scores": {"rainfall_mm": 0.32, "soil_moisture": 0.28},
                "ks_pvalues": {"rainfall_mm": 0.0001, "soil_moisture": 0.002},
                "max_psi": 0.32,
                "min_pvalue": 0.0001,
                "drift_detected": True,
                "retrain_triggered": True,
            }

            # Send to Kafka drift-alerts topic
            kafka_producer.send("drift-alerts", key="ZONE_CHIKMAGALUR", value=alert_payload)
            kafka_producer.flush(timeout=2)

            # Receive on WebSocket
            received_alert = False
            start_time = time.time()
            while time.time() - start_time < 5.0:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=1.5)
                    msg = json.loads(raw)
                    if msg.get("type") == "drift_alert":
                        d = msg.get("data", {})
                        if d.get("zone_id") == "ZONE_CHIKMAGALUR":
                            assert d.get("drift_detected") is True
                            assert d.get("max_psi") >= 0.20
                            received_alert = True
                            break
                except asyncio.TimeoutError:
                    break

            assert received_alert, "Did not receive expected 'drift_alert' on WebSocket"

    asyncio.run(run())


def test_websocket_model_reload_broadcast(ensure_service_ready):
    """Verify WebSocket clients receive 'model_reload' broadcast when /reload-model is called."""
    async def run():
        async with websockets.connect(BASE_WS_URL) as ws:
            # Trigger POST /reload-model
            reload_resp = requests.post(f"{BASE_HTTP_URL}/reload-model", timeout=5.0)
            assert reload_resp.status_code == 200

            # Await reload notification on WebSocket
            received_reload = False
            start_time = time.time()
            while time.time() - start_time < 4.0:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=1.5)
                    msg = json.loads(raw)
                    if msg.get("type") == "model_reload":
                        data = msg.get("data", {})
                        assert "active_version" in data
                        assert "previous_version" in data
                        assert "reloaded_at" in data
                        received_reload = True
                        break
                except asyncio.TimeoutError:
                    break

            assert received_reload, "Did not receive expected 'model_reload' on WebSocket"

    asyncio.run(run())


def test_api_drift_events_and_model_performance(ensure_service_ready):
    """Verify GET /api/drift/events and GET /api/model/performance expose persistent drift history."""
    resp = requests.get(f"{BASE_HTTP_URL}/api/drift/events", timeout=3.0)
    assert resp.status_code == 200
    events = resp.json()
    assert isinstance(events, list)

    perf_resp = requests.get(f"{BASE_HTTP_URL}/api/model/performance", timeout=3.0)
    assert perf_resp.status_code == 200
    perf_data = perf_resp.json()
    assert "drift_event_history" in perf_data
    assert "evaluation_history" in perf_data
    assert "psi_threshold" in perf_data
    assert "multi_city_drift_threshold" in perf_data
    assert perf_data["psi_threshold"] == 0.20
    assert perf_data["multi_city_drift_threshold"] == 4


def test_retrain_outcome_webhook_and_websocket_broadcast(ensure_service_ready):
    """Verify POST /api/retrain/outcome broadcasts approval and rejection events over WebSocket."""
    async def run():
        async with websockets.connect(BASE_WS_URL) as ws:
            # 1. Test Approval Scenario
            approval_payload = {
                "event_id": "test_evt_approval_101",
                "decision": "approved",
                "promoted": True,
                "champion_version": "v1",
                "challenger_version": "v2",
                "final_production_model_version": "v2",
                "decision_timestamp": datetime.now(timezone.utc).isoformat(),
                "deployment_timestamp": datetime.now(timezone.utc).isoformat(),
                "f1_improvement": 0.0245,
                "reason": "Challenger achieved +2.45% F1 improvement on validation window.",
            }

            resp = requests.post(f"{BASE_HTTP_URL}/api/retrain/outcome", json=approval_payload, timeout=5.0)
            assert resp.status_code == 200
            assert resp.json()["status"] == "success"

            # Receive on WebSocket
            received_approval = False
            start_time = time.time()
            while time.time() - start_time < 4.0:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=1.5)
                    msg = json.loads(raw)
                    if msg.get("type") == "retrain_outcome":
                        d = msg.get("data", {})
                        if d.get("event_id") == "test_evt_approval_101":
                            assert d.get("decision") == "approved"
                            assert d.get("promoted") is True
                            assert d.get("final_production_model_version") == "v2"
                            received_approval = True
                            break
                except asyncio.TimeoutError:
                    break

            assert received_approval, "Did not receive expected 'retrain_outcome' (approved) on WebSocket"

            # 2. Test Rejection Scenario
            rejection_payload = {
                "event_id": "test_evt_rejection_202",
                "decision": "rejected",
                "promoted": False,
                "champion_version": "v1",
                "challenger_version": "v3",
                "final_production_model_version": "v1",
                "decision_timestamp": datetime.now(timezone.utc).isoformat(),
                "deployment_timestamp": None,
                "f1_improvement": -0.0050,
                "reason": "Challenger F1 degradation (-0.50% vs champion). Gate rejected.",
            }

            resp2 = requests.post(f"{BASE_HTTP_URL}/api/retrain/outcome", json=rejection_payload, timeout=5.0)
            assert resp2.status_code == 200
            assert resp2.json()["status"] == "success"

            # Receive on WebSocket
            received_rejection = False
            start_time = time.time()
            while time.time() - start_time < 4.0:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=1.5)
                    msg = json.loads(raw)
                    if msg.get("type") == "retrain_outcome":
                        d = msg.get("data", {})
                        if d.get("event_id") == "test_evt_rejection_202":
                            assert d.get("decision") == "rejected"
                            assert d.get("promoted") is False
                            assert d.get("final_production_model_version") == "v1"
                            received_rejection = True
                            break
                except asyncio.TimeoutError:
                    break

            assert received_rejection, "Did not receive expected 'retrain_outcome' (rejected) on WebSocket"

    asyncio.run(run())
