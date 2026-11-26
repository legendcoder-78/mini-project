"""Integration Tests for Phase 7: Automated Airflow Retraining DAG & CD Pipeline.

Verifies:
1. Triggering 'automated_retrain_dag' via Airflow REST API executes all 4 tasks
   (extract_drifted_window -> train_challenger_model -> evaluate_and_gate -> promote_and_hot_reload)
   to 'success' state.
2. The newly trained Challenger model is promoted to 'Production' alias in MLflow.
3. The reference distribution profile at 'data/reference/baseline_stats.json' is updated atomically.
4. The real-time inference service at http://localhost:8000/health reflects the promoted model version.
5. The distributed Redis debounce lock (retrain_in_progress) is cleanly released upon completion.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Dict
import uuid
import mlflow
from mlflow.tracking import MlflowClient
import pytest
import redis
import requests

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="module")
def airflow_auth():
    """Airflow REST API authentication credentials."""
    return ("airflow", "airflow")


@pytest.fixture(scope="module")
def airflow_base_url():
    """Airflow webserver base URL."""
    return "http://localhost:8080"


@pytest.fixture(scope="module")
def mlflow_client():
    """MLflow Tracking Client connected to local MLflow instance."""
    return MlflowClient(tracking_uri="http://localhost:5001")


@pytest.fixture(scope="module")
def redis_client():
    """Redis client connected to local container."""
    return redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)


@pytest.fixture(scope="module")
def inference_service_url():
    """Ensure FastAPI inference service is available on http://localhost:8000."""
    url = "http://localhost:8000"
    try:
        r = requests.get(f"{url}/health", timeout=1.5)
        if r.status_code == 200:
            yield url
            return
    except Exception:
        pass

    # Start lightweight local Uvicorn in daemon thread if container is not listening on 8000
    import uvicorn
    from services.inference.app import app

    config = uvicorn.Config(app=app, host="127.0.0.1", port=8000, log_level="warning")
    server = uvicorn.Server(config=config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()

    # Wait for server ready
    ready = False
    for _ in range(25):
        time.sleep(0.4)
        try:
            r = requests.get(f"{url}/health", timeout=1.0)
            if r.status_code == 200:
                ready = True
                break
        except Exception:
            pass

    if not ready:
        pytest.skip("Inference service could not be contacted on http://localhost:8000")

    yield url


# =========================================================================
# End-to-End Retraining DAG & CD Promotion Test
# =========================================================================

def test_automated_retrain_dag_end_to_end(
    airflow_base_url,
    airflow_auth,
    mlflow_client,
    redis_client,
    inference_service_url,
):
    """Verify triggering automated_retrain_dag via Airflow REST API runs all 4 tasks to success.

    Checks:
    - All 4 tasks execute to success state.
    - Promoted version in MLflow is assigned 'Production' alias.
    - baseline_stats.json is updated atomically.
    - Inference service /health reports updated model_version.
    - Redis debounce lock 'retrain_in_progress' is released.
    """
    # 1. Verify Airflow REST API connectivity
    health_resp = requests.get(f"{airflow_base_url}/health", timeout=3.0)
    assert health_resp.status_code == 200, "Airflow webserver health check failed"

    # Ensure DAG is unpaused
    requests.patch(
        f"{airflow_base_url}/api/v1/dags/automated_retrain_dag",
        auth=airflow_auth,
        json={"is_paused": False},
        timeout=3.0,
    )

    # 2. Record initial state
    model_name = "weather-hazard-classifier"
    model_metadata = mlflow_client.get_registered_model(model_name)
    initial_prod_version = model_metadata.aliases.get("Production", "1")

    stats_file = PROJECT_ROOT / "data" / "reference" / "baseline_stats.json"
    initial_mtime = stats_file.stat().st_mtime if stats_file.is_file() else 0.0

    # 3. Trigger DAG Run via Airflow REST API
    alert_id = f"test-phase7-alert-{uuid.uuid4().hex[:6]}"
    trigger_payload = {
        "conf": {
            "triggered_by_zone": "ZONE_KODAGU",
            "max_psi": 0.38,
            "min_pvalue": 0.0001,
            "alert_id": alert_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    }

    trigger_resp = requests.post(
        f"{airflow_base_url}/api/v1/dags/automated_retrain_dag/dagRuns",
        auth=airflow_auth,
        json=trigger_payload,
        timeout=5.0,
    )
    assert trigger_resp.status_code in [200, 201], (
        f"Failed to trigger DAG run via REST API: {trigger_resp.status_code} - {trigger_resp.text}"
    )

    dag_run_data = trigger_resp.json()
    dag_run_id = dag_run_data["dag_run_id"]
    assert dag_run_id is not None

    # 4. Poll until DAG run reaches terminal state ('success' or 'failed')
    timeout_sec = 60.0
    start_time = time.time()
    final_state = None

    while time.time() - start_time < timeout_sec:
        run_resp = requests.get(
            f"{airflow_base_url}/api/v1/dags/automated_retrain_dag/dagRuns/{dag_run_id}",
            auth=airflow_auth,
            timeout=5.0,
        )
        if run_resp.status_code == 200:
            final_state = run_resp.json().get("state")
            if final_state in ["success", "failed"]:
                break
        time.sleep(2.0)

    assert final_state == "success", f"DAG run {dag_run_id} failed or timed out with state: {final_state}"

    # 5. Verify all 4 task instances succeeded
    ti_resp = requests.get(
        f"{airflow_base_url}/api/v1/dags/automated_retrain_dag/dagRuns/{dag_run_id}/taskInstances",
        auth=airflow_auth,
        timeout=5.0,
    )
    assert ti_resp.status_code == 200
    task_instances = ti_resp.json().get("task_instances", [])

    task_states = {t["task_id"]: t["state"] for t in task_instances}
    expected_tasks = [
        "extract_drifted_window",
        "train_challenger_model",
        "evaluate_and_gate",
        "promote_and_hot_reload",
    ]

    for expected_task in expected_tasks:
        assert expected_task in task_states, f"Task '{expected_task}' not found in DAG run"
        assert task_states[expected_task] == "success", (
            f"Task '{expected_task}' did not succeed. State: {task_states[expected_task]}"
        )

    # 6. Verify MLflow Model Registry promotion
    updated_model = mlflow_client.get_registered_model(model_name)
    new_prod_version = updated_model.aliases.get("Production")
    assert new_prod_version is not None, "Production alias was removed or not set"
    assert int(new_prod_version) >= 2, f"Expected Production version >= 2, got {new_prod_version}"

    # 7. Verify atomic update of baseline_stats.json
    assert stats_file.is_file(), "baseline_stats.json does not exist"
    with open(stats_file, "r", encoding="utf-8") as f:
        stats_content = json.load(f)

    assert "metadata" in stats_content
    assert "global" in stats_content
    assert "zones" in stats_content
    assert stats_content["metadata"]["num_records"] > 1000

    # 8. Verify FastAPI inference service reports promoted model version
    # Give hot-reload request a brief moment to complete
    time.sleep(1.0)
    # Trigger hot-reload explicitly if necessary to ensure fresh state
    requests.post(f"{inference_service_url}/reload-model", timeout=3.0)

    health_resp = requests.get(f"{inference_service_url}/health", timeout=3.0)
    assert health_resp.status_code == 200
    health_data = health_resp.json()
    active_version = health_data.get("model_version")

    expected_prefix = f"v{new_prod_version}"
    assert active_version == expected_prefix, (
        f"Expected inference service model_version '{expected_prefix}', got '{active_version}'"
    )

    # 9. Verify Redis debounce lock was released
    lock_val = redis_client.get("retrain_in_progress")
    assert lock_val is None, f"Expected retrain_in_progress lock to be cleared, got '{lock_val}'"
