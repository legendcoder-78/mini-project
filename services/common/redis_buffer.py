"""Spatial-Temporal Feature Buffer in Redis.

Maintains bounded sliding-window feature buffers partitioned by spatial micro-climate zone.
Uses atomic Redis pipelines (LPUSH + LTRIM) to guarantee memory never exceeds the
configured window cap (e.g., 500 events per zone).
"""

import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union

# Ensure project root is in sys.path for direct CLI or submodule imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import redis
import yaml

from services.common.schemas import TelemetryEvent

logger = logging.getLogger(__name__)


def _load_default_config() -> Dict[str, Any]:
    """Load streaming and drift settings from config/settings.yaml if available."""
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / "config" / "settings.yaml"

    if config_path.is_file():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning("Could not read settings from %s: %s", config_path, e)
    return {}


class SpatialTemporalBuffer:
    """Manages zone-partitioned, bounded FIFO sliding windows in Redis.

    Guarantees strict sliding-window cap enforcement per spatial zone via atomic
    Redis pipelines (LPUSH + LTRIM 0 {sliding_window_size - 1}).
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db: Optional[int] = None,
        sliding_window_size: Optional[int] = None,
        key_prefix: Optional[str] = None,
        client: Optional[redis.Redis] = None,
    ):
        config = _load_default_config()
        redis_cfg = config.get("streaming", {}).get("redis", {})
        drift_cfg = config.get("drift_thresholds", {})

        self.host = (
            host
            if host is not None
            else (os.environ.get("REDIS_HOST") or redis_cfg.get("host", "localhost"))
        )
        self.port = int(
            os.environ.get("REDIS_PORT")
            or (port if port is not None else redis_cfg.get("port", 6379))
        )
        self.db = int(
            os.environ.get("REDIS_DB")
            or (db if db is not None else redis_cfg.get("db", 0))
        )
        self.sliding_window_size = int(
            sliding_window_size
            if sliding_window_size is not None
            else drift_cfg.get("sliding_window_size", 500)
        )
        self.key_prefix = (
            key_prefix
            if key_prefix is not None
            else redis_cfg.get("window_key_prefix", "window:")
        )

        if client is not None:
            self.client = client
        else:
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=True,
            )

    def ping(self) -> bool:
        """Verify Redis connectivity."""
        try:
            return bool(self.client.ping())
        except Exception as exc:
            logger.error("Failed to connect to Redis at %s:%d: %s", self.host, self.port, exc)
            return False

    def push_event(self, event: Union[TelemetryEvent, Dict[str, Any]]) -> int:
        """Atomically push a telemetry event into its zone sliding window and trim to cap.

        Args:
            event: Validated TelemetryEvent or dictionary adhering to TelemetryEvent schema.

        Returns:
            Current window size after push and trim.
        """
        if isinstance(event, dict):
            event = TelemetryEvent.model_validate(event)

        key = f"{self.key_prefix}{event.zone_id}"
        payload_json = event.to_json()

        # Atomic pipeline: LPUSH newest event, LTRIM to sliding_window_size, get LLEN
        pipe = self.client.pipeline()
        pipe.lpush(key, payload_json)
        pipe.ltrim(key, 0, self.sliding_window_size - 1)
        pipe.llen(key)
        results = pipe.execute()

        current_size = int(results[2])
        return current_size

    def get_zone_window(
        self, zone_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve sliding window records for a spatial zone in chronological order.

        Args:
            zone_id: Micro-climate spatial zone identifier.
            limit: Optional maximum number of recent records to return (defaults to all).

        Returns:
            List of parsed dictionary events ordered chronologically (oldest to newest).
        """
        key = f"{self.key_prefix}{zone_id}"

        if limit is not None and limit > 0:
            raw_records = self.client.lrange(key, 0, limit - 1)
        else:
            raw_records = self.client.lrange(key, 0, -1)

        if not raw_records:
            return []

        # LPUSH stores newest at index 0. Reversing yields chronological order (oldest -> newest)
        chronological_records = list(reversed(raw_records))
        return [json.loads(r) for r in chronological_records]

    def get_zone_dataframe(
        self, zone_id: str, limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Retrieve sliding window as a flattened pandas DataFrame for ML/drift evaluation.

        Flattens the nested 'features' dictionary into direct column values.
        """
        records = self.get_zone_window(zone_id=zone_id, limit=limit)
        if not records:
            return pd.DataFrame()

        rows = []
        for r in records:
            row = {
                "event_id": r.get("event_id"),
                "timestamp": r.get("timestamp"),
                "latitude": r.get("lat"),
                "longitude": r.get("lon"),
                "zone_id": r.get("zone_id"),
                "delayed_label": r.get("delayed_label"),
            }
            features = r.get("features", {})
            if isinstance(features, dict):
                row.update(features)
            rows.append(row)

        return pd.DataFrame(rows)

    def get_window_size(self, zone_id: str) -> int:
        """Get the current number of events buffered in the specified zone window."""
        key = f"{self.key_prefix}{zone_id}"
        return int(self.client.llen(key))

    def get_all_active_zones(self) -> List[str]:
        """Scan and return all unique spatial zone IDs currently buffered in Redis."""
        pattern = f"{self.key_prefix}*"
        try:
            keys = list(self.client.scan_iter(match=pattern))
        except Exception:
            keys = self.client.keys(pattern)

        prefix_len = len(self.key_prefix)
        zones = set()
        for k in keys:
            k_str = k if isinstance(k, str) else k.decode("utf-8")
            zone_id = k_str[prefix_len:]
            if zone_id:
                zones.add(zone_id)

        return sorted(list(zones))

    def clear_zone_window(self, zone_id: str) -> bool:
        """Delete all buffered events for a specific spatial zone."""
        key = f"{self.key_prefix}{zone_id}"
        return bool(self.client.delete(key))

    def clear_all_windows(self) -> int:
        """Utility method to reset all sliding window buffers across all zones."""
        pattern = f"{self.key_prefix}*"
        try:
            keys = list(self.client.scan_iter(match=pattern))
        except Exception:
            keys = self.client.keys(pattern)

        if not keys:
            return 0

        return int(self.client.delete(*keys))
