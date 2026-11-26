"""Kafka Telemetry Replay Producer.

Streams timestamped geospatial weather observations from CSV into the
Kafka 'raw-telemetry' topic, partitioned by zone_id. Supports configurable
replay speed, simulation modes (normal, monsoon, transition), and on-the-fly
drift amplification for stress testing.
"""

import argparse
import json
import logging
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path for direct CLI script execution
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from kafka import KafkaProducer
import pandas as pd
import yaml

from services.common.schemas import TelemetryEvent

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("telemetry_producer")


def load_settings(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from config/settings.yaml if available."""
    if config_path:
        path = Path(config_path)
    else:
        project_root = Path(__file__).resolve().parent.parent.parent
        path = project_root / "config" / "settings.yaml"

    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning("Could not read settings from %s: %s", path, e)
    return {}


class TelemetryProducer:
    """Streams validated environmental telemetry events to Apache Kafka."""

    def __init__(
        self,
        bootstrap_servers: str = "localhost:29092",
        topic: str = "raw-telemetry",
        csv_path: Optional[str] = None,
        rate: float = 5.0,
        mode: str = "transition",
        drift_multiplier: float = 1.0,
        max_events: Optional[int] = None,
        loop: bool = False,
    ):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.rate = max(0.0, float(rate))
        self.mode = mode.lower()
        self.drift_multiplier = float(drift_multiplier)
        self.max_events = int(max_events) if max_events is not None else None
        self.loop = loop

        # Resolve CSV path robustly against current directory or project root
        project_root = Path(__file__).resolve().parent.parent.parent
        if csv_path:
            p = Path(csv_path)
            if not p.is_file() and (project_root / csv_path).is_file():
                self.csv_path = project_root / csv_path
            else:
                self.csv_path = p
        else:
            self.csv_path = project_root / "data" / "processed" / "stream_simulation.csv"

        self.running = True
        self.total_emitted = 0
        self.delivery_errors = 0

        # Initialize Kafka Producer
        logger.info("Initializing KafkaProducer with bootstrap_servers=%s", self.bootstrap_servers)
        self.producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if isinstance(k, str) else k,
            acks="all",
            retries=3,
            api_version=(3, 7, 0),
        )

        # Setup graceful signal handlers
        self._register_signals()

    def _register_signals(self):
        """Register SIGINT/SIGTERM handlers if running on the main thread."""
        if threading.current_thread() is threading.main_thread():
            try:
                signal.signal(signal.SIGINT, self._handle_signal)
                signal.signal(signal.SIGTERM, self._handle_signal)
            except (ValueError, AttributeError) as e:
                logger.debug("Signal registration skipped: %s", e)

    def _handle_signal(self, sig, frame):
        logger.info("Caught signal %s. Initiating graceful producer shutdown...", sig)
        self.running = False

    def _on_delivery_error(self, exc: Exception):
        self.delivery_errors += 1
        logger.error("Message delivery failed: %s", exc)

    def load_dataset(self) -> pd.DataFrame:
        """Load and prepare rows from CSV based on configured simulation mode."""
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Telemetry source CSV not found at {self.csv_path}")

        logger.info("Reading telemetry records from %s", self.csv_path)
        df = pd.read_csv(self.csv_path)

        if df.empty:
            raise ValueError(f"Telemetry source CSV is empty: {self.csv_path}")

        logger.info(
            "Loaded %d raw records. Preparing dataset for mode='%s' (drift_multiplier=%.2f)",
            len(df),
            self.mode,
            self.drift_multiplier,
        )
        return df

    def _build_event(
        self,
        row: pd.Series,
        apply_drift: bool = False,
    ) -> TelemetryEvent:
        """Convert a DataFrame row into a validated TelemetryEvent."""
        rainfall = float(row["rainfall_mm"])
        soil_moist = float(row["soil_moisture"])
        temp = float(row["temperature_c"])
        humidity = float(row["humidity_pct"])
        wind = float(row["wind_speed_kmh"])

        if apply_drift and self.drift_multiplier != 1.0:
            rainfall = round(rainfall * self.drift_multiplier, 2)
            # Bound soil moisture physically in [0.0, 1.0]
            soil_moist = round(
                min(1.0, max(0.0, soil_moist * (1.0 + (self.drift_multiplier - 1.0) * 0.2))),
                3,
            )

        features = {
            "rainfall_mm": rainfall,
            "soil_moisture": soil_moist,
            "temperature_c": temp,
            "humidity_pct": humidity,
            "wind_speed_kmh": wind,
        }

        lat = float(row["latitude"]) if "latitude" in row else float(row["lat"])
        lon = float(row["longitude"]) if "longitude" in row else float(row["lon"])
        zone_id = str(row["zone_id"])
        timestamp = str(row["timestamp"])

        delayed_label = (
            int(row["hazard_label"])
            if ("hazard_label" in row and not pd.isna(row["hazard_label"]))
            else None
        )

        return TelemetryEvent(
            timestamp=timestamp,
            lat=lat,
            lon=lon,
            zone_id=zone_id,
            features=features,
            delayed_label=delayed_label,
        )

    def prepare_event_stream(self, df: pd.DataFrame) -> List[tuple[pd.Series, bool]]:
        """Slice and partition DataFrame rows with drift flags based on simulation mode."""
        total_rows = len(df)
        max_rows = self.max_events if self.max_events is not None else total_rows

        if self.mode == "normal":
            # Select non-hazard / pre-drift rows
            if "hazard_label" in df.columns:
                normal_df = df[df["hazard_label"] == 0].reset_index(drop=True)
                if len(normal_df) < max_rows:
                    normal_df = df.iloc[:max_rows].reset_index(drop=True)
                df_sliced = normal_df.iloc[:max_rows]
            else:
                df_sliced = df.iloc[:max_rows]
            return [(row, False) for _, row in df_sliced.iterrows()]

        elif self.mode == "monsoon":
            # Select drifted monsoon rows with drift multiplier applied
            df_sliced = df.iloc[:max_rows]
            return [(row, True) for _, row in df_sliced.iterrows()]

        else:  # "transition" mode (default)
            # First half normal, second half monsoon with drift amplification
            df_sliced = df.iloc[:max_rows]
            n_events = len(df_sliced)
            midpoint = n_events // 2

            stream_items = []
            for idx, (_, row) in enumerate(df_sliced.iterrows()):
                is_drift_period = idx >= midpoint
                stream_items.append((row, is_drift_period))

            return stream_items

    def run(self) -> int:
        """Execute the streaming replay loop.

        Returns:
            Total number of events published.
        """
        df = self.load_dataset()
        sleep_interval = 1.0 / self.rate if self.rate > 0 else 0.0

        logger.info(
            "Starting Kafka telemetry replay -> Topic: '%s' | Rate: %.1f evt/s | Max: %s",
            self.topic,
            self.rate,
            self.max_events or "Unlimited",
        )

        start_time = time.time()
        last_log_time = start_time

        try:
            while self.running:
                event_tuples = self.prepare_event_stream(df)

                for row, apply_drift in event_tuples:
                    if not self.running:
                        break

                    if self.max_events is not None and self.total_emitted >= self.max_events:
                        self.running = False
                        break

                    send_start = time.time()

                    # Construct validated event
                    event = self._build_event(row, apply_drift=apply_drift)

                    # Send to Kafka keyed by zone_id
                    future = self.producer.send(
                        topic=self.topic,
                        key=event.zone_id,
                        value=event.to_dict(),
                    )
                    future.add_errback(self._on_delivery_error)
                    self.total_emitted += 1

                    # Periodic throughput and status report
                    now = time.time()
                    if self.total_emitted % 50 == 0 or (now - last_log_time >= 5.0):
                        elapsed = now - start_time
                        cur_rate = self.total_emitted / elapsed if elapsed > 0 else 0.0
                        logger.info(
                            "Emitted %d events | Rate: %.1f evt/s | Zone: %s | Rain: %.1fmm | Drift: %s",
                            self.total_emitted,
                            cur_rate,
                            event.zone_id,
                            event.features.get("rainfall_mm", 0.0),
                            apply_drift,
                        )
                        sys.stdout.flush()
                        last_log_time = now

                    # Sleep to enforce configured rate
                    if sleep_interval > 0:
                        spent = time.time() - send_start
                        time_to_sleep = sleep_interval - spent
                        if time_to_sleep > 0:
                            time.sleep(time_to_sleep)

                if not self.loop or not self.running:
                    break

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt detected. Stopping streaming loop...")
        finally:
            logger.info("Flushing producer buffers...")
            self.producer.flush(timeout=10)
            self.producer.close(timeout=5)

            total_elapsed = time.time() - start_time
            avg_rate = self.total_emitted / total_elapsed if total_elapsed > 0 else 0.0
            logger.info(
                "Streaming finished. Total emitted: %d events in %.2fs (avg %.2f evt/s). Errors: %d",
                self.total_emitted,
                total_elapsed,
                avg_rate,
                self.delivery_errors,
            )
            sys.stdout.flush()

        return self.total_emitted


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    settings = load_settings()
    default_kafka_servers = (
        settings.get("streaming", {}).get("kafka", {}).get("bootstrap_servers", "localhost:29092")
    )
    default_topic = (
        settings.get("streaming", {})
        .get("kafka", {})
        .get("topics", {})
        .get("raw_telemetry", "raw-telemetry")
    )
    default_csv = (
        settings.get("data", {}).get("stream_simulation_path", "data/processed/stream_simulation.csv")
    )

    parser = argparse.ArgumentParser(
        description="MLOps Weather Telemetry Replay Producer (Apache Kafka)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=5.0,
        help="Events per second to publish",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["normal", "monsoon", "transition"],
        default="transition",
        help="Simulation profile: normal (pre-drift), monsoon (drifted), transition (normal -> monsoon)",
    )
    parser.add_argument(
        "--drift-multiplier",
        type=float,
        default=1.0,
        help="Multiplier to amplify rainfall and soil moisture for drift stress testing",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Maximum total events to emit before exiting (default: stream full file)",
    )
    parser.add_argument(
        "--bootstrap-servers",
        type=str,
        default=default_kafka_servers,
        help="Kafka bootstrap server address",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=default_topic,
        help="Kafka destination topic name",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=default_csv,
        help="Path to processed telemetry simulation CSV",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Continuously repeat dataset replay when EOF is reached",
    )
    return parser.parse_args()


def main():
    """Main entrypoint for CLI execution."""
    args = parse_args()
    producer = TelemetryProducer(
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
        csv_path=args.csv_path,
        rate=args.rate,
        mode=args.mode,
        drift_multiplier=args.drift_multiplier,
        max_events=args.max_events,
        loop=args.loop,
    )
    sys.exit(0 if producer.run() >= 0 else 1)


if __name__ == "__main__":
    main()
