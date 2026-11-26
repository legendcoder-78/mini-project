"""Dataset Acquisition & Realistic Telemetry Generator.

Fetches historical hourly weather data via the free Open-Meteo Archive API
or falls back cleanly to generating a realistic, deterministic multi-season
environmental telemetry dataset (~15,000 rows).
"""

import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import requests

# Ensure project root is in sys.path for direct CLI script execution
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml_core.preprocessing.geo_encoder import load_spatial_zones, _normalize_zones_list

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_synthetic_telemetry(
    num_records: int = 15000,
    seed: int = 42,
    zones_config: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
) -> pd.DataFrame:
    """Generate deterministic, realistic multi-season environmental telemetry.

    Features realistic seasonal dynamics:
    - Pre-monsoon (Jan-May): low rainfall, moderate/low soil moisture, higher temps, low hazard rates.
    - Monsoon (Jun-Sep): high rainfall spikes, saturated soil, high humidity, elevated hazard rates.
    - Post-monsoon (Oct-Dec): retreating rains, moderate soil moisture.

    Args:
        num_records: Total number of rows to generate (~15,000).
        seed: Random seed for deterministic reproducibility.
        zones_config: Spatial zones config. If None, loaded from default config.

    Returns:
        DataFrame containing:
            [timestamp, latitude, longitude, rainfall_mm, soil_moisture,
             temperature_c, humidity_pct, wind_speed_kmh, hazard_label]
    """
    rng = np.random.default_rng(seed)

    if zones_config is None:
        zones_config = load_spatial_zones()
    zones = _normalize_zones_list(zones_config)

    num_zones = len(zones)
    records_per_zone = num_records // num_zones
    remainder = num_records % num_zones

    dfs: List[pd.DataFrame] = []

    # Timeline spanning multi-season transitions: 2023-01-01 to 2024-06-30
    base_timestamps = pd.date_range(
        start="2023-01-01 00:00:00",
        end="2024-06-30 23:00:00",
        periods=records_per_zone + (1 if remainder > 0 else 0),
        tz="UTC",
    ).floor("s")

    for i, zone in enumerate(zones):
        zone_count = records_per_zone + (1 if i < remainder else 0)
        ts_subset = base_timestamps[:zone_count]

        # Coordinates sampled within zone bounding box
        min_lat, max_lat = float(zone["min_lat"]), float(zone["max_lat"])
        min_lon, max_lon = float(zone["min_lon"]), float(zone["max_lon"])

        # Fixed coordinates with slight variation
        lat_base = (min_lat + max_lat) / 2.0
        lon_base = (min_lon + max_lon) / 2.0
        latitudes = np.clip(
            lat_base + rng.normal(0, 0.05, size=zone_count), min_lat, max_lat
        ).round(4)
        longitudes = np.clip(
            lon_base + rng.normal(0, 0.05, size=zone_count), min_lon, max_lon
        ).round(4)

        months = ts_subset.month.to_numpy()
        hours = ts_subset.hour.to_numpy()

        # Season classification:
        # Pre-monsoon: Jan-May (months 1-5)
        # Southwest Monsoon: Jun-Sep (months 6-9)
        # Northeast / Retreating: Oct-Dec (months 10-12)
        is_monsoon = (months >= 6) & (months <= 9)
        is_pre_monsoon = months <= 5
        is_post_monsoon = months >= 10

        # Base temperature: diurnal variation + seasonal shift
        # Hotter in pre-monsoon (March-May), cooler during monsoon
        temp_season_offset = np.where(
            is_pre_monsoon,
            4.0 + 2.0 * np.sin(np.pi * (months - 1) / 5),
            np.where(is_monsoon, -3.5, 0.0),
        )
        diurnal_temp = 4.5 * np.sin(2 * np.pi * (hours - 9) / 24)
        temperature = 25.0 + temp_season_offset + diurnal_temp + rng.normal(0, 1.2, size=zone_count)
        temperature = np.clip(temperature, 14.0, 42.0).round(2)

        # Rainfall modeling (mm/hr):
        # Pre-monsoon: ~95% zero, occasional light convective showers (0-8 mm)
        rain_prob = np.where(is_monsoon, 0.48, np.where(is_post_monsoon, 0.15, 0.04))
        rain_occurred = rng.random(size=zone_count) < rain_prob

        # Exponential / Gamma distributed rainfall intensity
        rain_intensity = np.where(
            is_monsoon,
            rng.exponential(scale=14.0, size=zone_count) + rng.choice([0, 35, 65], p=[0.88, 0.09, 0.03], size=zone_count),
            rng.exponential(scale=2.5, size=zone_count),
        )
        rainfall = np.where(rain_occurred, rain_intensity, 0.0)
        rainfall = np.clip(rainfall, 0.0, 150.0).round(2)

        # Soil moisture (0.05 to 0.95 volumetric fraction):
        # Accumulates during rainfall, decays during dry spells
        soil_moisture = np.zeros(zone_count, dtype=float)
        current_sm = 0.22 if is_pre_monsoon[0] else 0.65
        for t in range(zone_count):
            if rainfall[t] > 0:
                current_sm += (rainfall[t] / 80.0)
            else:
                current_sm *= 0.995  # slow drying
            # Seasonal baseline pull
            seasonal_pull = 0.70 if is_monsoon[t] else (0.22 if is_pre_monsoon[t] else 0.45)
            current_sm = 0.92 * current_sm + 0.08 * seasonal_pull
            current_sm = np.clip(current_sm + rng.normal(0, 0.005), 0.08, 0.95)
            soil_moisture[t] = round(float(current_sm), 4)

        # Humidity (%): inversely related to temperature, heightened by rainfall
        base_humidity = np.where(is_monsoon, 88.0, np.where(is_pre_monsoon, 55.0, 70.0))
        humidity = (
            base_humidity
            - 1.2 * (temperature - 25.0)
            + 0.35 * rainfall
            + rng.normal(0, 3.5, size=zone_count)
        )
        humidity = np.clip(humidity, 20.0, 100.0).round(2)

        # Wind speed (km/h): higher during monsoon convective squalls
        base_wind = np.where(is_monsoon, 24.0, 11.0)
        squall_boost = np.where(rainfall > 25.0, rng.uniform(15.0, 35.0, size=zone_count), 0.0)
        wind_speed = base_wind + squall_boost + rng.normal(0, 3.0, size=zone_count)
        wind_speed = np.clip(wind_speed, 1.0, 95.0).round(2)

        # Hazard label (0 = Normal, 1 = Extreme Hazard):
        # Deterministic physical criterion representing flash flood / cloudburst / landslide trigger:
        # High rainfall (> 35mm/hr), OR saturated soil with heavy rainfall (> 20mm & soil > 0.65),
        # OR extreme convective squall (wind > 50km/h & rain > 15mm)
        hazard_condition = (
            (rainfall >= 25.0)
            | ((rainfall >= 12.0) & (soil_moisture >= 0.55))
            | ((wind_speed >= 40.0) & (rainfall >= 8.0))
            | (rainfall >= 45.0)
        )
        hazard_label = np.where(hazard_condition, 1, 0).astype(int)

        zone_df = pd.DataFrame(
            {
                "timestamp": [ts.strftime("%Y-%m-%dT%H:%M:%SZ") for ts in ts_subset],
                "latitude": latitudes,
                "longitude": longitudes,
                "rainfall_mm": rainfall,
                "soil_moisture": soil_moisture,
                "temperature_c": temperature,
                "humidity_pct": humidity,
                "wind_speed_kmh": wind_speed,
                "hazard_label": hazard_label,
            }
        )
        dfs.append(zone_df)

    combined_df = pd.concat(dfs, ignore_index=True)
    # Sort chronologically
    combined_df["dt_temp"] = pd.to_datetime(combined_df["timestamp"], utc=True)
    combined_df = combined_df.sort_values(by="dt_temp").drop(columns=["dt_temp"]).reset_index(drop=True)
    return combined_df


def fetch_open_meteo_archive(
    zones_config: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
    timeout_sec: float = 6.0,
) -> Optional[pd.DataFrame]:
    """Attempt to pull historical hourly weather data via free Open-Meteo Archive API."""
    if zones_config is None:
        zones_config = load_spatial_zones()
    zones = _normalize_zones_list(zones_config)

    records: List[pd.DataFrame] = []
    base_url = "https://archive-api.open-meteo.com/v1/archive"

    logger.info("Attempting to query Open-Meteo Archive API...")
    for zone in zones:
        lat = (float(zone["min_lat"]) + float(zone["max_lat"])) / 2.0
        lon = (float(zone["min_lon"]) + float(zone["max_lon"])) / 2.0

        params = {
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "precipitation,soil_moisture_0_to_7cm,temperature_2m,relative_humidity_2m,wind_speed_10m",
            "timezone": "UTC",
        }

        try:
            resp = requests.get(base_url, params=params, timeout=timeout_sec)
            if resp.status_code != 200:
                logger.warning("Open-Meteo returned status %s for zone %s", resp.status_code, zone["zone_id"])
                return None
            data = resp.json()
            if "hourly" not in data or "time" not in data["hourly"]:
                return None

            hourly = data["hourly"]
            times = hourly["time"]
            precip = np.nan_to_num(hourly.get("precipitation", [0.0] * len(times)), nan=0.0)
            soil = np.nan_to_num(hourly.get("soil_moisture_0_to_7cm", [0.3] * len(times)), nan=0.3)
            temp = np.nan_to_num(hourly.get("temperature_2m", [25.0] * len(times)), nan=25.0)
            hum = np.nan_to_num(hourly.get("relative_humidity_2m", [70.0] * len(times)), nan=70.0)
            wind = np.nan_to_num(hourly.get("wind_speed_10m", [10.0] * len(times)), nan=10.0)

            # Format timestamp to ISO 8601 UTC
            iso_times = [pd.to_datetime(t, utc=True).isoformat() for t in times]
            n_rows = len(iso_times)

            # Calibrated hazard label for reanalysis grids (0.25 deg grid cell averages)
            hazard = np.where(
                (precip >= 8.0) | ((precip >= 4.0) & (soil >= 0.40)) | ((wind >= 25.0) & (precip >= 3.0)),
                1,
                0,
            ).astype(int)

            zone_df = pd.DataFrame(
                {
                    "timestamp": iso_times,
                    "latitude": np.full(n_rows, round(lat, 4)),
                    "longitude": np.full(n_rows, round(lon, 4)),
                    "rainfall_mm": precip.round(2),
                    "soil_moisture": soil.round(4),
                    "temperature_c": temp.round(2),
                    "humidity_pct": hum.round(2),
                    "wind_speed_kmh": wind.round(2),
                    "hazard_label": hazard,
                }
            )
            records.append(zone_df)
        except Exception as e:
            logger.warning("Open-Meteo API request failed: %s. Falling back to synthetic telemetry.", str(e))
            return None

    if records:
        combined = pd.concat(records, ignore_index=True)
        combined["dt_temp"] = pd.to_datetime(combined["timestamp"])
        combined = combined.sort_values(by="dt_temp").drop(columns=["dt_temp"]).reset_index(drop=True)
        return combined

    return None


def fetch_or_generate_dataset(
    output_path: Optional[Union[str, Path]] = None,
    num_records: int = 15000,
    try_live_api: bool = True,
    seed: int = 42,
    zones_config: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
) -> pd.DataFrame:
    """Fetch live data from Open-Meteo or fall back cleanly to synthetic telemetry generator.

    Args:
        output_path: File path to save output CSV (e.g. data/raw/weather_telemetry_raw.csv).
        num_records: Number of records to generate if synthetic (~15,000).
        try_live_api: Whether to attempt querying the live Open-Meteo API first.
        seed: Random seed for deterministic synthetic dataset.
        zones_config: Optional custom zones configuration.

    Returns:
        DataFrame with raw environmental telemetry.
    """
    if output_path is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        output_path = project_root / "data" / "raw" / "weather_telemetry_raw.csv"
    else:
        output_path = Path(output_path)

    df: Optional[pd.DataFrame] = None

    if try_live_api:
        try:
            df = fetch_open_meteo_archive(zones_config=zones_config)
            if df is not None and len(df) >= 1000:
                logger.info("Successfully fetched %d records from Open-Meteo Archive API", len(df))
        except Exception as e:
            logger.warning("Live API fetch encountered exception: %s", str(e))
            df = None

    if df is None:
        logger.info("Generating %d deterministic synthetic records (seed=%d)...", num_records, seed)
        df = generate_synthetic_telemetry(num_records=num_records, seed=seed, zones_config=zones_config)
        logger.info("Generated %d synthetic telemetry rows", len(df))

    # Ensure target directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved raw telemetry dataset to %s", output_path)

    return df


if __name__ == "__main__":
    fetch_or_generate_dataset()
