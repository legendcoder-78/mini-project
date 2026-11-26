"""Geospatial Encoder for mapping spatial coordinates to discrete zone IDs."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


def load_spatial_zones(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load spatial zones from JSON configuration file."""
    if config_path is None:
        # Default to config/spatial_zones.json relative to repository root
        project_root = Path(__file__).resolve().parent.parent.parent
        config_path = project_root / "config" / "spatial_zones.json"
    else:
        config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Spatial zones configuration not found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_zones_list(zones_config: Union[Dict[str, Any], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Normalize various zone config schemas into a standardized list of zone dictionaries."""
    if isinstance(zones_config, list):
        return zones_config
    if isinstance(zones_config, dict):
        if "zones" in zones_config and isinstance(zones_config["zones"], list):
            return zones_config["zones"]
        # In case the dict is keyed by zone_id: {"ZONE_KODAGU": {"min_lat": ...}}
        normalized = []
        for zone_id, props in zones_config.items():
            if isinstance(props, dict):
                item = {"zone_id": zone_id}
                item.update(props)
                normalized.append(item)
        if normalized:
            return normalized
    raise ValueError(f"Invalid zones_config structure: {type(zones_config)}")


def assign_spatial_zone(
    lat: float,
    lon: float,
    zones_config: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    default_fallback: str = "OUT_OF_BOUNDS",
) -> str:
    """Deterministically map a (lat, lon) coordinate pair to a zone_id.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        zones_config: Spatial zones configuration dict or list. If None, loaded from default config.
        default_fallback: Fallback zone name if coordinate does not fall into any defined bounding box.

    Returns:
        zone_id string or default_fallback ("OUT_OF_BOUNDS").
    """
    if pd.isna(lat) or pd.isna(lon):
        return default_fallback

    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (ValueError, TypeError):
        return default_fallback

    if zones_config is None:
        zones_config = load_spatial_zones()

    zones_list = _normalize_zones_list(zones_config)

    for zone in zones_list:
        min_lat = float(zone["min_lat"])
        max_lat = float(zone["max_lat"])
        min_lon = float(zone["min_lon"])
        max_lon = float(zone["max_lon"])

        # Inclusive check on boundaries
        if min_lat <= lat_f <= max_lat and min_lon <= lon_f <= max_lon:
            return zone["zone_id"]

    return default_fallback


def encode_geospatial_zones(
    df: pd.DataFrame,
    zones_config: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    target_col: str = "zone_id",
    default_fallback: str = "OUT_OF_BOUNDS",
) -> pd.DataFrame:
    """Vectorized geospatial zone assignment for a pandas DataFrame.

    Args:
        df: Input DataFrame containing latitude and longitude columns.
        zones_config: Spatial zones configuration dict or list. If None, loaded from default config.
        lat_col: Column name representing latitude.
        lon_col: Column name representing longitude.
        target_col: Name of the resulting column to store assigned zone_id.
        default_fallback: Fallback label for points outside defined zones.

    Returns:
        DataFrame copy with target_col appended or updated.
    """
    if lat_col not in df.columns or lon_col not in df.columns:
        raise KeyError(f"Input DataFrame must contain '{lat_col}' and '{lon_col}' columns")

    df_out = df.copy()
    if df_out.empty:
        df_out[target_col] = pd.Series(dtype=str)
        return df_out

    if zones_config is None:
        zones_config = load_spatial_zones()

    zones_list = _normalize_zones_list(zones_config)

    # Initialize with default fallback
    zone_assignments = np.full(len(df_out), default_fallback, dtype=object)

    latitudes = pd.to_numeric(df_out[lat_col], errors="coerce").to_numpy()
    longitudes = pd.to_numeric(df_out[lon_col], errors="coerce").to_numpy()

    # Iterate through zones and apply masks
    for zone in zones_list:
        min_lat = float(zone["min_lat"])
        max_lat = float(zone["max_lat"])
        min_lon = float(zone["min_lon"])
        max_lon = float(zone["max_lon"])
        zone_id = zone["zone_id"]

        mask = (
            (latitudes >= min_lat)
            & (latitudes <= max_lat)
            & (longitudes >= min_lon)
            & (longitudes <= max_lon)
            & (zone_assignments == default_fallback)  # Keep first matching zone if overlap
        )
        zone_assignments[mask] = zone_id

    df_out[target_col] = zone_assignments
    return df_out
