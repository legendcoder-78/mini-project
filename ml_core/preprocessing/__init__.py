"""Preprocessing module for geospatial encoding, cleaning, and data fetching."""

from ml_core.preprocessing.geo_encoder import assign_spatial_zone, encode_geospatial_zones

__all__ = [
    "assign_spatial_zone",
    "encode_geospatial_zones",
]
