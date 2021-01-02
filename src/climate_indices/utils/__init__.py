"""Provides utility routines and helpers."""

# License: MIT

from .defaults import (
    get_coordinate_standard_name,
    get_depth_name,
    get_lat_name,
    get_lon_name,
    get_time_name,
)
from .eofs import calc_truncated_svd
from .latlon import select_latlon_box, wrap_coordinate


__all__ = [
    'calc_truncated_svd',
    'get_coordinate_standard_name',
    'get_depth_name',
    'get_lat_name',
    'get_lon_name',
    'get_time_name',
    'select_latlon_box',
    'wrap_coordinate',
]
