"""Provides helper routines for working with lat-lon grids."""

# License: MIT

import collections.abc
import numbers
from typing import Any, Optional, Sequence

import numpy as np

from .defaults import get_lat_name, get_lon_name


def wrap_coordinate(coord: Any, base: float, period: float) -> Any:
    """Wrap coordinate to interval [base, base + period).

    Parameters
    ----------
    coord : array-like
        Array of coordinate values to wrap.

    base : float
        Base point for wrapped coordinate.

    period : float
        Width of wrapped interval.

    Returns
    -------
    wrapped : array-like
        Array of wrapped coordinate values.
    """
    if period == 0.0:
        raise ValueError('Period must not be zero')

    def _wrap(x):
        return ((x - base + 2 * period) % period) + base

    if isinstance(coord, numbers.Number):
        return _wrap(coord)

    if isinstance(coord, collections.abc.Sequence):
        return [_wrap(x) for x in coord]

    original_dtype = coord.dtype

    wrapped = _wrap(coord.astype(np.float64))

    return wrapped.astype(original_dtype)


def _get_longitude_mask(lon: Any, lon_bounds: Sequence) -> Any:
    """Get mask for specified longitudes."""
    # Normalize longitudes to be in the interval [0, 360)
    lon = wrap_coordinate(lon, base=0.0, period=360.0)
    lon_bounds = wrap_coordinate(lon_bounds, base=0.0, period=360.0)

    if lon_bounds[0] > lon_bounds[1]:
        return (((lon >= lon_bounds[0]) & (lon <= 360)) |
                ((lon >= 0) & (lon <= lon_bounds[1])))

    return (lon >= lon_bounds[0]) & (lon <= lon_bounds[1])


def select_latlon_box(data: Any, lat_bounds: Sequence,
                      lon_bounds: Sequence, drop: bool = False,
                      lat_name: Optional[str] = None,
                      lon_name: Optional[str] = None):
    """Select data in given latitude-longitude box."""
    lat_name = lat_name if lat_name is not None else get_lat_name(data)
    lon_name = lon_name if lon_name is not None else get_lon_name(data)

    if len(lat_bounds) != 2:
        raise ValueError('Latitude bounds must be a list of length 2')

    if len(lon_bounds) != 2:
        raise ValueError('Longitude bounds must be a list of length 2')

    lat_bounds = np.array(sorted(lat_bounds))
    lon_bounds = np.array(lon_bounds)

    region_data = data.where((data[lat_name] >= lat_bounds[0]) &
                             (data[lat_name] <= lat_bounds[1]), drop=drop)

    lon_mask = _get_longitude_mask(data[lon_name], lon_bounds=lon_bounds)

    region_data = region_data.where(lon_mask, drop=drop)

    return region_data
