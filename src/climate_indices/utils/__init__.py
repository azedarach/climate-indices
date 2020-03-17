"""
Provides helper routines for CMIP5 DBN study.
"""


from __future__ import absolute_import


import numpy as np
import pandas as pd
import xarray as xr

from .defaults import get_coordinate_standard_name
from .eofs import eofs, reofs
from .validation import (check_array_shape, check_base_period, check_data_array,
                         check_fixed_missing_values, check_unit_axis_sums,
                         detect_frequency, get_valid_variables,
                         has_fixed_missing_values, is_daily_data, is_dask_array,
                         is_data_array, is_dataset, is_monthly_data,
                         is_xarray_object)


def downsample_data(da, frequency=None, time_name=None):
    """Perform down-sampling of data."""

    if frequency is None:
        return da

    if frequency not in ('daily', 'monthly'):
        raise ValueError('Unrecognized down-sampling frequency %r' %
                         frequency)

    if time_name is None:
        time_name = get_coordinate_standard_name(da, 'time')

    current_frequency = pd.infer_freq(da[time_name].values[:3])
    target_frequency = '1D' if frequency == 'daily' else '1MS'

    current_timestep = (pd.to_datetime(da[time_name].values[0]) +
                        pd.tseries.frequencies.to_offset(current_frequency))
    target_timestep = (pd.to_datetime(da[time_name].values[0]) +
                       pd.tseries.frequencies.to_offset(target_frequency))

    if target_timestep < current_timestep:
        raise ValueError('Downsampling frequency appears to be higher'
                         ' than current frequency')

    return da.resample({time_name: target_frequency}).mean(time_name)


def select_latlon_box(data, lat_bounds, lon_bounds,
                      lat_name=None, lon_name=None):
    """Select data in given latitude-longitude box."""

    if lat_name is None:
        lat_name = get_coordinate_standard_name(data, 'lat')

    if lon_name is None:
        lon_name = get_coordinate_standard_name(data, 'lon')

    if len(lat_bounds) != 2:
        raise ValueError('Latitude bounds must be a list of length 2')

    if len(lon_bounds) != 2:
        raise ValueError('Longitude bounds must be a list of length 2')

    lat_bounds = np.array(lat_bounds)
    lon_bounds = np.array(lon_bounds)

    if data[lat_name].values[0] > data[lat_name].values[-1]:
        lat_bounds = lat_bounds[::-1]

    region_data = data.sel({lat_name : slice(lat_bounds[0], lat_bounds[1])})

    lon = data[lon_name]

    if np.any(lon_bounds < 0) & np.any(lon < 0):
        # Both bounds and original coordinates are given in convention
        # [-180, 180]

        if lon_bounds[0] > lon_bounds[1]:
            lon_mask = (((lon >= lon_bounds[0]) & (lon <= 180)) |
                        ((lon >= -180) & (lon <= lon_bounds[1])))
        else:
            lon_mask = (lon >= lon_bounds[0]) & (lon <= lon_bounds[1])

    elif np.any(lon_bounds < 0) & ~np.any(lon < 0):
        # Bounds are given in convention [-180, 180] but data is in
        # convention [0, 360] -- convert bounds to [0, 360]
        lon_bounds = np.where(lon_bounds < 0, lon_bounds + 360, lon_bounds)

        if lon_bounds[0] > lon_bounds[1]:
            lon_mask = (((lon >= lon_bounds[0]) & (lon <= 360)) |
                        ((lon >= 0) & (lon <= lon_bounds[1])))
        else:
            lon_mask = (lon >= lon_bounds[0]) & (lon <= lon_bounds[1])

    elif ~np.any(lon_bounds < 0) & np.any(lon < 0):
        # Bounds are given in convention [0, 360] but data is in
        # convention [-180, 180] -- convert bounds to [-180, 180]
        lon_bounds = np.where(lon_bounds > 180, lon_bounds - 360, lon_bounds)

        if lon_bounds[0] > lon_bounds[1]:
            lon_mask = (((lon >= lon_bounds[0]) & (lon <= 180)) |
                        ((lon >= -180) & (lon <= lon_bounds[1])))
        else:
            lon_mask = (lon >= lon_bounds[0]) & (lon <= lon_bounds[1])

    else:
        # Bounds and data are given in convention [0, 360]

        if lon_bounds[0] > lon_bounds[1]:
            lon_mask = (((lon >= lon_bounds[0]) & (lon <= 360)) |
                        ((lon >= 0) & (lon <= lon_bounds[1])))
        else:
            lon_mask = (lon >= lon_bounds[0]) & (lon <= lon_bounds[1])

    region_data = region_data.where(lon_mask, drop=True)

    return region_data


def meridional_mean(data, lat_bounds=None, latitude_weight=False,
                    lat_name=None):
    """Calculate meridional mean between latitude bounds."""

    if lat_name is None:
        lat_name = get_coordinate_standard_name(data, 'lat')

    if latitude_weight:
        weights = np.cos(np.deg2rad(data[lat_name]))
        data = data * weights

    lat_bounds = sorted(lat_bounds)
    if data[lat_name][0] > data[lat_name][-1]:
        lat_bounds = lat_bounds[::-1]

    lat_slice = slice(lat_bounds[0], lat_bounds[1])

    return data.sel({lat_name : lat_slice}).mean(dim=[lat_name])


def standardized_anomalies(da, base_period=None, standardize_by=None,
                           time_name=None):
    """Calculate standardized anomalies."""

    if time_name is None:
        time_name = get_coordinate_standard_name(da, 'time')

    base_period = check_base_period(da, base_period=base_period,
                                    time_name=time_name)

    base_period_da = da.where(
        (da[time_name] >= base_period[0]) &
        (da[time_name] <= base_period[1]), drop=True)

    if standardize_by == 'dayofyear':
        base_period_groups = base_period_da[time_name].dt.dayofyear
        groups = da[time_name].dt.dayofyear
    elif standardize_by == 'month':
        base_period_groups = base_period_da[time_name].dt.month
        groups = da[time_name].dt.month
    elif standardize_by == 'season':
        base_period_groups = base_period_da[time_name].dt.season
        groups = da[time_name].dt.season
    else:
        base_period_groups = None
        groups = None

    if base_period_groups is not None:

        clim_mean = base_period_da.groupby(base_period_groups).mean(time_name)
        clim_std = base_period_da.groupby(base_period_groups).std(time_name)

        std_anom = xr.apply_ufunc(
            lambda x, m, s: (x - m) / s, da.groupby(groups),
            clim_mean, clim_std, dask='allowed')

    else:

        clim_mean = base_period_da.mean(time_name)
        clim_std = base_period_da.std(time_name)

        std_anom = ((da - clim_mean) / clim_std)

    return std_anom
