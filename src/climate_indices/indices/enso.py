"""
Provides routines for calculating ENSO indices.
"""

from __future__ import absolute_import

import calendar
import numbers
import os

import geopandas as gp
import numpy as np
import regionmask as rm
import xarray as xr

from ..utils import (check_base_period, check_data_array,
                     detect_frequency, eofs, get_coordinate_standard_name,
                     select_latlon_box, standardized_anomalies)


INTEGER_TYPES = (numbers.Integral, np.integer)

MEI_REGION_SHP = os.path.join(
    os.path.dirname(__file__), 'ne_110m_mei_region.shp')


def _mean_region_anomaly(data, lat_bounds, lon_bounds, standardize=False,
                         standardize_index=False, base_period=None,
                         area_weight=False, lat_name=None, lon_name=None,
                         time_name=None):
    """Calculate mean anomaly in rectangular region.

    Parameters
    ----------
    data : xarray.DataArray
        Array containing variable values.

    lat_bounds : list, shape (2,)
        Latitude bounds of region.

    lon_bounds : list, shape (2,)
        Longitude bounds of region.

    standardize : bool
        If True, calculate standardized anomalies.

    standardize_index : bool
        If True, standardize index by mean and standard deviation within
        base period.

    base_period : list
        Earliest and latest time to use for standardization.

    area_weight : bool
        If True, weight grid points by cos(latitude) when
        calculating mean.

    lat_name : str
        Name of the latitude coordinate.

    lon_name : str
        Name of the longitude coordinate.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    index : xarray.DataArray
        Array containing the values of area averaged anomaly.
    """

    # Ensure that the data provided is a data array
    data = check_data_array(data)

    # Get coordinate names
    if lat_name is None:
        lat_name = get_coordinate_standard_name(data, 'lat')

    if lon_name is None:
        lon_name = get_coordinate_standard_name(data, 'lon')

    if time_name is None:
        time_name = get_coordinate_standard_name(data, 'time')

    base_period = check_base_period(data, base_period=base_period,
                                    time_name=time_name)

    input_frequency = detect_frequency(data, time_name=time_name)

    if input_frequency not in ('daily', 'monthly'):
        raise RuntimeError('Can only calculate index for daily or monthly data')

    region_data = select_latlon_box(
        data, lat_bounds=lat_bounds, lon_bounds=lon_bounds,
        lat_name=lat_name, lon_name=lon_name)

    if standardize:

        if input_frequency == 'daily':
            standardize_by = 'dayofyear'
        else:
            standardize_by = 'month'

        anom = standardized_anomalies(
            region_data, base_period=base_period,
            standardize_by=standardize_by, time_name=time_name)

    else:

        base_period_data = region_data.where(
            (region_data[time_name] >= base_period[0]) &
            (region_data[time_name] <= base_period[1]), drop=True)

        if input_frequency == 'daily':

            daily_means = base_period_data.groupby(
                base_period_data[time_name].dt.dayofyear).mean(time_name)

            anom = (region_data.groupby(region_data[time_name].dt.dayofyear) -
                    daily_means)

        else:

            monthly_means = base_period_data.groupby(
                base_period_data[time_name].dt.month).mean(time_name)

            anom = (region_data.groupby(region_data[time_name].dt.month) -
                    monthly_means)

    if area_weight:

        weights = np.cos(np.deg2rad(anom[lat_name]))
        anom = anom * weights

    index = anom.mean(dim=[lat_name, lon_name])

    if standardize_index:

        base_period_index = index.where(
            (index[time_name] >= base_period[0]) &
            (index[time_name] <= base_period[1]), drop=True)

        if input_frequency == 'daily':

            index_mean = base_period_index.groupby(
                base_period_index[time_name].dt.dayofyear).mean(time_name)
            index_std = base_period_index.groupby(
                base_period_index[time_name].dt.dayofyear).std(time_name)

            index = xr.apply_ufunc(
                lambda x, m, s: (x - m) / s,
                index.groupby(index[time_name].dt.dayofyear),
                index_mean, index_std, dask='allowed')

        else:

            index_mean = base_period_index.groupby(
                base_period_index[time_name].dt.month).mean(time_name)
            index_std = base_period_index.groupby(
                base_period_index[time_name].dt.month).std(time_name)

            index = xr.apply_ufunc(
                lambda x, m, s: (x - m) / s,
                index.groupby(index[time_name].dt.month),
                index_mean, index_std, dask='allowed')

    return index


def nino1(sst, frequency='monthly', base_period=None, standardize=False,
          lat_name=None, lon_name=None, time_name=None):
    """Calculate Nino1 index.

    Parameters
    ----------
    sst : xarray.DataArray
        Array containing SST values.

    frequency : str
        If given, downsample data to requested frequency.

    base_period : list
        Earliest and latest time to use for standardization.

    standardize : bool
        If True, standardize index using mean and standard deviation
        within the base period.

    lat_name : str
        Name of the latitude coordinate.

    lon_name : str
        Name of the longitude coordinate.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    index : xarray.DataArray
        Array containing the values of the Nino1 index.
    """

    lat_bounds = [-10.0, -5.0]
    lon_bounds = [-90.0, -80.0]

    # Ensure that the data provided is a data array
    sst = check_data_array(sst)

    # Get coordinate names
    if time_name is None:
        time_name = get_coordinate_standard_name(sst, 'time')

    if frequency is None:
        frequency = 'monthly'

    if frequency not in ('daily', 'monthly'):
        raise ValueError("Unsupported frequency '%r'" % frequency)

    input_frequency = detect_frequency(sst, time_name=time_name)

    if input_frequency not in ('daily', 'monthly'):
        raise RuntimeError('Can only calculate index for daily or monthly data')

    if input_frequency == 'daily' and frequency == 'monthly':

        sst = sst.resample({time_name : '1MS'}).mean()

    index = _mean_region_anomaly(
        sst, lat_bounds=lat_bounds, lon_bounds=lon_bounds,
        standardize=False, standardize_index=standardize,
        base_period=base_period,
        area_weight=False, lat_name=lat_name, lon_name=lon_name,
        time_name=time_name).rename('nino1')

    if input_frequency == 'monthly' and frequency == 'daily':

        index = index.resample({time_name : '1D'}).interpolate('linear')

    return index


def nino2(sst, frequency='monthly', base_period=None, standardize=False,
          lat_name=None, lon_name=None, time_name=None):
    """Calculate Nino2 index.

    Parameters
    ----------
    sst : xarray.DataArray
        Array containing SST values.

    frequency : str
        If given, downsample data to requested frequency.

    base_period : list
        Earliest and latest time to use for standardization.

    standardize : bool
        If True, standardize index using mean and standard deviation
        within the base period.

    lat_name : str
        Name of the latitude coordinate.

    lon_name : str
        Name of the longitude coordinate.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    index : xarray.DataArray
        Array containing the values of the Nino2 index.
    """

    lat_bounds = [-5.0, 0.0]
    lon_bounds = [-90.0, -80.0]

    # Ensure that the data provided is a data array
    sst = check_data_array(sst)

    # Get coordinate names
    if time_name is None:
        time_name = get_coordinate_standard_name(sst, 'time')

    if frequency is None:
        frequency = 'monthly'

    if frequency not in ('daily', 'monthly'):
        raise ValueError("Unsupported frequency '%r'" % frequency)

    input_frequency = detect_frequency(sst, time_name=time_name)

    if input_frequency not in ('daily', 'monthly'):
        raise RuntimeError('Can only calculate index for daily or monthly data')

    if input_frequency == 'daily' and frequency == 'monthly':

        sst = sst.resample({time_name : '1MS'}).mean()

    index = _mean_region_anomaly(
        sst, lat_bounds=lat_bounds, lon_bounds=lon_bounds,
        standardize=False, standardize_index=standardize,
        base_period=base_period, area_weight=False,
        lat_name=lat_name, lon_name=lon_name,
        time_name=time_name).rename('nino2')

    if input_frequency == 'monthly' and frequency == 'daily':

        index = index.resample({time_name : '1D'}).interpolate('linear')

    return index


def nino3(sst, frequency='monthly', base_period=None, standardize=False,
          lat_name=None, lon_name=None, time_name=None):
    """Calculate Nino3 index.

    Parameters
    ----------
    sst : xarray.DataArray
        Array containing SST values.

    frequency : str
        If given, downsample data to requested frequency.

    base_period : list
        Earliest and latest time to use for standardization.

    standardize : bool
        If True, standardize index using mean and standard deviation
        within the base period.

    lat_name : str
        Name of the latitude coordinate.

    lon_name : str
        Name of the longitude coordinate.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    index : xarray.DataArray
        Array containing the values of the Nino3 index.
    """

    lat_bounds = [-5.0, 5.0]
    lon_bounds = [-150.0, -90.0]

    # Ensure that the data provided is a data array
    sst = check_data_array(sst)

    # Get coordinate names
    if time_name is None:
        time_name = get_coordinate_standard_name(sst, 'time')

    if frequency is None:
        frequency = 'monthly'

    if frequency not in ('daily', 'monthly'):
        raise ValueError("Unsupported frequency '%r'" % frequency)

    input_frequency = detect_frequency(sst, time_name=time_name)

    if input_frequency not in ('daily', 'monthly'):
        raise RuntimeError('Can only calculate index for daily or monthly data')

    if input_frequency == 'daily' and frequency == 'monthly':

        sst = sst.resample({time_name : '1MS'}).mean()

    index = _mean_region_anomaly(
        sst, lat_bounds=lat_bounds, lon_bounds=lon_bounds,
        standardize=False, standardize_index=standardize,
        base_period=base_period, area_weight=False,
        lat_name=lat_name, lon_name=lon_name,
        time_name=time_name).rename('nino3')

    if input_frequency == 'monthly' and frequency == 'daily':

        index = index.resample({time_name : '1D'}).interpolate('linear')

    return index


def nino34(sst, frequency='monthly', base_period=None, standardize=False,
           lat_name=None, lon_name=None, time_name=None):
    """Calculate Nino3.4 index.

    Parameters
    ----------
    sst : xarray.DataArray
        Array containing SST values.

    frequency : str
        If given, downsample data to requested frequency.

    base_period : list
        Earliest and latest time to use for standardization.

    standardize : bool
        If True, standardize index using mean and standard deviation
        within the base period.

    lat_name : str
        Name of the latitude coordinate.

    lon_name : str
        Name of the longitude coordinate.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    index : xarray.DataArray
        Array containing the values of the Nino3.4 index.
    """

    lat_bounds = [-5.0, 5.0]
    lon_bounds = [-170.0, -120.0]

    # Ensure that the data provided is a data array
    sst = check_data_array(sst)

    # Get coordinate names
    if time_name is None:
        time_name = get_coordinate_standard_name(sst, 'time')

    if frequency is None:
        frequency = 'monthly'

    if frequency not in ('daily', 'monthly'):
        raise ValueError("Unsupported frequency '%r'" % frequency)

    input_frequency = detect_frequency(sst, time_name=time_name)

    if input_frequency not in ('daily', 'monthly'):
        raise RuntimeError('Can only calculate index for daily or monthly data')

    if input_frequency == 'daily' and frequency == 'monthly':

        sst = sst.resample({time_name : '1MS'}).mean()

    index = _mean_region_anomaly(
        sst, lat_bounds=lat_bounds, lon_bounds=lon_bounds,
        standardize=False, standardize_index=standardize,
        base_period=base_period, area_weight=False,
        lat_name=lat_name, lon_name=lon_name,
        time_name=time_name).rename('nino34')

    if input_frequency == 'monthly' and frequency == 'daily':

        index = index.resample({time_name : '1D'}).interpolate('linear')

    return index


def nino4(sst, frequency='monthly', base_period=None, standardize=False,
          lat_name=None, lon_name=None, time_name=None):
    """Calculate Nino4 index.

    Parameters
    ----------
    sst : xarray.DataArray
        Array containing SST values.

    frequency : str
        If given, downsample data to requested frequency.

    base_period : list
        Earliest and latest time to use for standardization.

    standardize : bool
        If True, standardize index using mean and standard deviation
        within the base period.

    lat_name : str
        Name of the latitude coordinate.

    lon_name : str
        Name of the longitude coordinate.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    index : xarray.DataArray
        Array containing the values of the Nino4 index.
    """

    lat_bounds = [-5.0, 5.0]
    lon_bounds = [160.0, -150.0]

    # Ensure that the data provided is a data array
    sst = check_data_array(sst)

    # Get coordinate names
    if time_name is None:
        time_name = get_coordinate_standard_name(sst, 'time')

    if frequency is None:
        frequency = 'monthly'

    if frequency not in ('daily', 'monthly'):
        raise ValueError("Unsupported frequency '%r'" % frequency)

    input_frequency = detect_frequency(sst, time_name=time_name)

    if input_frequency not in ('daily', 'monthly'):
        raise RuntimeError('Can only calculate index for daily or monthly data')

    if input_frequency == 'daily' and frequency == 'monthly':

        sst = sst.resample({time_name : '1MS'}).mean()

    index = _mean_region_anomaly(
        sst, lat_bounds=lat_bounds, lon_bounds=lon_bounds,
        standardize=False, standardize_index=standardize,
        base_period=base_period, area_weight=False,
        lat_name=lat_name, lon_name=lon_name,
        time_name=time_name).rename('nino4')

    if input_frequency == 'monthly' and frequency == 'daily':

        index = index.resample({time_name : '1D'}).interpolate('linear')

    return index


def soi(mslp, frequency='monthly', base_period=None, method='nearest',
        lat_name=None, lon_name=None, time_name=None):
    """Calculate SOI index based on definition used by CPC.

    Parameters
    ----------
    mslp : xarray.DataArray
        Array containing MSLP values.

    frequency : str
        If given, downsample data to requested frequency.

    base_period : list
        Earliest and latest time to use for standardization.

    method : str
        Method used to select grid points at which MSLP
        values are extracted.

    lat_name : str
        Name of the latitude coordinate.

    lon_name : str
        Name of the longitude coordinate.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    index : xarray.DataArray
        Array containing the values of the SOI.
    """

    # Ensure that the data provided is a data array
    mslp = check_data_array(mslp)

    # Get coordinate names
    if time_name is None:
        time_name = get_coordinate_standard_name(mslp, 'time')

    if lat_name is None:
        lat_name = get_coordinate_standard_name(mslp, 'lat')

    if lon_name is None:
        lon_name = get_coordinate_standard_name(mslp, 'lon')

    darwin_coords = {lat_name: 12.4634, lon_name: 130.8456}
    tahiti_coords = {lat_name: 17.6509, lon_name: 149.4260}

    base_period = check_base_period(mslp, base_period=base_period,
                                    time_name=time_name)

    if frequency is None:
        frequency = 'monthly'

    if frequency not in ('daily', 'monthly'):
        raise ValueError("Unsupported frequency '%r'" % frequency)

    input_frequency = detect_frequency(mslp, time_name=time_name)

    if input_frequency not in ('daily', 'monthly'):
        raise RuntimeError('Can only calculate index for daily or monthly data')

    darwin_mslp = mslp.sel(darwin_coords, method=method)
    tahiti_mslp = mslp.sel(tahiti_coords, method=method)

    if input_frequency == 'daily' and frequency == 'monthly':

        darwin_mslp = darwin_mslp.resample({time_name : '1MS'}).mean()
        tahiti_mslp = tahiti_mslp.resample({time_name : '1MS'}).mean()

    if input_frequency == 'monthly' or frequency == 'monthly':
        standardize_by = 'month'
    else:
        standardize_by = 'dayofyear'

    darwin_mslp_std_anom = standardized_anomalies(
        darwin_mslp, base_period=base_period, standardize_by=standardize_by,
        time_name=time_name)
    tahiti_mslp_std_anom = standardized_anomalies(
        tahiti_mslp, base_period=base_period, standardize_by=standardize_by,
        time_name=time_name)

    index = (tahiti_mslp_std_anom - darwin_mslp_std_anom).rename('soi')

    index = standardized_anomalies(index, base_period=base_period,
                                   standardize_by=standardize_by, time_name=time_name)

    if input_frequency == 'monthly' and frequency == 'daily':

        index = index.resample({time_name : '1D'}).interpolate('linear')

    return index


def troup_soi(mslp, frequency='monthly', base_period=None,
              standardize=False, method='nearest',
              lat_name=None, lon_name=None, time_name=None):
    """Calculate SOI index based on definition used by BOM.

    Parameters
    ----------
    mslp : xarray.DataArray
        Array containing MSLP values.

    frequency : str
        If given, downsample data to requested frequency.

    base_period : list
        Earliest and latest time to use for standardization.

    standardize : bool
        If True, standardize index by the mean and standard
        deviation within the base period.

    method : str
        Method used to select grid points at which MSLP
        values are extracted.

    lat_name : str
        Name of the latitude coordinate.

    lon_name : str
        Name of the longitude coordinate.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    index : xarray.DataArray
        Array containing the values of the SOI.
    """

    # Ensure that the data provided is a data array
    mslp = check_data_array(mslp)

    # Get coordinate names
    if time_name is None:
        time_name = get_coordinate_standard_name(mslp, 'time')

    if lat_name is None:
        lat_name = get_coordinate_standard_name(mslp, 'lat')

    if lon_name is None:
        lon_name = get_coordinate_standard_name(mslp, 'lon')

    darwin_coords = {lat_name: 12.4634, lon_name: 130.8456}
    tahiti_coords = {lat_name: 17.6509, lon_name: 149.4260}

    base_period = check_base_period(mslp, base_period=base_period,
                                    time_name=time_name)

    if frequency is None:
        frequency = 'monthly'

    if frequency not in ('daily', 'monthly'):
        raise ValueError("Unsupported frequency '%r'" % frequency)

    input_frequency = detect_frequency(mslp, time_name=time_name)

    if input_frequency not in ('daily', 'monthly'):
        raise RuntimeError('Can only calculate index for daily or monthly data')

    darwin_mslp = mslp.sel(darwin_coords, method=method)
    tahiti_mslp = mslp.sel(tahiti_coords, method=method)

    if input_frequency == 'daily' and frequency == 'monthly':

        darwin_mslp = darwin_mslp.resample({time_name : '1MS'}).mean()
        tahiti_mslp = tahiti_mslp.resample({time_name : '1MS'}).mean()

    mslp_diff = tahiti_mslp - darwin_mslp
    base_period_mslp_diff = mslp_diff.where(
        (mslp_diff[time_name] >= base_period[0]) &
        (mslp_diff[time_name] <= base_period[1]), drop=True)

    if input_frequency == 'monthly' or frequency == 'monthly':

        base_period_groups = base_period_mslp_diff[time_name].dt.month
        groups = mslp_diff[time_name].dt.month

    else:

        base_period_groups = base_period_mslp_diff[time_name].dt.dayofyear
        groups = mslp_diff[time_name].dt.dayofyear

    clim_mean = base_period_mslp_diff.groupby(
        base_period_groups).mean(time_name)
    clim_std = base_period_mslp_diff.groupby(
        base_period_groups).std(time_name)

    index = xr.apply_ufunc(
        lambda x, m, s: (x - m) / s,
        mslp_diff.groupby(groups), clim_mean, clim_std,
        dask='allowed').rename('soi')

    if standardize:

        base_period_index = index.where(
            (index[time_name] >= base_period[0]) &
            (index[time_name] <= base_period[1]), drop=True)

        index_mean = base_period_index.groupby(
            base_period_groups).mean(time_name)
        index_std = base_period_index.groupby(
            base_period_groups).std(time_name)

        index = xr.apply_ufunc(
            lambda x, m, s: (x - m) / s,
            index.groupby(groups), index_mean, index_std,
            dask='allowed')

    if input_frequency == 'monthly' and frequency == 'daily':

        index = index.resample({time_name : '1D'}).interpolate('linear')

    return index


def _check_consistent_coordinate_names(
    *arrays, lat_name=None, lon_name=None, time_name=None):
    """Check all arrays have consistent coordinate names."""

    if lat_name is None:
        lat_name = get_coordinate_standard_name(arrays[0], 'lat')

    if lon_name is None:
        lon_name = get_coordinate_standard_name(arrays[0], 'lon')

    if time_name is None:
        time_name = get_coordinate_standard_name(arrays[0], 'time')

    for arr in arrays:
        if time_name not in arr.dims:
            raise ValueError(
                "Could not find time coordinate '%s' in input" % time_name)

        if lat_name not in arr.dims:
            raise ValueError(
                "Could not find latitude coordinate '%s' in input" % lat_name)

        if lon_name not in arr.dims:
            raise ValueError(
                "Could not find longitude coordinate '%s' in input" % lon_name)


def _check_consistent_input_frequency(*arrays, time_name=None):
    """Check all arrays are given at same sampling rate."""

    input_frequency = None
    for arr in arrays:

        if time_name is None:
            arr_time_name = get_coordinate_standard_name(arr, 'time')
        else:
            arr_time_name = time_name

        arr_input_frequency = detect_frequency(arr, time_name=arr_time_name)

        if input_frequency is None:
            input_frequency = arr_input_frequency
        else:
            if arr_input_frequency != input_frequency:
                raise ValueError('Arrays have different sampling rates')


def _get_mei_season_str(m):
    """Return season corresponding to integer month."""

    if m == 1:
        return 'DJ'

    if m == 2:
        return 'JF'

    if m == 3:
        return 'FM'

    if m == 4:
        return 'MA'

    if m == 5:
        return 'AM'

    if m == 6:
        return 'MJ'

    if m == 7:
        return 'JJ'

    if m == 8:
        return 'JA'

    if m == 9:
        return 'AS'

    if m == 10:
        return 'SO'

    if m == 11:
        return 'ON'

    if m == 12:
        return 'ND'

    raise ValueError("Invalid month '%r'" % m)


def _convert_monthly_to_seasonal_data(data, time_name=None):
    """Get bimonthly seasonal data from monthly inputs."""

    if time_name is None:
        time_name = get_coordinate_standard_name(data, 'time')

    # NB, the date assigned to each sample corresponds to
    # the first day of the second month of the season.
    seasonal_data = data.rolling(
        dim={time_name: 2}, center=True, min_periods=1).mean()

    # For convenience, add label for season.
    def get_season_label(months):

        seasons = []
        for m in months:
            seasons.append(_get_mei_season_str(m))

        return seasons

    seasons = get_season_label(seasonal_data[time_name].dt.month)

    season_data = xr.DataArray(
        seasons, coords={time_name: seasonal_data[time_name]},
        dims=[time_name])

    seasonal_data['season'] = season_data

    return seasonal_data


def restrict_to_mei_analysis_region(data, lat_name=None, lon_name=None):
    """Restrict data to MEI analysis region."""

    if lat_name is None:
        lat_name = get_coordinate_standard_name(data, 'lat')

    if lon_name is None:
        lon_name = get_coordinate_standard_name(data, 'lon')

    shape_data = gp.read_file(MEI_REGION_SHP)
    region_mask = rm.Regions(shape_data['geometry'], numbers=[0])

    if np.all(data[lon_name] >= 0):
        mask = region_mask.mask(
            data, wrap_lon=True, lat_name=lat_name, lon_name=lon_name)
    else:
        mask = region_mask.mask(
            data, lat_name=lat_name, lon_name=lon_name)

    return data.where(mask == 0, drop=True)


def calculate_mei_standardized_seasonal_anomaly(
    data, base_period=None, time_name=None):
    """Calculate two monthly seasonal anomaly."""

    data = check_data_array(data)

    if time_name is None:
        time_name = get_coordinate_standard_name(data, 'time')

    base_period = check_base_period(data, base_period=base_period,
                                    time_name=time_name)

    input_frequency = detect_frequency(
        data, time_name=time_name)

    if input_frequency not in ('daily', 'monthly'):
        raise RuntimeError(
            'Can only calculate anomalies for daily or monthly data')

    if input_frequency == 'daily':
        monthly_data = data.resample({time_name : '1MS'}).mean()
    else:
        monthly_data = data

    seasonal_data = _convert_monthly_to_seasonal_data(
        monthly_data, time_name=time_name)

    base_period_seasonal_data = seasonal_data.where(
        (seasonal_data[time_name] >= base_period[0]) &
        (seasonal_data[time_name] <= base_period[1]), drop=True)

    seasonal_means = base_period_seasonal_data.groupby(
        base_period_seasonal_data['season']).mean(time_name)
    seasonal_std = base_period_seasonal_data.groupby(
        base_period_seasonal_data['season']).std(time_name)

    return xr.apply_ufunc(
        lambda x, m, s: (x - m) / s,
        seasonal_data.groupby(seasonal_data['season']),
        seasonal_means, seasonal_std, dask='allowed')


def _fix_mei_phases(mslp_eofs, uwnd_eofs, vwnd_eofs, sst_eofs, olr_eofs,
                    lon_name=None, lat_name=None):
    """Ensure maximum positive SST anomaly occurs in Nino region."""

    if lat_name is None:
        lat_name = get_coordinate_standard_name(sst_eofs, 'lat')

    if lon_name is None:
        lon_name = get_coordinate_standard_name(sst_eofs, 'lon')

    min_lon = sst_eofs[lon_name].min()
    max_lon = sst_eofs[lon_name].max()

    if min_lon < 0 and max_lon <= 180:
        leading_sst_eof = sst_eofs['EOFs'].where(
            (sst_eofs[lon_name] <= -120) &
            (sst_eofs[lon_name] >= -170) &
            (sst_eofs[lat_name] >= -5) &
            (sst_eofs[lat_name] <= 5), drop=True).isel(mode=0)
    else:
        leading_sst_eof = sst_eofs['EOFs'].where(
            (sst_eofs[lon_name] >= 190) &
            (sst_eofs[lon_name] <= 240) &
            (sst_eofs[lat_name] >= -5) &
            (sst_eofs[lat_name] <= 5), drop=True).isel(mode=0)

    flipped_leading_sst_eof = -leading_sst_eof
    nino34_max_anom = leading_sst_eof.max()
    flipped_nino34_max_anom = flipped_leading_sst_eof.max()

    if flipped_nino34_max_anom > nino34_max_anom:
        mslp_eofs['EOFs'] = -mslp_eofs['EOFs']
        mslp_eofs['PCs'] = -mslp_eofs['PCs']

        uwnd_eofs['EOFs'] = -uwnd_eofs['EOFs']
        uwnd_eofs['PCs'] = -uwnd_eofs['PCs']

        vwnd_eofs['EOFs'] = -vwnd_eofs['EOFs']
        vwnd_eofs['PCs'] = -vwnd_eofs['PCs']

        sst_eofs['EOFs'] = -sst_eofs['EOFs']
        sst_eofs['PCs'] = -sst_eofs['PCs']

        olr_eofs['EOFs'] = -olr_eofs['EOFs']
        olr_eofs['PCs'] = -olr_eofs['PCs']

    return mslp_eofs, uwnd_eofs, vwnd_eofs, sst_eofs, olr_eofs


def _project_onto_subspace_by_season(anom_data, seasonal_eofs,
                                     normalization=None,
                                     lat_name=None, time_name=None):
    """Project data onto the corresponding seasonal EOF."""

    anom_data = check_data_array(anom_data)
    seasonal_eofs = check_data_array(seasonal_eofs)

    if normalization is not None:
        normalization = check_data_array(normalization)
    else:
        normalization = xr.DataArray(
            np.ones(seasonal_eofs.sizes['season']),
            coords={'season': seasonal_eofs['season']},
            dims=['season'])

    if lat_name is None:
        lat_name = get_coordinate_standard_name(anom_data, 'lat')

    if time_name is None:
        time_name = get_coordinate_standard_name(anom_data, 'time')

    input_frequency = detect_frequency(anom_data, time_name=time_name)

    weights = np.cos(np.deg2rad(anom_data[lat_name])).clip(0., 1.) ** 0.5

    seasonal_eofs = seasonal_eofs.fillna(0)
    anom_data = (anom_data * weights).fillna(0)

    if input_frequency == 'daily':
        raise NotImplementedError('Calculation of daily index not implemented')

    def _project(x):
        s = x['season'][0].item()
        n = normalization.sel(season=s, drop=True).item()
        return x.dot(seasonal_eofs.sel(season=s, drop=True)) / n

    projection = anom_data.groupby('season').map(_project)

    return projection


def mei_loading_pattern(mslp_anom, uwnd_anom, vwnd_anom, sst_anom, olr_anom,
                        n_modes=10, lat_name=None, lon_name=None, time_name=None):
    """Calculate seasonal EOFs associated with standardized anomalies.

    Parameters
    ----------
    mslp_anom : xarray.DataArray
        Array containing standardized seasonal MSLP anomalies.

    uwnd_anom : xarray.DataArray
        Array containing standardized seasonal surface zonal wind anomalies.

    vwnd_anom : xarray.DataArray
        Array containing standardized seasonal surface meridional wind anomalies.

    sst_anom : xarray.DataArray
        Array containing standardized seasonal SST anomalies.

    olr_anom : xarray.DataArray
        Array containing standardized seasonal top-of-atmosphere OLR anomalies.

    n_modes : integer
        Number of EOF modes to calculate. If None, only the leading
        mode is calculated.

    lat_name : str
        Name of the latitude coordinate.

    lon_name : str
        Name of the longitude coordinate.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    eofs : xarray.Dataset
        Dataset containing the seasonal EOFs.
    """

    # Ensure all inputs are data arrays
    mslp_anom = check_data_array(mslp_anom)
    uwnd_anom = check_data_array(uwnd_anom)
    vwnd_anom = check_data_array(vwnd_anom)
    sst_anom = check_data_array(sst_anom)
    olr_anom = check_data_array(olr_anom)

    # Get and check coordinate names
    if time_name is None:
        time_name = get_coordinate_standard_name(mslp_anom, 'time')

    if lat_name is None:
        lat_name = get_coordinate_standard_name(mslp_anom, 'lat')

    if lon_name is None:
        lon_name = get_coordinate_standard_name(mslp_anom, 'lon')

    _check_consistent_coordinate_names(
        mslp_anom, uwnd_anom, vwnd_anom, sst_anom, olr_anom,
        lat_name=lat_name, lon_name=lon_name, time_name=time_name)

    if n_modes is None:
        n_modes = 1

    if not isinstance(n_modes, INTEGER_TYPES) or n_modes < 1:
        raise ValueError(
            'Number of modes must be at least 1, but got %r' % n_modes)

    # Restrict all variables to common time period and analysis
    # region.
    start_time = max(
        [mslp_anom[time_name].min(), uwnd_anom[time_name].min(),
         vwnd_anom[time_name].min(), sst_anom[time_name].min(),
         olr_anom[time_name].min()])
    end_time = min(
        [mslp_anom[time_name].max(), uwnd_anom[time_name].max(),
         vwnd_anom[time_name].max(), sst_anom[time_name].max(),
         olr_anom[time_name].max()])

    def _restrict_time_period(data):
        return data.where(
            (data[time_name] >= start_time) &
            (data[time_name] <= end_time), drop=True)

    mslp_anom = _restrict_time_period(mslp_anom)
    uwnd_anom = _restrict_time_period(uwnd_anom)
    vwnd_anom = _restrict_time_period(vwnd_anom)
    sst_anom = _restrict_time_period(sst_anom)
    olr_anom = _restrict_time_period(olr_anom)

    def _scos_weights(data):
        return np.cos(np.deg2rad(data[lat_name])).clip(0., 1.) ** 0.5

    mslp_weights = _scos_weights(mslp_anom)
    uwnd_weights = _scos_weights(uwnd_anom)
    vwnd_weights = _scos_weights(vwnd_anom)
    sst_weights = _scos_weights(sst_anom)
    olr_weights = _scos_weights(olr_anom)

    weights = [mslp_weights, uwnd_weights, vwnd_weights,
               sst_weights, olr_weights]

    # Calculate EOFs for each season.
    seasons = ['DJ', 'JF', 'FM', 'MA', 'AM', 'MJ',
               'JJ', 'JA', 'AS', 'SO', 'ON', 'ND']
    seasons_da = xr.DataArray(seasons, coords={'season': seasons},
                              dims=['season'], name='season')

    all_mslp_eofs = []
    all_uwnd_eofs = []
    all_vwnd_eofs = []
    all_sst_eofs = []
    all_olr_eofs = []
    all_normalizations = []

    for season in seasons:

        season_mslp = mslp_anom.where(
            mslp_anom['season'] == season, drop=True)

        season_uwnd = uwnd_anom.where(
            uwnd_anom['season'] == season, drop=True)

        season_vwnd = vwnd_anom.where(
            vwnd_anom['season'] == season, drop=True)

        season_sst = sst_anom.where(
            sst_anom['season'] == season, drop=True)

        season_olr = olr_anom.where(
            olr_anom['season'] == season, drop=True)

        mslp_eofs, uwnd_eofs, vwnd_eofs, sst_eofs, olr_eofs = \
            eofs(season_mslp, season_uwnd, season_vwnd, season_sst, season_olr,
            weight=weights, n_modes=n_modes, sample_dim=time_name)

        mslp_eofs, uwnd_eofs, vwnd_eofs, sst_eofs, olr_eofs = \
            _fix_mei_phases(
                mslp_eofs, uwnd_eofs, vwnd_eofs, sst_eofs, olr_eofs,
                lon_name=lon_name, lat_name=lat_name)

        normalization = mslp_eofs['PCs'].sel(mode=0).std(time_name)

        all_mslp_eofs.append(mslp_eofs['EOFs'])
        all_uwnd_eofs.append(uwnd_eofs['EOFs'])
        all_vwnd_eofs.append(vwnd_eofs['EOFs'])
        all_sst_eofs.append(sst_eofs['EOFs'])
        all_olr_eofs.append(olr_eofs['EOFs'])
        all_normalizations.append(normalization)

    all_mslp_eofs = xr.concat(all_mslp_eofs, seasons_da)
    all_uwnd_eofs = xr.concat(all_uwnd_eofs, seasons_da)
    all_vwnd_eofs = xr.concat(all_vwnd_eofs, seasons_da)
    all_sst_eofs = xr.concat(all_sst_eofs, seasons_da)
    all_olr_eofs = xr.concat(all_olr_eofs, seasons_da)
    season_normalizations = xr.concat(all_normalizations, seasons_da)

    data_vars = {'mslp': all_mslp_eofs,
                 'uwnd': all_uwnd_eofs,
                 'vwnd': all_vwnd_eofs,
                 'sst': all_sst_eofs,
                 'olr': all_olr_eofs,
                 'normalization': season_normalizations
                }

    eofs_ds = xr.Dataset(data_vars)

    return eofs_ds


def mei(mslp, uwnd, vwnd, sst, olr, frequency='monthly', base_period=None,
        lat_name=None, lon_name=None, time_name=None):
    """Calculate MEIv2.

    Parameters
    ----------
    mslp : xarray.DataArray
        Array containing MSLP values.

    uwnd : xarray.DataArray
        Array containing surface zonal wind values.

    vwnd : xarray.DataArray
        Array containing surface meridional wind values.

    sst : xarray.DataArray
        Array containing SST values.

    olr : xarray.DataArray
        Array containing top-of-atmosphere OLR values.

    frequency : str
        If given, downsample data to requested frequency.

    base_period : list
        Earliest and latest time to use for standardization.

    lat_name : str
        Name of the latitude coordinate.

    lon_name : str
        Name of the longitude coordinate.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    result : xarray.Dataset
        Dataset containing the MEI loading patterns and index.
    """

    # Ensure all inputs are data arrays
    mslp = check_data_array(mslp)
    uwnd = check_data_array(uwnd)
    vwnd = check_data_array(vwnd)
    sst = check_data_array(sst)
    olr = check_data_array(olr)

    # Get and check coordinate names
    if time_name is None:
        time_name = get_coordinate_standard_name(mslp, 'time')

    if lat_name is None:
        lat_name = get_coordinate_standard_name(mslp, 'lat')

    if lon_name is None:
        lon_name = get_coordinate_standard_name(mslp, 'lon')

    _check_consistent_coordinate_names(
        mslp, uwnd, vwnd, sst, olr,
        lat_name=lat_name, lon_name=lon_name, time_name=time_name)

    _check_consistent_input_frequency(
        mslp, uwnd, vwnd, sst, olr, time_name=time_name)

    base_period = check_base_period(mslp, base_period=base_period,
                                    time_name=time_name)

    if frequency is None:
        frequency = 'monthly'

    if frequency not in ('daily', 'monthly'):
        raise ValueError("Unsupported frequency '%r'" % frequency)

    input_frequency = detect_frequency(mslp, time_name=time_name)

    mslp = restrict_to_mei_analysis_region(
        mslp, lat_name=lat_name, lon_name=lon_name)
    uwnd = restrict_to_mei_analysis_region(
        uwnd, lat_name=lat_name, lon_name=lon_name)
    vwnd = restrict_to_mei_analysis_region(
        vwnd, lat_name=lat_name, lon_name=lon_name)
    sst = restrict_to_mei_analysis_region(
        sst, lat_name=lat_name, lon_name=lon_name)
    olr = restrict_to_mei_analysis_region(
        olr, lat_name=lat_name, lon_name=lon_name)

    # Currently, only seasonal calculation of the index
    # is supported, with daily values obtained via
    # interpolation.
    if input_frequency == 'daily':
        mslp = mslp.resample({time_name : '1MS'}).mean()
        uwnd = uwnd.resample({time_name : '1MS'}).mean()
        vwnd = vwnd.resample({time_name : '1MS'}).mean()
        sst = sst.resample({time_name : '1MS'}).mean()
        olr = olr.resample({time_name : '1MS'}).mean()

    mslp_anom = calculate_mei_standardized_seasonal_anomaly(
        mslp, base_period=base_period, time_name=time_name)
    uwnd_anom = calculate_mei_standardized_seasonal_anomaly(
        uwnd, base_period=base_period, time_name=time_name)
    vwnd_anom = calculate_mei_standardized_seasonal_anomaly(
        vwnd, base_period=base_period, time_name=time_name)
    sst_anom = calculate_mei_standardized_seasonal_anomaly(
        sst, base_period=base_period, time_name=time_name)
    olr_anom = calculate_mei_standardized_seasonal_anomaly(
        olr, base_period=base_period, time_name=time_name)

    base_period_mslp_anom = mslp_anom.where(
        (mslp_anom[time_name] >= base_period[0]) &
        (mslp_anom[time_name] <= base_period[1]), drop=True)
    base_period_uwnd_anom = uwnd_anom.where(
        (uwnd_anom[time_name] >= base_period[0]) &
        (uwnd_anom[time_name] <= base_period[1]), drop=True)
    base_period_vwnd_anom = vwnd_anom.where(
        (vwnd_anom[time_name] >= base_period[0]) &
        (vwnd_anom[time_name] <= base_period[1]), drop=True)
    base_period_sst_anom = sst_anom.where(
        (sst_anom[time_name] >= base_period[0]) &
        (sst_anom[time_name] <= base_period[1]), drop=True)
    base_period_olr_anom = olr_anom.where(
        (olr_anom[time_name] >= base_period[0]) &
        (olr_anom[time_name] <= base_period[1]), drop=True)

    mei_eofs_ds = mei_loading_pattern(
        base_period_mslp_anom, base_period_uwnd_anom,
        base_period_vwnd_anom, base_period_sst_anom,
        base_period_olr_anom, n_modes=10,
        lat_name=lat_name, lon_name=lon_name, time_name=time_name)

    mslp_index = _project_onto_subspace_by_season(
        mslp_anom, mei_eofs_ds['mslp'].sel(mode=0),
        normalization=mei_eofs_ds['normalization'],
        lat_name=lat_name, time_name=time_name)
    uwnd_index = _project_onto_subspace_by_season(
        uwnd_anom, mei_eofs_ds['uwnd'].sel(mode=0),
        normalization=mei_eofs_ds['normalization'],
        lat_name=lat_name, time_name=time_name)
    vwnd_index = _project_onto_subspace_by_season(
        vwnd_anom, mei_eofs_ds['vwnd'].sel(mode=0),
        normalization=mei_eofs_ds['normalization'],
        lat_name=lat_name, time_name=time_name)
    sst_index = _project_onto_subspace_by_season(
        sst_anom, mei_eofs_ds['sst'].sel(mode=0),
        normalization=mei_eofs_ds['normalization'],
        lat_name=lat_name, time_name=time_name)
    olr_index = _project_onto_subspace_by_season(
        olr_anom, mei_eofs_ds['olr'].sel(mode=0),
        normalization=mei_eofs_ds['normalization'],
        lat_name=lat_name, time_name=time_name)

    index = (mslp_index + uwnd_index + vwnd_index +
             sst_index + olr_index).rename('mei')

    if frequency == 'daily':
        index = index.resample({time_name : '1D'}).interpolate('linear')

    data_vars = {'mslp_pattern' : mei_eofs_ds['mslp'].sel(mode=0, drop=True),
                 'uwnd_pattern' : mei_eofs_ds['uwnd'].sel(mode=0, drop=True),
                 'vwnd_pattern' : mei_eofs_ds['vwnd'].sel(mode=0, drop=True),
                 'sst_pattern' : mei_eofs_ds['sst'].sel(mode=0, drop=True),
                 'olr_pattern' : mei_eofs_ds['olr'].sel(mode=0, drop=True),
                 'index' : index}

    return xr.Dataset(data_vars)
