"""
Provides routines for computing SAM indices.
"""


import numbers

import numpy as np
import xarray as xr


from ..utils import (check_base_period, check_data_array,
                     detect_frequency, downsample_data, eofs,
                     get_coordinate_standard_name, is_daily_data,
                     is_monthly_data, standardized_anomalies)


INTEGER_TYPES = (numbers.Integral, np.integer)


def gong_wang_sam(mslp, frequency=None, standardize_by='month',
                  low_latitude=-40, high_latitude=-65, method='nearest',
                  base_period=None, lat_name=None, lon_name=None,
                  time_name=None):
    """Calculates the Gong and Wang SAM index.

    The SAM index introduced by Gong and Wang is defined as the
    difference between the low- and high-latitude zonal means of
    standardised MSLP anomalies. By default, the low-latitude zonal mean
    is taken at 40S and the high-latitude mean at 65S.

    See Gong, D. and Wang, S., "Definition of Antarctic oscillation
    index", Geophys. Res. Lett. 26, 459 - 462 (1999),
    doi:10.1029/1999GL900003 .

    Parameters
    ----------
    mslp : xarray.DataArray
        Array containing MSLP values.

    frequency : str
        If given, downsample data to requested frequency.

    standardize_by : str
        Time intervals to standardize within.

    low_latitude : float
        Low latitude to take zonal mean at.

    high_latitude : float
        High latitude to take zonal mean at.

    method : str
        Method used to select grid points.

    base_period : list
        Earliest and latest times to use for standardization.

    lat_name : str
        Name of the latitude coordinate.

    lon_name : str
        Name of the longitude coordinate.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    index : xarray.DataArray
        Array containing the values of the SAM index.
    """

    # Ensure that the data provided is a data array
    mslp = check_data_array(mslp)

    # Get coordinate names
    if lat_name is None:
        lat_name = get_coordinate_standard_name(mslp, 'lat')

    if lon_name is None:
        lon_name = get_coordinate_standard_name(mslp, 'lon')

    if time_name is None:
        time_name = get_coordinate_standard_name(mslp, 'time')

    # If required, resample the data to a different time frequency.
    if frequency is not None:
        mslp = downsample_data(mslp, frequency=frequency,
                               time_name=time_name)

    # Calculate zonal means of SLP
    zmslp_low_lat = mslp.sel(
        {lat_name : low_latitude}, method=method).mean(lon_name)
    zmslp_high_lat = mslp.sel(
        {lat_name : high_latitude}, method=method).mean(lon_name)

    # Calculate standardized anomalies of zonal mean SLP
    stand_zmslp_low_lat = standardized_anomalies(
        zmslp_low_lat, base_period=base_period, standardize_by=standardize_by,
        time_name=time_name)
    stand_zmslp_high_lat = standardized_anomalies(
        zmslp_high_lat, base_period=base_period, standardize_by=standardize_by,
        time_name=time_name)

    return (stand_zmslp_low_lat - stand_zmslp_high_lat).rename('sam_index')


def marshall_sam(mslp, frequency=None, use_valdivia=False, interpolate=True,
                 method='nearest', base_period=None, standardize_by='month',
                 lat_name=None, lon_name=None, time_name=None):
    """Calculates the Marshall SAM index.

    The Marshall SAM index is defined as the difference between
    standardized MSLP anomalies averaged between low- and high-latitude
    stations.

    See Marshall, G. J., "Trends in the Southern Annular Mode
    from Observations and Reanalyses", J. Climate, 16, 4134 - 4143
    (2003), doi:10.1175/1520-0442(2003)016<4134:TITSAM>2.0.CO;2 .

    Parameters
    ----------
    mslp : xarray.DataArray
        Array containing MSLP values.

    frequency : str
        If given, downsample data to requested frequency.

    use_valdivia : boolean
        Use Valdivia station location.

    interpolate : boolean
        Interpolate data to station locations.

    method : str
        Interpolation method.

    base_period : list
        Earliest and latest times to use for standardization.

    standardize_by : str
        Time intervals to standardize within.

    lat_name : str
        Name of the latitude coordinate.

    lon_name : str
        Name of the longitude coordinate.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    index: xarray.DataArray
        Array containing the values of the SAM index.
    """

    # Ensure that the data provided is a data array
    mslp = check_data_array(mslp)

    # Get coordinate names
    if lat_name is None:
        lat_name = get_coordinate_standard_name(mslp, 'lat')

    if lon_name is None:
        lon_name = get_coordinate_standard_name(mslp, 'lon')

    if time_name is None:
        time_name = get_coordinate_standard_name(mslp, 'time')

    low_lat_stations = [
        {'lat' : -46.9, 'lon' : 37.9},  # Marion Island
        {'lat' : -37.8, 'lon' : 77.5},  # Ile Nouvelle Amsterdam
        {'lat' : -42.9, 'lon' : 147.3}, # Hobart
        {'lat' : -43.5, 'lon' : 172.6}, # Christchurch
        {'lat' : -40.4, 'lon' : 350.1}, # Gough Island
    ]

    if use_valdivia:
        # Use Valdivia station location.
        low_lat_stations.append({'lat': -39.6, 'lon': 286.9})
    else:
        # Use Puerto Montt station location.
        low_lat_stations.append({'lat' : -41.3, 'lon': 286.9})

    high_lat_stations = [
        {'lat' : -70.8, 'lon' : 11.8},  # Novolazarevskaya
        {'lat' : -67.6, 'lon' : 62.9},  # Mawson
        {'lat' : -66.6, 'lon' : 93.0},  # Mirny
        {'lat' : -66.3, 'lon' : 110.5}, # Casey
        {'lat' : -66.7, 'lon' : 140.0}, # Dumont D'urville
        {'lat' : -65.2, 'lon' : 295.7}, # Faraday/Vernadsky
    ]

    low_lat_station_data = []
    high_lat_station_data = []

    # If required, resample the data to a different time frequency.
    if frequency is not None:
        station_mslp = downsample_data(mslp, frequency=frequency,
                                       time_name=time_name)
    else:
        station_mslp = mslp

    # Interpolate or select MSLP values at each of the stations.
    if interpolate:
        station_lats = ([station['lat'] for station in low_lat_stations] +
                        [station['lat'] for station in high_lat_stations])
        station_lons = ([station['lon'] for station in low_lat_stations] +
                        [station['lon'] for station in high_lat_stations])

        station_lats = sorted(station_lats)
        if station_mslp[lat_name].values[0] > station_mslp[lat_name].values[-1]:
            station_lats = station_lats[::-1]

        station_lons = sorted(station_lons)

        station_mslp = station_mslp.interp(
            {lat_name: station_lats, lon_name: station_lons},
            method=method)

    for station in low_lat_stations:

        coordinates = {
            lat_name: station['lat'],
            lon_name: station['lon']
        }

        low_lat_station_data.append(
            station_mslp.sel(coordinates, method=method))

    for station in high_lat_stations:

        coordinates = {
            lat_name: station['lat'],
            lon_name: station['lon']
        }

        high_lat_station_data.append(
            station_mslp.sel(coordinates, method=method))

    mslp_low_lat = xr.concat(low_lat_station_data, dim='station')
    mslp_high_lat = xr.concat(high_lat_station_data, dim='station')

    # Calculate proxy zonal mean as mean of stations
    zmslp_low_lat = mslp_low_lat.mean('station')
    zmslp_high_lat = mslp_high_lat.mean('station')

    # Calculate standardized anomalies of zonal mean SLP
    stand_zmslp_low_lat = standardized_anomalies(
        zmslp_low_lat, base_period=base_period, standardize_by=standardize_by,
        time_name=time_name)
    stand_zmslp_high_lat = standardized_anomalies(
        zmslp_high_lat, base_period=base_period, standardize_by=standardize_by,
        time_name=time_name)

    return (stand_zmslp_low_lat - stand_zmslp_high_lat).rename('sam_index')


def _fix_sam_pc_phase(eofs_ds, sam_mode, lat_name=None):
    """Fixes the sign of the SAM mode and PCs to chosen convention."""

    if lat_name is None:
        lat_name = get_coordinate_standard_name(eofs_ds, 'lat')

    sam_eof = eofs_ds.sel(mode=sam_mode, drop=True)['EOFs']
    lat_max = sam_eof.where(sam_eof == sam_eof.max(), drop=True)[lat_name].item()
    lat_min = sam_eof.where(sam_eof == sam_eof.min(), drop=True)[lat_name].item()

    if lat_max < lat_min:
        eofs_ds['EOFs'] = xr.where(eofs_ds['mode'] == sam_mode,
                                   -eofs_ds['EOFs'], eofs_ds['EOFs'])
        eofs_ds['PCs'] = xr.where(eofs_ds['mode'] == sam_mode,
                                  -eofs_ds['PCs'], eofs_ds['PCs'])

    return eofs_ds


def sam_loading_pattern(hgt_anom, sam_mode=0, n_modes=None, weights=None,
                        lat_name=None, lon_name=None, time_name=None):
    """Calculate loading pattern for the principal component SAM index.

    The positive phase of the SAM is defined such that the anomalous
    lows occur at high latitudes.

    See, e.g., https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/history/method.shtml .

    Parameters
    ----------
    hgt_anom : xarray.DataArray
        Array containing geopotential height anomalies.

    sam_mode : integer
        Mode whose PC is taken to be the index.

    n_modes : integer
        Number of modes to compute in EOFs calculation.
        If None, only the minimum number of modes needed to include the
        requested SAM mode are computed.

    weights : xarray.DataArray
        Weights to apply in calculating the EOFs.

    lat_name : str
        Name of the latitude coordinate.

    lon_name : str
        Name of the longitude coordinate.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    pattern: xarray.Dataset
        Dataset containing the computed EOFs and PCs.
    """

    # Ensure that the data provided is a data array
    hgt_anom = check_data_array(hgt_anom)

    if lat_name is None:
        lat_name = get_coordinate_standard_name(hgt_anom, 'lat')

    if lon_name is None:
        lon_name = get_coordinate_standard_name(hgt_anom, 'lon')

    if time_name is None:
        time_name = get_coordinate_standard_name(hgt_anom, 'time')

    if not isinstance(sam_mode, INTEGER_TYPES) or sam_mode < 0:
        raise ValueError(
            'Invalid SAM mode: got %r but must be a non-negative integer')

    if n_modes is None:
        n_modes = sam_mode + 1

    eofs_ds = eofs(hgt_anom, sample_dim=time_name,
                   weight=weights, n_modes=n_modes)

    # Fix signs such that positive SAM corresponds to anomalous
    # lows at higher latitude
    eofs_ds = _fix_sam_pc_phase(eofs_ds, sam_mode, lat_name=lat_name)

    return eofs_ds


def pc_sam(hgt, frequency=None, base_period=None, sam_mode=0, n_modes=None,
           low_latitude_boundary=-20, lat_name=None, lon_name=None,
           time_name=None):
    """Calculate principal component based SAM index.

    The returned SAM index is taken to be the principal component (PC)
    obtained by projecting geopotential height anomalies onto
    the chosen empirical orthogonal function (EOF) mode calculated from
    the anomalies. A square root of cos(latitude) weighting is applied
    when calculating the EOFs.

    Note that the SAM index provided by the NOAA CPC adopts a
    normalization convention in which the index is normalized by the
    standard deviation of the monthly mean PCs over the base period
    used to compute the SAM pattern.

    See, e.g., https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/history/method.shtml .

    Parameters
    ----------
    input_data : xarray.DataArray
        Array containing geopotential height values.

    frequency : None | 'daily' | 'monthly'
        Frequency to calculate index at.

    base_period : list
        If given, a two element list containing the
        earliest and latest dates to include when calculating the
        EOFs. If None, the EOFs are computed using the full
        dataset.

    sam_mode : integer
        Mode whose PC is taken to be the index.

    n_modes : integer
        Number of modes to compute in EOFs calculation.
        If None, only the minimum number of modes needed to include the
        requested SAM mode are computed.

    low_latitude_boundary : float
        Low-latitude bound for analysis region.

    lat_name : str
        Name of the latitude coordinate.

    lon_name : str
        Name of the longitude coordinate.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    result : xarray.Dataset
        Dataset containing the SAM loading pattern and index.
    """

    # Ensure that the data provided is a data array
    hgt = check_data_array(hgt)

    if lat_name is None:
        lat_name = get_coordinate_standard_name(hgt, 'lat')

    if lon_name is None:
        lon_name = get_coordinate_standard_name(hgt, 'lon')

    if time_name is None:
        time_name = get_coordinate_standard_name(hgt, 'time')

    # Restrict to data polewards of boundary latitude
    hgt = hgt.where(hgt[lat_name] <= low_latitude_boundary, drop=True)

    # Get subset of data to use for computing anomalies and EOFs.
    base_period = check_base_period(hgt, base_period=base_period,
                                    time_name=time_name)

    if frequency is None:
        frequency = 'monthly'

    if frequency not in ('daily', 'monthly'):
        raise ValueError("Unsupported frequency '%r'" % frequency)

    input_frequency = detect_frequency(hgt, time_name=time_name)

    if input_frequency not in ('daily', 'monthly'):
        raise RuntimeError('Can only calculate index for daily or monthly data')

    # Calculate base period climatology and anomalies.
    base_period_hgt = hgt.where(
        (hgt[time_name] >= base_period[0]) &
        (hgt[time_name] <= base_period[1]), drop=True)

    if input_frequency == 'daily':

        base_period_monthly_hgt = downsample_data(
            base_period_hgt, frequency='monthly', time_name=time_name)

        monthly_clim = base_period_monthly_hgt.groupby(
            base_period_monthly_hgt[time_name].dt.month).mean(time_name)

        monthly_hgt_anom = (
            base_period_monthly_hgt.groupby(
                base_period_monthly_hgt[time_name].dt.month) -
            monthly_clim)

        if frequency == 'daily':

            daily_clim = base_period_hgt.groupby(
                base_period_hgt[time_name].dt.dayofyear).mean(time_name)

            hgt_anom = hgt.groupby(hgt[time_name].dt.dayofyear) - daily_clim

        elif frequency == 'monthly':

            hgt = downsample_data(hgt, frequency='monthly',
                                  time_name=time_name)

            hgt_anom = hgt.groupby(hgt[time_name].dt.month) - monthly_clim

    else:

        if frequency == 'daily':
            raise RuntimeError(
                'Attempting to calculate daily index from monthly data')

        monthly_clim = base_period_hgt.groupby(
            base_period_hgt[time_name].dt.month).mean(time_name)

        hgt_anom = hgt.groupby(hgt[time_name].dt.month) - monthly_clim

        monthly_hgt_anom = hgt_anom.where(
            (hgt_anom[time_name] >= base_period[0]) &
            (hgt_anom[time_name] <= base_period[1]), drop=True)

    # Get square root of cos(latitude) weights.
    scos_weights = np.cos(np.deg2rad(hgt_anom[lat_name])).clip(0., 1.) ** 0.5

    # Calculate loading pattern from monthly anomalies in base period.
    loadings_ds = sam_loading_pattern(
        monthly_hgt_anom, sam_mode=sam_mode, n_modes=n_modes,
        weights=scos_weights, lat_name=lat_name, lon_name=lon_name,
        time_name=time_name)

    pcs_std = loadings_ds.sel(mode=sam_mode)['PCs'].std(ddof=1)

    # Project weighted anomalies onto SAM mode
    sam_eof = loadings_ds.sel(mode=sam_mode)['EOFs']
    index = ((hgt_anom * scos_weights).fillna(0)).dot(
        sam_eof.fillna(0)).rename('sam_index')

    data_vars = {'pattern': sam_eof, 'index': index / pcs_std}

    return xr.Dataset(data_vars)
