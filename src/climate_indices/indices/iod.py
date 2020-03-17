"""
Provides routines for calculating indices associated with the IOD.
"""


from __future__ import absolute_import

from copy import deepcopy
import dask.array as da
import numpy as np
import scipy.signal
import xarray as xr

from scipy.fft import irfft, rfft, rfftfreq

from ..utils import (check_base_period, check_data_array,
                     check_fixed_missing_values, detect_frequency,
                     get_coordinate_standard_name,
                     get_valid_variables, is_dask_array)


def _apply_fft_high_pass_filter(data, fmin, fs=None, workers=None,
                                detrend=True, time_name=None):
    """Apply high-pass filter to FFT of given data.

    Parameters
    ----------
    data : xarray.DataArray
        Data to filter.

    fmin : float
        Lowest frequency in pass band.

    fs : float
        Sampling frequency.

    workers : int
        Number of parallel jobs to use in computing FFT.

    detrend : bool
        If True, remove linear trend from data before computing FFT.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    filtered : xarray.DataArray
        Array containing the high-pass filtered data.
    """

    data = check_data_array(data)

    if time_name is None:
        time_name = get_coordinate_standard_name(data, 'time')

    feature_dims = [d for d in data.dims if d != time_name]

    # Handle case in which data is simply a time-series
    if not feature_dims:
        original_shape = None
    else:
        original_shape = [data.sizes[d] for d in feature_dims]

    time_dim_pos = data.get_axis_num(time_name)
    if time_dim_pos != 0:
        data = data.transpose(*([time_name] + feature_dims))

    # Convert to 2D array
    n_samples = data.sizes[time_name]

    if feature_dims:
        n_features = np.product(original_shape)
    else:
        n_features = 1

    flat_data = data.values.reshape((n_samples, n_features))

    check_fixed_missing_values(flat_data, axis=0)

    valid_data, valid_features = get_valid_variables(flat_data)

    valid_data = valid_data.swapaxes(0, 1)

    if detrend:
        valid_data = scipy.signal.detrend(
            valid_data, axis=-1, type='linear')

    # Compute spectrum and apply high-pass filter
    spectrum = rfft(valid_data, axis=-1, workers=workers)
    fft_freqs = rfftfreq(n_samples, d=(1.0 / fs))

    filter_mask = fft_freqs < fmin

    spectrum[..., filter_mask] = 0.0

    filtered_valid_data = irfft(
        spectrum, n=n_samples, axis=-1, workers=workers).swapaxes(0, 1)

    if is_dask_array(flat_data):
        filtered_cols = [None] * n_features
        pos = 0
        for j in range(n_features):
            if j in valid_features:
                filtered_cols[j] = filtered_valid_data[:, pos].reshape(
                    (n_samples, 1))
                pos += 1
            else:
                filtered_cols[j] = da.full((n_samples, 1), np.NaN)

        filtered_data = da.hstack(filtered_cols)
    else:
        filtered_data = np.full((n_samples, n_features), np.NaN)
        filtered_data[:, valid_features] = filtered_valid_data

    if original_shape:
        filtered_data = filtered_data.reshape([n_samples,] + original_shape)
        filtered_dims = [time_name] + feature_dims
    else:
        filtered_data = filtered_data.ravel()
        filtered_dims = [time_name]

    filtered_coords = deepcopy(data.coords)

    result = xr.DataArray(
        filtered_data, coords=filtered_coords, dims=filtered_dims)

    if time_dim_pos != 0:
        result = result.transpose(*data.dims)

    return result


def _calculate_monthly_anomaly(data, apply_filter=False, base_period=None,
                               lat_name=None, lon_name=None, time_name=None):
    """Calculate monthly anomalies at each grid point."""

    # Ensure that the data provided is a data array
    data = check_data_array(data)

    # Get coordinate names
    if lat_name is None:
        lat_name = get_coordinate_standard_name(data, 'lat')

    if lon_name is None:
        lon_name = get_coordinate_standard_name(data, 'lon')

    if time_name is None:
        time_name = get_coordinate_standard_name(data, 'time')

    # Get subset of data to use for computing anomalies
    base_period = check_base_period(
        data, base_period=base_period, time_name=time_name)

    input_frequency = detect_frequency(data, time_name=time_name)

    if input_frequency not in ('daily', 'monthly'):
        raise RuntimeError(
            'Can only calculate anomalies for daily or monthly data')

    if input_frequency == 'daily':
        data = data.resample({time_name : '1MS'}).mean()

    base_period_data = data.where(
        (data[time_name] >= base_period[0]) &
        (data[time_name] <= base_period[1]), drop=True)

    monthly_clim = base_period_data.groupby(
        base_period_data[time_name].dt.month).mean(time_name)

    monthly_anom = data.groupby(data[time_name].dt.month) - monthly_clim

    if apply_filter:
        monthly_anom = monthly_anom.rolling(
            {time_name : 3}).mean().dropna(time_name, how='all')

        # Approximate sampling frequency
        seconds_per_day = 60 * 60 * 24.0
        fs = 1.0 / (seconds_per_day * 30)

        # Remove all modes with period greater than 7 years
        fmin = 1.0 / (seconds_per_day * 365.25 * 7)

        monthly_anom = _apply_fft_high_pass_filter(
            monthly_anom, fmin=fmin, fs=fs, detrend=True,
            time_name=time_name)

    return monthly_anom


def dmi(sst, frequency='monthly', apply_filter=False, base_period=None,
        area_weight=False, lat_name=None, lon_name=None, time_name=None):
    """Calculate dipole mode index.

    The DMI introduced by Saji et al is defined as the
    difference in SST anomalies between the western and
    south-eastern tropical Indian ocean, defined as the
    regions 50E - 70E, 10S - 10N and 90E - 110E, 10S - 0N,
    respectively.

    See Saji, N. H. et al, "A dipole mode in the tropical
    Indian Ocean", Nature 401, 360 - 363 (1999).

    Parameters
    ----------
    sst : xarray.DataArray
        Array containing SST values.

    frequency : None | 'daily' | 'monthly'
        Frequency to calculate index at.

    apply_filter : bool
        If True, filter the data by applying a three month
        running mean and remove harmonics with periods longer
        than seven years.

    base_period : list
        Earliest and latest times to use for standardization.

    area_weight : bool
        If True, multiply by cos latitude weights before taking
        mean over region.

    lat_name : str
        Name of latitude coordinate.

    lon_name : str
        Name of longitude coordinate.

    time_name : str
        Name of time coordinate.

    Returns
    -------
    index : xarray.DataArray
        Array containing the values of the dipole mode index.
    """

    # Ensure that the data provided is a data array
    sst = check_data_array(sst)

    # Get coordinate names
    if lat_name is None:
        lat_name = get_coordinate_standard_name(sst, 'lat')

    if lon_name is None:
        lon_name = get_coordinate_standard_name(sst, 'lon')

    if time_name is None:
        time_name = get_coordinate_standard_name(sst, 'time')

    if frequency is None:
        frequency = 'monthly'

    if frequency not in ('daily', 'monthly'):
        raise ValueError("Unsupported frequency '%r'" % frequency)

    western_bounds = {'lat' : [-10.0, 10.0], 'lon' : [50.0, 70.0]}
    eastern_bounds = {'lat' : [-10.0, 0.0], 'lon' : [90.0, 110.0]}

    if sst[lat_name].values[0] > sst[lat_name].values[-1]:
        left_index = 1
        right_index = 0
    else:
        left_index = 0
        right_index = 1

    western_box = {lat_name : slice(western_bounds['lat'][left_index],
                                    western_bounds['lat'][right_index]),
                   lon_name : slice(western_bounds['lon'][0],
                                    western_bounds['lon'][1])}
    eastern_box = {lat_name : slice(eastern_bounds['lat'][left_index],
                                    eastern_bounds['lat'][right_index]),
                   lon_name : slice(eastern_bounds['lon'][0],
                                    eastern_bounds['lon'][1])}

    western_sst = sst.sel(western_box)
    eastern_sst = sst.sel(eastern_box)

    western_sst_anom = _calculate_monthly_anomaly(
        western_sst, apply_filter=apply_filter, base_period=base_period,
        lat_name=lat_name, lon_name=lon_name, time_name=time_name)

    if area_weight:
        weights = np.cos(np.deg2rad(western_sst_anom[lat_name]))
        western_sst_anom = (western_sst_anom * weights)

    western_sst_anom = western_sst_anom.mean(dim=[lat_name, lon_name])

    eastern_sst_anom = _calculate_monthly_anomaly(
        eastern_sst, apply_filter=apply_filter, base_period=base_period,
        lat_name=lat_name, lon_name=lon_name, time_name=time_name)

    if area_weight:
        weights = np.cos(np.deg2rad(eastern_sst_anom[lat_name]))
        eastern_sst_anom = (eastern_sst_anom * weights)

    eastern_sst_anom = eastern_sst_anom.mean(dim=[lat_name, lon_name])

    index = (western_sst_anom - eastern_sst_anom).rename('dmi')

    if frequency == 'daily':
        index = index.resample({time_name : '1D'}).interpolate('linear')

    return index


def zwi(uwnd, frequency='monthly', apply_filter=False, base_period=None,
        area_weight=False, lat_name=None, lon_name=None, time_name=None):
    """Calculate dipole mode index.

    The ZWI introduced by Saji et al is defined as the
    area-averaged surface zonal wind anomalies in the
    central and eastern tropical Indian ocean,
    defined as the region 5S - 5N, 70E - 90E.

    See Saji, N. H. et al, "A dipole mode in the tropical
    Indian Ocean", Nature 401, 360 - 363 (1999).

    Parameters
    ----------
    uwnd : xarray.DataArray
        Array containing zonal wind values.

    frequency : None | 'daily' | 'monthly'
        Frequency to calculate index at.

    apply_filter : bool
        If True, filter the data by applying a three month
        running mean and remove harmonics with periods longer
        than seven years.

    base_period : list
        Earliest and latest times to use for standardization.

    area_weight : bool
        If True, multiply by cos latitude weights before taking
        mean over region.

    lat_name : str
        Name of latitude coordinate.

    lon_name : str
        Name of longitude coordinate.

    time_name : str
        Name of time coordinate.

    Returns
    -------
    index : xarray.DataArray
        Array containing the values of the zonal wind index.
    """

    # Ensure that the data provided is a data array
    uwnd = check_data_array(uwnd)

    # Get coordinate names
    if lat_name is None:
        lat_name = get_coordinate_standard_name(uwnd, 'lat')

    if lon_name is None:
        lon_name = get_coordinate_standard_name(uwnd, 'lon')

    if time_name is None:
        time_name = get_coordinate_standard_name(uwnd, 'time')

    if frequency is None:
        frequency = 'monthly'

    if frequency not in ('daily', 'monthly'):
        raise ValueError("Unsupported frequency '%r'" % frequency)

    region = {lat_name : slice(-5.0, 5.0),
              lon_name : slice(70.0, 90.0)}

    if uwnd[lat_name].values[0] > uwnd[lat_name].values[-1]:
        region[lat_name] = slice(5.0, -5.0)

    uwnd = uwnd.sel(region)

    uwnd_anom = _calculate_monthly_anomaly(
        uwnd, apply_filter=apply_filter, base_period=base_period,
        lat_name=lat_name, lon_name=lon_name, time_name=time_name)

    if area_weight:
        weights = np.cos(np.deg2rad(uwnd_anom[lat_name]))
        uwnd_anom = (uwnd_anom * weights)

    index = uwnd_anom.mean(dim=[lat_name, lon_name]).rename('zwi')

    if frequency == 'daily':
        index = index.resample({time_name : '1D'}).interpolate('linear')

    return index
