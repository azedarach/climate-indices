"""
Provides routines for computing MJO indices.
"""


from copy import deepcopy
import numbers

import dask
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr


from scipy.fft import irfft, rfft
from scipy.stats import linregress


from ..utils import (check_base_period, check_data_array,
                     check_fixed_missing_values, detect_frequency,
                     eofs, get_coordinate_standard_name,
                     get_valid_variables, is_daily_data,
                     is_dask_array, meridional_mean)


INTEGER_TYPES = (numbers.Integral, np.integer)


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


def _compute_smooth_seasonal_cycle(seasonal_cycle, n_harmonics=4,
                                   workers=None, sample_dim='dayofyear'):
    """Calculate smooth seasonal cycle based on lowest order harmonics."""

    # Convert to flat array for performing FFT
    feature_dims = [d for d in seasonal_cycle.dims if d != sample_dim]

    if not feature_dims:
        original_shape = None
    else:
        original_shape = [seasonal_cycle.sizes[d] for d in feature_dims]

    sample_dim_pos = seasonal_cycle.get_axis_num(sample_dim)
    if sample_dim_pos != 0:
        seasonal_cycle = seasonal_cycle.transpose(
            *([sample_dim] + feature_dims))

    n_samples = seasonal_cycle.sizes[sample_dim]

    if feature_dims:
        n_features = np.product(original_shape)
    else:
        n_features = 1

    flat_data = seasonal_cycle.values.reshape((n_samples, n_features))

    check_fixed_missing_values(flat_data, axis=0)

    valid_data, valid_features = get_valid_variables(flat_data)

    valid_data = valid_data.swapaxes(0, 1)

    if is_dask_array(valid_data):

        spectrum = da.fft.rfft(valid_data, axis=-1)

        def _filter(freqs):
            n_freqs = freqs.shape[0]
            return da.concatenate(
                [freqs[:n_harmonics], da.zeros(n_freqs - n_harmonics)])

        filtered_spectrum = da.apply_along_axis(_filter, axis=-1, arr=spectrum)

        filtered_valid_data = da.fft.irfft(
            filtered_spectrum, n=n_samples, axis=-1).swapaxes(0, 1)

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
        spectrum = rfft(valid_data, axis=-1, workers=workers)
        spectrum[..., n_harmonics:] = 0.0

        filtered_valid_data = irfft(
            spectrum, n=n_samples, axis=-1, workers=workers).swapaxes(0, 1)

        filtered_data = np.full((n_samples, n_features), np.NaN)
        filtered_data[:, valid_features] = filtered_valid_data

    if original_shape:
        filtered_data = filtered_data.reshape([n_samples,] + original_shape)
        filtered_dims = [sample_dim] + feature_dims
    else:
        filtered_data = filtered_data.ravel()
        filtered_dims = [sample_dim]

    filtered_coords = deepcopy(seasonal_cycle.coords)

    smooth_seasonal_cycle = xr.DataArray(
        filtered_data, coords=filtered_coords, dims=filtered_dims)

    if sample_dim_pos != 0:
        smooth_seasonal_cycle = smooth_seasonal_cycle.transpose(
            *seasonal_cycle.dims)

    return smooth_seasonal_cycle


def _subtract_seasonal_cycle(data, base_period=None,
                             n_harmonics=4, workers=None, time_name=None):
    """Remove time mean and leading harmonics of annual cycle."""

    # Ensure input data is a data array.
    data = check_data_array(data)

    if time_name is None:
        time_name = get_coordinate_standard_name(data, 'time')

    base_period = check_base_period(data, base_period=base_period,
                                    time_name=time_name)

    # Restrict to base period for computing annual cycle.
    base_period_data = data.where(
        (data[time_name] >= base_period[0]) &
        (data[time_name] <= base_period[1]), drop=True)

    input_frequency = detect_frequency(data, time_name=time_name)

    if input_frequency == 'daily':
        base_period_groups = base_period_data[time_name].dt.dayofyear
        groups = data[time_name].dt.dayofyear
        sample_dim = 'dayofyear'
    elif input_frequency == 'monthly':
        base_period_groups = base_period_data[time_name].dt.month
        groups = data[time_name].dt.month
        sample_dim = 'month'
    else:
        raise ValueError("Unsupported sampling rate '%r'" % input_frequency)

    seasonal_cycle = base_period_data.groupby(
        base_period_groups).mean(time_name)

    smooth_seasonal_cycle = _compute_smooth_seasonal_cycle(
        seasonal_cycle, n_harmonics=n_harmonics, sample_dim=sample_dim)

    # Subtract seasonal cycle and leading higher harmonics
    return data.groupby(groups) - smooth_seasonal_cycle


def _subtract_monthly_linear_regression(data, predictor,
                                        base_period=None, time_name=None):
    """Subtract regressed values of predictand at each grid point."""

    data = check_data_array(data)
    predictor = check_data_array(predictor)

    if time_name is None:
        time_name = get_coordinate_standard_name(data, 'time')

    if time_name not in predictor.dims:
        raise ValueError(
            "Could not find time dimension '%s' in predictor array" %
            time_name)

    base_period = check_base_period(data, base_period=base_period,
                                    time_name=time_name)

    outcome_frequency = detect_frequency(data, time_name=time_name)

    if outcome_frequency != 'daily':
        raise ValueError('Outcome data must be daily resolution')

    predictor_frequency = detect_frequency(predictor, time_name=time_name)

    if predictor_frequency != 'daily':
        predictor = predictor.resample({time_name : '1D'}).interpolate('linear')

    # Align daily mean predictor times with outcome times
    def _floor_to_daily_resolution(t):

        if isinstance(t, np.datetime64):

            return np.datetime64(pd.to_datetime(t).floor('D'))

        raise ValueError("Unsupported time value '%r'" % t)

    outcomes_start_date = _floor_to_daily_resolution(
        data[time_name].min().values)
    outcomes_end_date = _floor_to_daily_resolution(
        data[time_name].max().values)

    predictor = predictor.where(
        (predictor[time_name] >= outcomes_start_date) &
        (predictor[time_name] <= outcomes_end_date), drop=True)

    if predictor.sizes[time_name] != data.sizes[time_name]:
        raise RuntimeError('Incorrect number of predictor values given')

    predictor[time_name] = data[time_name]

    base_period_outcome = data.where(
        (data[time_name] >= base_period[0]) &
        (data[time_name] <= base_period[1]), drop=True)

    base_period_predictor = predictor.where(
        (predictor[time_name] >= base_period[0]) &
        (predictor[time_name] <= base_period[1]), drop=True)

    # Ensure time-points are aligned.
    base_period_outcome = base_period_outcome.resample(
        {time_name : '1D'}).mean()
    base_period_predictor = base_period_predictor.resample(
        {time_name : '1D'}).mean()

    # Fit seasonal regression relationship for each month
    slopes = xr.zeros_like(
        base_period_outcome.isel(
            {time_name : 0}, drop=True)).expand_dims(
                {time_name: pd.date_range('2000-01-01', '2001-01-01', freq='1MS')})
    intercepts = xr.zeros_like(
        base_period_outcome.isel(
            {time_name : 0}, drop=True)).expand_dims(
                {time_name: pd.date_range('2000-01-01', '2001-01-01', freq='1MS')})

    if dask.is_dask_collection(base_period_outcome):
        _apply_along_axis = da.apply_along_axis
        apply_kwargs = dict(dtype=base_period_outcome.dtype, shape=(2,))
    else:
        _apply_along_axis = np.apply_along_axis
        apply_kwargs = {}

    def _fit_along_axis(outcomes, predictor, axis=-1, **kwargs):
        def _fit(y):
            return np.array(linregress(predictor, y)[:2])
        return _apply_along_axis(_fit, axis=axis, arr=outcomes, **kwargs)

    for month in range(1, 13):

        predictor_vals = base_period_predictor.where(
            base_period_predictor[time_name].dt.month == month, drop=True)
        outcome_vals = base_period_outcome.where(
            base_period_outcome[time_name].dt.month == month, drop=True)

        fit_result = xr.apply_ufunc(
            _fit_along_axis, outcome_vals, predictor_vals,
            input_core_dims=[[time_name], [time_name]],
            output_core_dims=[['fit_coefficients']],
            dask='allowed', kwargs=apply_kwargs,
            output_dtypes=[outcome_vals.dtype])

        slopes = xr.where(slopes[time_name].dt.month == month,
                          fit_result.isel(fit_coefficients=0),
                          slopes)
        intercepts = xr.where(intercepts[time_name].dt.month == month,
                              fit_result.isel(fit_coefficients=1),
                              intercepts)

    slopes = slopes.resample(
        {time_name : '1D'}).interpolate('linear').isel(
            {time_name : slice(0, -1)})
    slopes = slopes.assign_coords(
        {time_name : slopes[time_name].dt.dayofyear}).rename(
            {time_name :'dayofyear'})

    intercepts = intercepts.resample(
        {time_name : '1D'}).interpolate('linear').isel(
            {time_name : slice(0, -1)})
    intercepts = intercepts.assign_coords(
        {time_name : intercepts[time_name].dt.dayofyear}).rename(
            {time_name : 'dayofyear'})

    subtracted = data.groupby(data[time_name].dt.dayofyear) - intercepts

    def _subtract(x):
        day = x[time_name].dt.dayofyear[0]
        slope = slopes.sel(dayofyear=day)
        predictor_values = predictor.where(
            predictor[time_name].dt.dayofyear == day, drop=True)
        return x - predictor_values * slope

    subtracted = subtracted.groupby(subtracted[time_name].dt.dayofyear).map(_subtract)

    return subtracted


def _subtract_running_mean(data, n_steps, time_name=None):
    """Subtract running mean from data."""

    data = check_data_array(data)

    if time_name is None:
        time_name = get_coordinate_standard_name(data, 'time')

    if dask.is_dask_collection(data):
        chunk_sizes = data.data.chunksize

        if chunk_sizes[data.get_axis_num(time_name)] < n_steps:
            data = data.chunk({time_name : n_steps})

    running_mean = data.rolling({time_name : n_steps}).mean()

    return data - running_mean


def wh_rmm_anomalies(olr, u850, u250, base_period=None,
                     enso_index=None, subtract_running_mean=True,
                     n_running_mean_steps=120, lat_bounds=None,
                     lat_name=None, lon_name=None, time_name=None):
    """Calculate anomaly inputs for Wheeler-Hendon MJO index.

    Parameters
    ----------
    olr : xarray.DataArray
        Array containing top-of-atmosphere OLR values.

    u850 : xarray.DataArray
        Array containing values of zonal wind at 850 hPa.

    u250 : xarray.DataArray
        Array containing values of zonal wind at 250 hPa.

    base_period : list
        Earliest and latest times to use for standardization.

    enso_index : xarray.DataArray, optional
        If given, an array containing an ENSO index time-series
        used to remove ENSO related variability (e.g., monthly
        SST1 index values).

    subtract_running_mean : boolean, optional
        If True (default), subtract running mean of the previous
        days for each time-step.

    n_running_mean_steps : integer
        Number of days to include in running mean, by default 120.

    lat_bounds : list
        Latitude range over which to compute meridional mean,
        by default 15S to 15N.

    lat_name : str
        Name of the latitude coordinate.

    lon_name : str
        Name of the longitude coordinate.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    olr_anom : xarray.DataArray
        Array containing the values of the OLR anomalies.

    u850_anom : xarray.DataArray
        Array containing the values of the 850 hPa zonal wind
        anomalies.

    u250_anom : xarray.DataArray
        Array containing the values of the 250 hPa zonal wind anomalies.
    """

    # Ensure all inputs are data arrays.
    olr = check_data_array(olr)
    u850 = check_data_array(u850)
    u250 = check_data_array(u250)

    if enso_index is not None:
        enso_index = check_data_array(enso_index)

    # Index is defined in terms of daily resolution data.
    if not is_daily_data(olr):
        raise ValueError('Initial OLR data must be daily resolution')

    if not is_daily_data(u850):
        raise ValueError('Initial u850 data must be daily resolution')

    if not is_daily_data(u250):
        raise ValueError('Initial u250 data must be daily resolution')

    # Get coordinate names
    if lat_name is None:
        lat_name = get_coordinate_standard_name(olr, 'lat')

    if lon_name is None:
        lon_name = get_coordinate_standard_name(olr, 'lon')

    if time_name is None:
        time_name = get_coordinate_standard_name(olr, 'time')

    # For convenience, ensure all inputs have same coordinate names.
    _check_consistent_coordinate_names(
        olr, u850, u250, lat_name=lat_name, lon_name=lon_name,
        time_name=time_name)

    # Restrict to equatorial region.
    if lat_bounds is None:
        lat_bounds = [-15.0, 15.0]
    else:
        if len(lat_bounds) != 2:
            raise ValueError(
                'Latitude boundaries must be a 2 element list, but got %r' %
                lat_bounds)
        lat_bounds = sorted(lat_bounds)

    olr = olr.where(
        (olr[lat_name] >= lat_bounds[0]) &
        (olr[lat_name] <= lat_bounds[1]), drop=True).squeeze()
    u850 = u850.where(
        (u850[lat_name] >= lat_bounds[0]) &
        (u850[lat_name] <= lat_bounds[1]), drop=True).squeeze()
    u250 = u250.where(
        (u250[lat_name] >= lat_bounds[0]) &
        (u250[lat_name] <= lat_bounds[1]), drop=True).squeeze()

    # Ensure inputs cover same time period.
    start_time = max(
        [olr[time_name].min(), u850[time_name].min(),
         u250[time_name].min()])
    end_time = min(
        [olr[time_name].max(), u850[time_name].max(),
         u250[time_name].max()])

    def _restrict_time_period(data):
        return data.where(
            (data[time_name] >= start_time) &
            (data[time_name] <= end_time), drop=True)

    olr = _restrict_time_period(olr)
    u850 = _restrict_time_period(u850)
    u250 = _restrict_time_period(u250)

    base_period = check_base_period(olr, base_period=base_period,
                                    time_name=time_name)

    # Remove seasonal cycle
    olr_anom = _subtract_seasonal_cycle(
        olr, base_period=base_period, time_name=time_name)
    u850_anom = _subtract_seasonal_cycle(
        u850, base_period=base_period, time_name=time_name)
    u250_anom = _subtract_seasonal_cycle(
        u250, base_period=base_period, time_name=time_name)

    # If required, remove ENSO variability
    if enso_index is not None:

        olr_anom = _subtract_monthly_linear_regression(
            olr_anom, predictor=enso_index,
            base_period=base_period, time_name=time_name)
        u850_anom = _subtract_monthly_linear_regression(
            u850_anom, predictor=enso_index,
            base_period=base_period, time_name=time_name)
        u250_anom = _subtract_monthly_linear_regression(
            u250_anom, predictor=enso_index,
            base_period=base_period, time_name=time_name)

    # If required, apply running mean
    if subtract_running_mean:

        olr_anom = _subtract_running_mean(
            olr_anom, n_steps=n_running_mean_steps, time_name=time_name).dropna(
                time_name, how='all')
        u850_anom = _subtract_running_mean(
            u850_anom, n_steps=n_running_mean_steps, time_name=time_name).dropna(
                time_name, how='all')
        u250_anom = _subtract_running_mean(
            u250_anom, n_steps=n_running_mean_steps, time_name=time_name).dropna(
                time_name, how='all')

    olr_anom = meridional_mean(olr_anom, lat_bounds=lat_bounds,
                               lat_name=lat_name)
    u850_anom = meridional_mean(u850_anom, lat_bounds=lat_bounds,
                                lat_name=lat_name)
    u250_anom = meridional_mean(u250_anom, lat_bounds=lat_bounds,
                                lat_name=lat_name)

    return olr_anom, u850_anom, u250_anom


def wh_rmm_eofs(olr_anom, u850_anom, u250_anom, n_modes=2,
                lon_name=None, time_name=None):
    """Calculate combined EOFs of OLR and zonal wind anomalies.

    Parameters
    ----------
    olr_anom : xarray.DataArray
        Array containing values of OLR anomalies.

    u850_anom : xarray.DataArray
        Array containing values of 850 hPa zonal wind anomalies.

    u250_anom : xarray.DataArray
        Array containing values of 250 hPa zonal wind anomalies.

    n_modes : integer
        Number of EOF modes to calculate. If None, by default only
        the leading two modes are computed.

    lon_name : str
        Name of the longitude coordinate.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    olr_eofs : xarray.Dataset
        Dataset containing the calculated OLR EOFs.

    u850_eofs : xarray.Dataset
        Dataset containing the calculated 850 hPa zonal wind EOFs.

    u250_eofs : xarray.Dataset
        Dataset containing the calculated 250 hPa zonal wind EOFs.
    """

    olr_anom = check_data_array(olr_anom)
    u850_anom = check_data_array(u850_anom)
    u250_anom = check_data_array(u250_anom)

    if lon_name is None:
        lon_name = get_coordinate_standard_name(olr_anom, 'lon')

    if time_name is None:
        time_name = get_coordinate_standard_name(olr_anom, 'time')

    if lon_name not in u850_anom.dims or lon_name not in u250_anom.dims:
        raise ValueError(
            "Could not find longitude coordinate '%s' in zonal wind data" %
            time_name)

    if time_name not in u850_anom.dims or time_name not in u250_anom.dims:
        raise ValueError(
            "Could not find time coordinate '%s' in zonal wind data" %
            time_name)

    if n_modes is None:
        n_modes = 2

    if not isinstance(n_modes, INTEGER_TYPES) or n_modes < 1:
        raise ValueError('Number of modes must be a positive integer.')

    olr_eofs, u850_eofs, u250_eofs = eofs(
        olr_anom, u850_anom, u250_anom,
        n_modes=n_modes, sample_dim=time_name)

    # Define the leading mode such that it corresponds to negative
    # OLR anomalies over the maritime continent.
    lon_bounds = [60.0, 180.0]

    leading_olr_eof = olr_eofs['EOFs'].sel(mode=0)
    olr_max_anom = leading_olr_eof.where(
        (leading_olr_eof[lon_name] >= lon_bounds[0]) &
        (leading_olr_eof[lon_name] <= lon_bounds[1])).max().item()
    olr_min_anom = leading_olr_eof.where(
        (leading_olr_eof[lon_name] >= lon_bounds[0]) &
        (leading_olr_eof[lon_name] <= lon_bounds[1])).min().item()

    if np.abs(olr_max_anom) > np.abs(olr_min_anom):
        olr_eofs['EOFs'] = xr.where(
            olr_eofs['mode'] == 0, -olr_eofs['EOFs'], olr_eofs['EOFs'])
        olr_eofs['PCs'] = xr.where(
            olr_eofs['mode'] == 0, -olr_eofs['PCs'], olr_eofs['PCs'])

        u850_eofs['EOFs'] = xr.where(
            u850_eofs['mode'] == 0, -u850_eofs['EOFs'], u850_eofs['EOFs'])
        u850_eofs['PCs'] = xr.where(
            u850_eofs['mode'] == 0, -u850_eofs['PCs'], u850_eofs['PCs'])

        u250_eofs['EOFs'] = xr.where(
            u250_eofs['mode'] == 0, -u250_eofs['EOFs'], u250_eofs['EOFs'])
        u250_eofs['PCs'] = xr.where(
            u250_eofs['mode'] == 0, -u250_eofs['PCs'], u250_eofs['PCs'])

    # Similarly, define the second leading mode to have
    # positive OLR anomalies over the maritime continent.
    if n_modes > 1:
        second_olr_eof = olr_eofs['EOFs'].sel(mode=1)

        olr_max_anom = second_olr_eof.where(
            (second_olr_eof[lon_name] >= lon_bounds[0]) &
            (second_olr_eof[lon_name] <= lon_bounds[1])).max().item()
        olr_min_anom = second_olr_eof.where(
            (second_olr_eof[lon_name] >= lon_bounds[0]) &
            (second_olr_eof[lon_name] <= lon_bounds[1])).min().item()

        if np.abs(olr_min_anom) > np.abs(olr_max_anom):
            olr_eofs['EOFs'] = xr.where(
                olr_eofs['mode'] == 1, -olr_eofs['EOFs'], olr_eofs['EOFs'])
            olr_eofs['PCs'] = xr.where(
                olr_eofs['mode'] == 1, -olr_eofs['PCs'], olr_eofs['PCs'])

            u850_eofs['EOFs'] = xr.where(
                u850_eofs['mode'] == 1, -u850_eofs['EOFs'], u850_eofs['EOFs'])
            u850_eofs['PCs'] = xr.where(
                u850_eofs['mode'] == 1, -u850_eofs['PCs'], u850_eofs['PCs'])

            u250_eofs['EOFs'] = xr.where(
                u250_eofs['mode'] == 1, -u250_eofs['EOFs'], u250_eofs['EOFs'])
            u250_eofs['PCs'] = xr.where(
                u250_eofs['mode'] == 1, -u250_eofs['PCs'], u250_eofs['PCs'])

    return olr_eofs, u850_eofs, u250_eofs


def wh_rmm(olr, u850, u250, enso_index=None, base_period=None,
           subtract_running_mean=True, n_running_mean_steps=120,
           n_modes=2, lat_bounds=None, lat_name=None, lon_name=None,
           time_name=None):
    """Calculates the Wheeler-Hendon MJO index.

    See Wheeler, M. C. and Hendon, H., "An All-Season Real-Time Multivariate
    MJO Index: Development of an Index for Monitoring and Prediction",
    Monthly Weather Review 132, 1917 - 1932 (2004),
    doi:

    Parameters
    ----------
    olr : xarray.DataArray
        Array containing top-of-atmosphere OLR values.

    u850 : xarray.DataArray
        Array containing values of zonal wind at 850 hPa.

    u250 : xarray.DataArray
        Array containing values of zonal wind at 250 hPa.

    enso_index : xarray.DataArray, optional
        If given, an array containing an ENSO index time-series
        used to remove ENSO related variability (e.g., monthly
        SST1 index values).

    base_period : list
        Earliest and latest times to use for standardization.

    subtract_running_mean : boolean, optional
        If True (default), subtract running mean of the previous
        days for each time-step.

    n_running_mean_steps : integer
        Number of days to include in running mean, by default 120.

    lat_bounds : list
        Latitude range over which to compute meridional mean,
        by default 15S to 15N.

    lat_name : str
        Name of the latitude coordinate.

    lon_name : str
        Name of the longitude coordinate.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    result : xarray.Dataset
        Dataset containing the RMM EOF patterns and the components of the
        index.
    """

    # Ensure all inputs are data arrays.
    olr = check_data_array(olr)
    u850 = check_data_array(u850)
    u250 = check_data_array(u250)

    if enso_index is not None:
        enso_index = check_data_array(enso_index)

    # Index is defined in terms of daily resolution data.
    if not is_daily_data(olr):
        raise ValueError('Initial OLR data must be daily resolution')

    if not is_daily_data(u850):
        raise ValueError('Initial u850 data must be daily resolution')

    if not is_daily_data(u250):
        raise ValueError('Initial u250 data must be daily resolution')

    # Get coordinate names
    if lat_name is None:
        lat_name = get_coordinate_standard_name(olr, 'lat')

    if lon_name is None:
        lon_name = get_coordinate_standard_name(olr, 'lon')

    if time_name is None:
        time_name = get_coordinate_standard_name(olr, 'time')

    # For convenience, ensure all inputs have same coordinate names.
    _check_consistent_coordinate_names(
        olr, u850, u250, lat_name=lat_name, lon_name=lon_name,
        time_name=time_name)

    if n_modes is None:
        n_modes = 2

    if not isinstance(n_modes, INTEGER_TYPES) or n_modes < 1:
        raise ValueError('Number of modes must be a positive integer.')

    # Ensure inputs cover same time period.
    start_time = max(
        [olr[time_name].min(), u850[time_name].min(),
         u250[time_name].min()])
    end_time = min(
        [olr[time_name].max(), u850[time_name].max(),
         u250[time_name].max()])

    def _restrict_time_period(data):
        return data.where(
            (data[time_name] >= start_time) &
            (data[time_name] <= end_time), drop=True)

    olr = _restrict_time_period(olr)
    u850 = _restrict_time_period(u850)
    u250 = _restrict_time_period(u250)

    base_period = check_base_period(olr, base_period=base_period,
                                    time_name=time_name)

    olr_anom, u850_anom, u250_anom = wh_rmm_anomalies(
        olr, u850, u250, base_period=base_period,
        enso_index=enso_index, subtract_running_mean=subtract_running_mean,
        n_running_mean_steps=n_running_mean_steps, lat_bounds=lat_bounds,
        lat_name=lat_name, lon_name=lon_name, time_name=time_name)

    base_period_olr_anom = olr_anom.where(
        (olr_anom[time_name] >= base_period[0]) &
        (olr_anom[time_name] <= base_period[1]), drop=True)
    base_period_u850_anom = u850_anom.where(
        (u850_anom[time_name] >= base_period[0]) &
        (u850_anom[time_name] <= base_period[1]), drop=True)
    base_period_u250_anom = u250_anom.where(
        (u250_anom[time_name] >= base_period[0]) &
        (u250_anom[time_name] <= base_period[1]), drop=True)

    olr_normalization = np.std(base_period_olr_anom)
    u850_normalization = np.std(base_period_u850_anom)
    u250_normalization = np.std(base_period_u250_anom)

    base_period_olr_anom = base_period_olr_anom / olr_normalization
    base_period_u850_anom = base_period_u850_anom / u850_normalization
    base_period_u250_anom = base_period_u250_anom / u250_normalization

    olr_eofs, u850_eofs, u250_eofs = wh_rmm_eofs(
        base_period_olr_anom, base_period_u850_anom, base_period_u250_anom,
        n_modes=n_modes, time_name=time_name)

    rmm1_normalization = olr_eofs['PCs'].sel(mode=0).std(time_name).item()
    rmm2_normalization = olr_eofs['PCs'].sel(mode=1).std(time_name).item()

    olr_pcs = (olr_anom / olr_normalization).dot(olr_eofs['EOFs'])
    u850_pcs = (u850_anom / u850_normalization).dot(u850_eofs['EOFs'])
    u250_pcs = (u250_anom / u250_normalization).dot(u250_eofs['EOFs'])

    rmm1 = (olr_pcs.sel(mode=0, drop=True) +
            u850_pcs.sel(mode=0, drop=True) +
            u250_pcs.sel(mode=0, drop=True)) / rmm1_normalization
    rmm2 = (olr_pcs.sel(mode=1, drop=True) +
            u850_pcs.sel(mode=1, drop=True) +
            u250_pcs.sel(mode=1, drop=True)) / rmm2_normalization

    rmm_olr_eofs = olr_eofs['EOFs'].reset_coords(
        [d for d in olr_eofs['EOFs'].coords if d not in ('mode', lon_name)],
        drop=True)
    rmm_u850_eofs = u850_eofs['EOFs'].reset_coords(
        [d for d in u850_eofs['EOFs'].coords if d not in ('mode', lon_name)],
        drop=True)
    rmm_u250_eofs = u250_eofs['EOFs'].reset_coords(
        [d for d in u250_eofs['EOFs'].coords if d not in ('mode', lon_name)],
        drop=True)

    data_vars = {'olr_eofs': rmm_olr_eofs,
                 'u850_eofs': rmm_u850_eofs,
                 'u250_eofs': rmm_u250_eofs,
                 'rmm1': rmm1,
                 'rmm2': rmm2}

    return xr.Dataset(data_vars)
