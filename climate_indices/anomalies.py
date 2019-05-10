import numpy as np


from .utils._dates import (NUMBER_OF_DAYS, NUMBER_OF_MONTHS,
                           _contains_leap_days, _get_day_of_year)
from .utils._validation import (_check_array_shape, _check_matching_lengths)


def center_data(data, ignore_nan=True, dtype=None):
    if ignore_nan:
        calc_mean = np.nanmean
    else:
        calc_mean = np.mean
    avg = calc_mean(data, axis=0, dtype=dtype)
    return data - avg[np.newaxis, ...]


def daily_anomalies(time, data, clim):
    """Calculate anomalies with respect to daily climatology."""
    _check_matching_lengths(time, data, 'daily_anomalies')

    has_leap_days = _contains_leap_days(time)
    if has_leap_days:
        expected_clim_shape = (NUMBER_OF_DAYS,) + data.shape[1:]
    else:
        expected_clim_shape = (NUMBER_OF_DAYS - 1,) + data.shape[1:]
    _check_array_shape(clim, expected_clim_shape, 'daily_anomalies')

    ydays = np.array([_get_day_of_year(t) - 1 for t in time], dtype='i8')
    unique_ydays = np.unique(ydays)

    means = np.empty(np.shape(data))
    for d in unique_ydays:
        means[ydays == d] = clim[d]

    return data - means


def standardized_daily_anomalies(time, data, clim, normalization=None, ddof=1):
    """Calculate standardized anomalies with respect to daily climatology."""
    anom = daily_anomalies(time, data, clim)
    if normalization is None:
        normalization = np.std(anom, axis=0, ddof=ddof, keepdims=True)
        return anom / normalization, normalization
    else:
        return anom / normalization


def monthly_anomalies(time, data, clim):
    """Calculate anomalies with respect to monthly climatology."""
    _check_matching_lengths(time, data, 'monthly_anomalies')

    expected_clim_shape = (NUMBER_OF_MONTHS,) + data.shape[1:]
    _check_array_shape(clim, expected_clim_shape, 'monthly_anomalies')

    months = np.array([t.month for t in time], dtype='i8')

    means = np.empty(np.shape(data))
    for i in range(NUMBER_OF_MONTHS):
        means[months == i + 1] = clim[i]
    return data - means


def standardize_series(data, axis=0, ddof=0):
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.std(data, axis=axis, ddof=ddof, keepdims=True)
    z = (data - mean) / std
    return z, mean, std


def standardized_monthly_anomalies(time, data, clim,
                                   normalization=None, ddof=1):
    """Calculate standardized anomalies with respect to monthly climatology."""
    anom = monthly_anomalies(time, data, clim)
    if normalization is None:
        normalization = np.std(anom, axis=0, ddof=ddof, keepdims=True)
        return anom / normalization, normalization
    else:
        return anom / normalization


__all__ = ['center_data', 'daily_anomalies', 'monthly_anomalies',
           'standardized_daily_anomalies', 'standardized_monthly_anomalies']
