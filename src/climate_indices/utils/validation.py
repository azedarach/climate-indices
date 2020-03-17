"""
Helper routines for validating inputs.
"""


from __future__ import absolute_import


import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr


from .defaults import get_coordinate_standard_name


def is_data_array(data):
    """Check if object is an xarray DataArray."""

    return isinstance(data, xr.DataArray)


def is_dataset(data):
    """Check if object is an xarray Dataset."""

    return isinstance(data, xr.Dataset)


def is_xarray_object(data):
    """Check if object is an xarray DataArray or Dataset."""

    return is_data_array(data) or is_dataset(data)


def is_dask_array(data):
    """Check if object is a dask array."""

    return isinstance(data, da.Array)


def detect_frequency(data, time_name=None):
    """Detect if the data is sampled at daily or monthly resolution."""

    if time_name is None:
        time_name = get_coordinate_standard_name(data, 'time')

    inferred_frequency = pd.infer_freq(data[time_name].values[:3])

    if inferred_frequency is not None:

        if inferred_frequency in ('H', '1H'):
            return 'hourly'

        if inferred_frequency in ('D', '1D'):
            return 'daily'

        if inferred_frequency in ('1M', '1MS', 'MS'):
            return 'monthly'

        if inferred_frequency in ('1A', '1AS', '1BYS'):
            return 'yearly'

        raise ValueError('Unable to detect data frequency')

    else:
        # Take the difference between the second and first time values
        dt = data[time_name][1] - data[time_name][0]

        # Covert and evaluate
        dt = dt.data.astype('timedelta64[D]').astype(int)

        if dt == 1:
            return 'daily'

        if 28 <= dt < 365:
            return 'monthly'

        if dt >= 365:
            return 'yearly'

        raise ValueError('Unable to detect data frequency.')


def is_daily_data(data, time_name=None):
    """Check if data is sampled at daily resolution."""

    return detect_frequency(data, time_name=time_name) == 'daily'


def is_monthly_data(data, time_name=None):
    """Check if data is sampled at monthly resolution."""

    return detect_frequency(data, time_name=time_name) == 'monthly'


def check_unit_axis_sums(x, whom, axis=0):
    """Check sums along array axis are one."""

    axis_sums = x.sum(axis=axis)

    if not np.all(np.isclose(axis_sums, 1)):
        raise ValueError(
            'Array with incorrect axis sums passed to %s. '
            'Expected sums along axis %d to be 1.'
            % (whom, axis))


def check_array_shape(x, shape, whom):
    """Check array has the desired shape."""

    if x.shape != shape:
        raise ValueError(
            'Array with wrong shape passed to %s. '
            'Expected %s, but got %s' % (whom, shape, x.shape))


def check_data_array(obj, variable=None):
    """Check that object is an xarray DataArray."""

    if is_data_array(obj):
        return obj

    if is_dataset(obj) and variable is not None:
        return obj[variable]

    input_type = type(obj)
    raise TypeError("Given object is of type '%r'" % input_type)


def check_base_period(data, base_period=None, time_name=None):
    """Get list containing limits of base period."""

    if time_name is None:
        time_name = get_coordinate_standard_name(data, 'time')

    if base_period is None:
        base_period = [data[time_name].min(), data[time_name].max()]
    else:
        if len(base_period) != 2:
            raise ValueError(
                'Incorrect length for base period: expected length 2 list '
                'but got %r' % base_period)

        base_period = sorted(base_period)

    return base_period


def has_fixed_missing_values(X, axis=0):
    """Check if NaN values occur in fixed features throughout array."""

    if is_dask_array(X):
        nan_mask = da.isnan(X)
    else:
        nan_mask = np.isnan(X)

    return (nan_mask.any(axis=axis) == nan_mask.all(axis=axis)).all()


def check_fixed_missing_values(X, axis=0):
    """Check if array has fixed missing values."""

    if not has_fixed_missing_values(X, axis=axis):
        raise ValueError(
            'variable has partial missing values')


def get_valid_variables(X):
    """Remove all-missing columns and record indices of non-missing features."""

    if is_dask_array(X):
        valid_vars = da.nonzero(da.logical_not(
            da.isnan(X[0])))[0].compute()
        valid_data = X[:, valid_vars]
    else:
        valid_vars = np.where(np.logical_not(np.isnan(X[0])))[0]
        valid_data = X[:, valid_vars]

    return valid_data, valid_vars
