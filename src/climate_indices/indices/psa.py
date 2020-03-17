"""
Provides routines for calculating indices of the PSA.
"""


from __future__ import absolute_import, division

import numbers

import dask.array
import numpy as np
import scipy.linalg
import xarray as xr

from ..utils import (check_base_period, check_data_array,
                     detect_frequency, downsample_data,
                     eofs, get_coordinate_standard_name,
                     get_valid_variables, is_dask_array,
                     reofs, select_latlon_box,
                     standardized_anomalies)


INTEGER_TYPES = (numbers.Integral, np.integer)


def _project_onto_subspace(data, eofs, weight=None, sample_dim=None,
                           mode_dim='mode'):
    """Project given data onto (possibly non-orthogonal) basis."""

    # Ensure given data is a data array
    data = check_data_array(data)

    if sample_dim is None:
        sample_dim = get_coordinate_standard_name(data, 'time')

    if mode_dim is None:
        raise ValueError('No mode dimension given')

    if weight is not None:
        data = data * weight

    feature_dims = [d for d in data.dims if d != sample_dim]
    original_shape = [data.sizes[d] for d in feature_dims]

    if data.get_axis_num(sample_dim) != 0:
        data = data.transpose(*([sample_dim] + feature_dims))

    n_samples = data.sizes[sample_dim]
    n_features = np.product(original_shape)

    flat_data = data.values.reshape((n_samples, n_features))
    valid_data, valid_features = get_valid_variables(flat_data)
    valid_data = valid_data.swapaxes(0, 1)

    if eofs.get_axis_num(mode_dim) != 0:
        eofs = eofs.transpose(*([mode_dim] + feature_dims))

    n_modes = eofs.sizes[mode_dim]

    flat_eofs = eofs.values.reshape((n_modes, n_features))
    valid_eofs = flat_eofs[:, valid_features].swapaxes(0, 1)

    if is_dask_array(flat_data):

        projection = dask.array.linalg.lstsq(valid_eofs, valid_data)

    else:

        projection = scipy.linalg.lstsq(valid_eofs, valid_data)[0]

    projection = xr.DataArray(
        projection.swapaxes(0, 1),
        coords={sample_dim : data[sample_dim],
                mode_dim : eofs[mode_dim]},
        dims=[sample_dim, mode_dim])

    return projection


def _fix_psa1_phase(eofs_ds, real_psa1_mode, lat_name=None, lon_name=None):
    """Fix PSA1 phase such that negative anomalies occur in Pacific sector."""

    if lat_name is None:
        lat_name = get_coordinate_standard_name(eofs_ds, 'lat')

    if lon_name is None:
        lon_name = get_coordinate_standard_name(eofs_ds, 'lon')

    psa1_pattern = eofs_ds['EOFs'].sel(mode=real_psa1_mode, drop=True).squeeze()

    box = select_latlon_box(
        psa1_pattern, lat_bounds=[-70.0, -50.0],
        lon_bounds=[230.0, 250.0],
        lat_name=lat_name, lon_name=lon_name)

    box_max = box.max().item()
    box_min = box.min().item()

    if box_max < -box_min:

        eofs_ds['EOFs'] = -eofs_ds['EOFs']
        eofs_ds['PCs'] = -eofs_ds['PCs']

    return eofs_ds


def real_pc_psa1(hgt, frequency='monthly', base_period=None, rotate=False,
                 psa1_mode=1, n_modes=None, n_rotated_modes=None,
                 low_latitude_boundary=-20, lat_name=None, lon_name=None,
                 time_name=None):
    """Calculate real principal component based PSA1 index.

    Parameters
    ----------
    hgt : xarray.DataArray
        Array containing geopotential height values.

    frequency : 'daily' | 'monthly'
        Sampling rate at which to calculate the index.

    base_period : list
        If given, a two element list containing the earliest and
        latest dates to include in base period for standardization and
        calculation of EOFs. If None, then the full time period is used.

    rotate : bool
        If True, calculate rotated EOFs.

    psa1_mode : integer
        Mode to take as corresponding to the real PSA1 mode.

    n_modes : integer
        Number of EOF modes to calculate.

    n_rotated_modes : integer
        If computing rotated EOFs, number of modes to include in
        rotation.

    low_latitude_boundary : float
        Low-latitude boundary for analysis region.

    lat_name : str
        Name of the latitude coordinate.

    lon_name : str
        Name of the longitude coordinate.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    index : xarray.Dataset
        Dataset containing the PSA1 index and loading pattern.
    """

    # Ensure that the data provided is a data array.
    hgt = check_data_array(hgt)

    # Get coordinate names.
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

    if psa1_mode is None:
        psa1_mode = 1

    if not isinstance(psa1_mode, INTEGER_TYPES) or psa1_mode < 0:
        raise ValueError('PSA1 mode must be a positive integer')

    if n_modes is None:
        n_modes = 10

    if not isinstance(n_modes, INTEGER_TYPES) or n_modes <= psa1_mode:
        raise ValueError('Number of modes must be an integer greater than %d' %
                         psa1_mode)

    if n_rotated_modes is None:
        n_rotated_modes = n_modes

    if not isinstance(n_rotated_modes, INTEGER_TYPES) or n_rotated_modes <= psa1_mode:
        raise ValueError('Number of modes must be an integer greater than %d' %
                         psa1_mode)

    if input_frequency == 'daily' and frequency == 'daily':

        # Calculate EOFs and index using daily anomalies.
        base_period_hgt = hgt.where(
            (hgt[time_name] >= base_period[0]) &
            (hgt[time_name] <= base_period[1]), drop=True)

        hgt_clim = base_period_hgt.groupby(
            base_period_hgt[time_name].dt.dayofyear, drop=True).mean(time_name)

        hgt_anom = hgt.groupby(hgt[time_name].dt.dayofyear) - hgt_clim

        base_period_hgt_anom = hgt_anom.where(
            (hgt_anom[time_name] >= base_period[0]) &
            (hgt_anom[time_name] <= base_period[1]), drop=True)

    elif input_frequency == 'daily' and frequency == 'monthly':

        # Calculate EOFs using daily anomalies.
        base_period_hgt = hgt.where(
            (hgt[time_name] >= base_period[0]) &
            (hgt[time_name] <= base_period[1]), drop=True)

        base_period_clim = base_period_hgt.groupby(
            base_period_hgt[time_name].dt.dayofyear).mean(time_name)

        hgt_anom = hgt.groupby(hgt[time_name].dt.dayofyear) - base_period_clim

        base_period_hgt_anom = hgt_anom.where(
            (hgt_anom[time_name] >= base_period[0]) &
            (hgt_anom[time_name] <= base_period[1]), drop=True)

        # Calculate index with respect to monthly anomalies.
        hgt = downsample_data(hgt, frequency='monthly', time_name=time_name)

        base_period_hgt = hgt.where(
            (hgt[time_name] >= base_period[0]) &
            (hgt[time_name] <= base_period[1]), drop=True)

        base_period_clim = base_period_hgt.groupby(
            base_period_hgt[time_name].dt.month).mean(time_name)

        hgt_anom = hgt.groupby(hgt[time_name].dt.month) - base_period_clim

    elif input_frequency == 'monthly' and frequency == 'daily':

        raise ValueError('Cannot calculate daily index from monthly data')

    else:

        # Calculate EOFs and index from monthly anomalies.
        base_period_hgt = hgt.where(
            (hgt[time_name] >= base_period[0]) &
            (hgt[time_name] <= base_period[1]), drop=True)

        base_period_clim = base_period_hgt.groupby(
            base_period_hgt[time_name].dt.month).mean(time_name)

        hgt_anom = hgt.groupby(hgt[time_name].dt.month) - base_period_clim

        base_period_hgt_anom = hgt_anom.where(
            (hgt_anom[time_name] >= base_period[0]) &
            (hgt_anom[time_name] <= base_period[1]), drop=True)

    # Get square root of cos(latitude) weights.
    scos_weights = np.cos(np.deg2rad(hgt_anom[lat_name])).clip(0., 1.) ** 0.5

    # Loading pattern is computed using austral winter anomalies.
    winter_hgt_anom = base_period_hgt_anom.where(
        base_period_hgt_anom[time_name].dt.season == 'JJA', drop=True)

    # Calculate loading pattern from monthly anomalies in base period.
    if rotate:
        eofs_ds = reofs(winter_hgt_anom, sample_dim=time_name,
                        weight=scos_weights, n_modes=n_modes,
                        n_rotated_modes=n_rotated_modes)
    else:
        eofs_ds = eofs(winter_hgt_anom, sample_dim=time_name,
                       weight=scos_weights, n_modes=n_modes)

    eofs_ds = _fix_psa1_phase(eofs_ds, real_psa1_mode=psa1_mode,
                              lat_name=lat_name, lon_name=lon_name)

    # Project weighted anomalies onto PNA mode.
    pcs = _project_onto_subspace(
        hgt_anom, eofs_ds['EOFs'], weight=scos_weights,
        sample_dim=time_name)

    index = pcs.sel(mode=psa1_mode, drop=True)

    # Normalize index by base period monthly means and
    # standard deviations.
    if frequency == 'monthly':
        base_period_index = index.where(
            (index[time_name] >= base_period[0]) &
            (index[time_name] <= base_period[1]), drop=True)

    else:

        monthly_hgt = downsample_data(
            hgt.where(
                (hgt[time_name] >= base_period[0]) &
                (hgt[time_name] <= base_period[1]), drop=True),
            frequency='monthly', time_name=time_name)

        monthly_clim = monthly_hgt.groupby(
            monthly_hgt[time_name].dt.month).mean(time_name)

        monthly_hgt_anom = (
            monthly_hgt.groupby(monthly_hgt[time_name].dt.month) -
            monthly_clim)

        base_period_index = _project_onto_subspace(
            monthly_hgt_anom, eofs_ds['EOFs'], weight=scos_weights,
            sample_dim=time_name)

        base_period_index = base_period_index.sel(mode=psa1_mode, drop=True)

    index_mean = base_period_index.groupby(
        base_period_index[time_name].dt.month).mean(time_name)

    index_std = base_period_index.groupby(
        base_period_index[time_name].dt.month).std(time_name)

    index = xr.apply_ufunc(
        lambda x, m, s: (x - m) / s,
        index.groupby(index[time_name].dt.month),
        index_mean, index_std, dask='allowed')

    psa1_pattern = eofs_ds['EOFs'].sel(mode=psa1_mode, drop=True)

    data_vars = {'index': index, 'pattern': psa1_pattern}

    return xr.Dataset(data_vars)
