"""
Provides routines for computing indices associated with Indo-Pacific SST.
"""

from __future__ import absolute_import

import numbers
import os

import dask.array
import geopandas as gp
import numpy as np
import regionmask as rm
import scipy.linalg
import xarray as xr


from ..utils import (check_base_period, check_data_array,
                     detect_frequency, get_coordinate_standard_name,
                     get_valid_variables, is_dask_array,
                     reofs, select_latlon_box, standardized_anomalies)


INTEGER_TYPES = (numbers.Integral, np.integer)

INDIAN_PACIFIC_OCEAN_REGION_SHP = os.path.join(
    os.path.dirname(__file__), 'indian_pacific_ocean.shp')


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


def dc_sst_loading_pattern(sst_anom, n_modes=None, n_rotated_modes=12,
                           weights=None, lat_name=None, lon_name=None,
                           time_name=None):
    """Calculate rotated EOFs for the SST1 and SST2 index.

    Parameters
    ----------
    sst_anom : xarray.DataArray
        Array containing SST anomalies.

    n_modes : integer
        Number of modes to compute in EOFs calculation.
        If None, the minimum number of modes needed to perform the
        rotation are computed.

    n_rotated_modes : integer
        Number of modes to include in VARIMAX rotation.
        If None, all computed modes are included.

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
        Dataset containing the computed rotated EOFs and PCs.
    """

    # Ensure that the data provided is a data array
    sst_anom = check_data_array(sst_anom)

    if lat_name is None:
        lat_name = get_coordinate_standard_name(sst_anom, 'lat')

    if lon_name is None:
        lon_name = get_coordinate_standard_name(sst_anom, 'lon')

    if time_name is None:
        time_name = get_coordinate_standard_name(sst_anom, 'time')

    if n_modes is None and n_rotated_modes is None:
        n_modes = 12
        n_rotated_modes = 12
    elif n_modes is None and n_rotated_modes is not None:
        n_modes = n_rotated_modes
    elif n_modes is not None and n_rotated_modes is None:
        n_rotated_modes = n_modes

    if not isinstance(n_modes, INTEGER_TYPES) or n_modes < 1:
        raise ValueError(
            'Invalid number of modes: got %r but must be at least 1')

    if not isinstance(n_rotated_modes, INTEGER_TYPES) or n_rotated_modes < 1:
        raise ValueError(
            'Invalid number of rotated modes: got %r but must be at least 1')

    if n_rotated_modes > n_modes:
        raise ValueError(
            'Number of rotated modes must not be greater than number of unrotated modes')

    eofs_ds = reofs(sst_anom, sample_dim=time_name, weight=weights,
                    n_modes=n_modes, n_rotated_modes=n_rotated_modes)

    return eofs_ds


def dc_sst(sst, frequency='monthly', n_modes=None, n_rotated_modes=12, base_period=None,
           lat_name=None, lon_name=None, time_name=None):
    """Calculate the SST1 and SST2 index of Indo-Pacific SST.

    The indices are defined as the PCs associated with the
    first and second rotated EOFs of standardized
    Indo-Pacific SST anomalies.

    See Drosdowsky, W. and Chambers, L. E., "Near-Global Sea Surface
    Temperature Anomalies as Predictors of Australian Seasonal
    Rainfall", Journal of Climate 14, 1677 - 1687 (2001).

    Parameters
    ----------
    sst : xarray.DataArray
        Array containing SST values.

    frequency : str
        If given, downsample data to requested frequency.

    n_modes : integer
        Number of EOFs to retain before rotation.

    n_rotated_modes : integer
        Number of EOFs to include in VARIMAX rotation.

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
    result : xarray.Dataset
        Dataset containing the SST index loading patterns and indices.
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

    base_period = check_base_period(sst, base_period=base_period,
                                    time_name=time_name)

    shape_data = gp.read_file(INDIAN_PACIFIC_OCEAN_REGION_SHP)

    region_mask = rm.Regions(shape_data['geometry'], numbers=[0])

    if not np.any(sst[lat_name] < 0):
        mask = region_mask.mask(sst, wrap_lon=True,
                                lat_name=lat_name, lon_name=lon_name)
    else:
        mask = region_mask.mask(sst, lat_name=lat_name, lon_name=lon_name)

    sst = sst.where(mask == 0)

    # EOFs are computed based on standardized monthly anomalies
    input_frequency = detect_frequency(sst, time_name=time_name)

    if input_frequency not in ('daily', 'monthly'):
        raise RuntimeError('Can only calculate index for daily or monthly data')

    if input_frequency == 'daily' and frequency == 'daily':

        base_period_monthly_sst = sst.where(
            (sst[time_name] >= base_period[0]) &
            (sst[time_name] <= base_period[1]), drop=True).resample(
                {time_name : '1MS'}).mean()

        base_period_monthly_sst_anom = standardized_anomalies(
            base_period_monthly_sst, base_period=base_period,
            standardize_by='month', time_name=time_name)

        sst_anom = standardized_anomalies(
            sst, base_period=base_period, standardize_by='dayofyear',
            time_name=time_name)

    elif input_frequency == 'daily' and frequency == 'monthly':

        sst = sst.resample({time_name : '1MS'}).mean()

        sst_anom = standardized_anomalies(
            sst, base_period=base_period, standardize_by='month',
            time_name=time_name)

        base_period_monthly_sst_anom = sst_anom.where(
            (sst_anom[time_name] >= base_period[0]) &
            (sst_anom[time_name] <= base_period[1]), drop=True)

    elif input_frequency == 'monthly' and frequency == 'daily':

        raise RuntimeError(
            'Attempting to calculate daily index from monthly data')

    else:

        sst_anom = standardized_anomalies(
            sst, base_period=base_period, standardize_by='month',
            time_name=time_name)

        base_period_monthly_sst_anom = sst_anom.where(
            (sst_anom[time_name] >= base_period[0]) &
            (sst_anom[time_name] <= base_period[1]), drop=True)

    # Get square root of cos(latitude) weights
    scos_weights = np.cos(np.deg2rad(sst_anom[lat_name])).clip(0., 1.) ** 0.5

    # Calculate VARIMAX rotated EOFs of standardized monthly anomalies.
    loadings_ds = dc_sst_loading_pattern(
        base_period_monthly_sst_anom, n_modes=n_modes,
        n_rotated_modes=n_rotated_modes, weights=scos_weights,
        lat_name=lat_name, lon_name=lon_name, time_name=time_name)

    # Project weighted anomalies onto first and second REOFs.
    rotated_pcs = _project_onto_subspace(
        sst_anom, loadings_ds['EOFs'], weight=scos_weights,
        sample_dim=time_name)

    sst1_index = rotated_pcs.sel(mode=0).drop_vars('mode').rename('sst1_index')
    sst2_index = rotated_pcs.sel(mode=1).drop_vars('mode').rename('sst2_index')

    data_vars = {'sst1_pattern' : loadings_ds['EOFs'].sel(mode=0, drop=True),
                 'sst2_pattern' : loadings_ds['EOFs'].sel(mode=1, drop=True),
                 'sst1_index' : sst1_index,
                 'sst2_index' : sst2_index}

    return xr.Dataset(data_vars)
