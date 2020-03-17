"""
Provides routines for calculating indices of the PNA.
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


def pointwise_pna3(hgt, frequency='monthly', base_period=None,
                   method='nearest', lat_name=None, lon_name=None,
                   time_name=None):
    """Calculate 3-point PNA index of Leathers, Yarnal and Palecki.

    The index is defined in terms of differences of
    standardized monthly mean geopotential height anomalies.

    See Leathers, D. J., Yarnal, B., and Palecki, M. A.,
    "The Pacific/North American Teleconnection Pattern and
    United States Climate. Part I: Regional Temperature and
    Precipitation Associations", Journal of Climate 4,
    517 - 528 (1991).

    Parameters
    ----------
    hgt : xarray.DataArray
        Array containing geopotential height values.

    frequency : 'monthly' | 'daily'
        Sampling rate at which to compute the index.

    base_period : list
        If given, a two element list containing the earliest
        and latest dates to include in the base period used
        for standardization. If None, the standardized
        anomalies are calculated with respect to the full
        time period.

    method : str
        Method used to select grid points.

    lat_name : str
        Name of the latitude coordinate.

    lon_name : str
        Name of the longitude coordinate.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    index : xarray.DataArray
        Array containing the values of the PNA index.
    """

    # Ensure that the data is provided as a data array.
    hgt = check_data_array(hgt)

    # Get coordinate names.
    if lat_name is None:
        lat_name = get_coordinate_standard_name(hgt, 'lat')

    if lon_name is None:
        lon_name = get_coordinate_standard_name(hgt, 'lon')

    if time_name is None:
        time_name = get_coordinate_standard_name(hgt, 'time')

    base_period = check_base_period(hgt, base_period=base_period,
                                    time_name=time_name)

    if frequency is None:
        frequency = 'monthly'

    if frequency not in ('daily', 'monthly'):
        raise ValueError("Unsupported frequency '%r'" % frequency)

    input_frequency = detect_frequency(hgt, time_name=time_name)

    if input_frequency not in ('daily', 'monthly'):
        raise RuntimeError('Can only calculate index for daily or monthly data')

    if np.any(hgt[lon_name] < 0):
        center_one = {lat_name : 47.9, lon_name : -170.0}
        center_two = {lat_name : 49.0, lon_name : -111.0}
        center_three = {lat_name : 29.7, lon_name : -86.3}
    else:
        center_one = {lat_name : 47.9, lon_name : 190.0}
        center_two = {lat_name : 49.0, lon_name : 249.0}
        center_three = {lat_name : 29.7, lon_name : 273.7}

    z1 = hgt.sel(center_one, method=method, drop=True)
    z2 = hgt.sel(center_two, method=method, drop=True)
    z3 = hgt.sel(center_three, method=method, drop=True)

    if input_frequency == 'daily' and frequency == 'daily':

        standardize_by = 'dayofyear'

    elif input_frequency == 'daily' and frequency == 'monthly':

        z1 = downsample_data(z1, frequency='monthly', time_name=time_name)
        z2 = downsample_data(z2, frequency='monthly', time_name=time_name)
        z3 = downsample_data(z3, frequency='monthly', time_name=time_name)

        standardize_by = 'month'

    elif input_frequency == 'monthly' and frequency == 'daily':

        raise ValueError('Cannot calculate daily index from monthly data')

    else:

        standardize_by = 'month'

    z1 = standardized_anomalies(z1, base_period=base_period,
                                standardize_by=standardize_by,
                                time_name=time_name)
    z2 = standardized_anomalies(z2, base_period=base_period,
                                standardize_by=standardize_by,
                                time_name=time_name)
    z3 = standardized_anomalies(z3, base_period=base_period,
                                standardize_by=standardize_by,
                                time_name=time_name)

    index = ((z2 - z1 - z3) / 3.0).rename('pna')

    return index


def pointwise_pna4(hgt, frequency='monthly', base_period=None,
                   method='nearest', lat_name=None, lon_name=None,
                   time_name=None):
    """Calculate 4-point PNA index of Wallace and Gutzler.

    The index is defined in terms of differences of
    standardized monthly mean geopotential height anomalies.

    See Wallace, J. M., and Gutzler, D. S., "Teleconnections in the
    Geopotential Height Field during the Northern Hemisphere Winter",
    Monthly Weather Review 109, 784 - 812, (1981).

    Parameters
    ----------
    hgt : xarray.DataArray
        Array containing geopotential height values.

    frequency : 'monthly' | 'daily'
        Sampling rate at which to compute the index.

    base_period : list
        If given, a two element list containing the earliest
        and latest dates to include in the base period used
        for standardization. If None, the standardized
        anomalies are calculated with respect to the full
        time period.

    method : str
        Method used to select grid points.

    lat_name : str
        Name of the latitude coordinate.

    lon_name : str
        Name of the longitude coordinate.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    index : xarray.DataArray
        Array containing the values of the PNA index.
    """

    # Ensure that the data is provided as a data array.
    hgt = check_data_array(hgt)

    # Get coordinate names.
    if lat_name is None:
        lat_name = get_coordinate_standard_name(hgt, 'lat')

    if lon_name is None:
        lon_name = get_coordinate_standard_name(hgt, 'lon')

    if time_name is None:
        time_name = get_coordinate_standard_name(hgt, 'time')

    base_period = check_base_period(hgt, base_period=base_period,
                                    time_name=time_name)

    if frequency is None:
        frequency = 'monthly'

    if frequency not in ('daily', 'monthly'):
        raise ValueError("Unsupported frequency '%r'" % frequency)

    input_frequency = detect_frequency(hgt, time_name=time_name)

    if input_frequency not in ('daily', 'monthly'):
        raise RuntimeError('Can only calculate index for daily or monthly data')

    if np.any(hgt[lon_name] < 0):
        center_one = {lat_name : 20.0, lon_name : -160.0}
        center_two = {lat_name : 45.0, lon_name : -165.0}
        center_three = {lat_name : 55.0, lon_name : -115.0}
        center_four = {lat_name : 30.0, lon_name : -85.0}
    else:
        center_one = {lat_name : 20.0, lon_name : 200.0}
        center_two = {lat_name : 45.0, lon_name : 195.0}
        center_three = {lat_name : 55.0, lon_name : 245.0}
        center_four = {lat_name : 30.0, lon_name : 275.0}

    z1 = hgt.sel(center_one, method=method, drop=True)
    z2 = hgt.sel(center_two, method=method, drop=True)
    z3 = hgt.sel(center_three, method=method, drop=True)
    z4 = hgt.sel(center_four, method=method, drop=True)

    if input_frequency == 'daily' and frequency == 'daily':

        standardize_by = 'dayofyear'

    elif input_frequency == 'daily' and frequency == 'monthly':

        z1 = downsample_data(z1, frequency='monthly', time_name=time_name)
        z2 = downsample_data(z2, frequency='monthly', time_name=time_name)
        z3 = downsample_data(z3, frequency='monthly', time_name=time_name)
        z4 = downsample_data(z4, frequency='monthly', time_name=time_name)

        standardize_by = 'month'

    elif input_frequency == 'monthly' and frequency == 'daily':

        raise ValueError('Cannot calculate daily index from monthly data')

    else:

        standardize_by = 'month'

    z1 = standardized_anomalies(z1, base_period=base_period,
                                standardize_by=standardize_by,
                                time_name=time_name)
    z2 = standardized_anomalies(z2, base_period=base_period,
                                standardize_by=standardize_by,
                                time_name=time_name)
    z3 = standardized_anomalies(z3, base_period=base_period,
                                standardize_by=standardize_by,
                                time_name=time_name)
    z4 = standardized_anomalies(z4, base_period=base_period,
                                standardize_by=standardize_by,
                                time_name=time_name)

    index = (0.25 * (z1 - z2 + z3 - z4)).rename('pna')

    return index


def modified_pointwise_pna4(hgt, frequency='monthly', base_period=None,
                            method='nearest', lat_name=None, lon_name=None,
                            time_name=None):
    """Calculate modified 4-point PNA index.

    The index is defined in terms of differences of
    standardized monthly mean geopotential height anomalies.

    See, e.g., https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/month_pna_index2.shtml

    Parameters
    ----------
    hgt : xarray.DataArray
        Array containing geopotential height values.

    frequency : 'monthly' | 'daily'
        Sampling rate at which to compute the index.

    base_period : list
        If given, a two element list containing the earliest
        and latest dates to include in the base period used
        for standardization. If None, the standardized
        anomalies are calculated with respect to the full
        time period.

    method : str
        Method used to select grid points.

    lat_name : str
        Name of the latitude coordinate.

    lon_name : str
        Name of the longitude coordinate.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    index : xarray.DataArray
        Array containing the values of the PNA index.
    """

    # Ensure that the data is provided as a data array.
    hgt = check_data_array(hgt)

    # Get coordinate names.
    if lat_name is None:
        lat_name = get_coordinate_standard_name(hgt, 'lat')

    if lon_name is None:
        lon_name = get_coordinate_standard_name(hgt, 'lon')

    if time_name is None:
        time_name = get_coordinate_standard_name(hgt, 'time')

    base_period = check_base_period(hgt, base_period=base_period,
                                    time_name=time_name)

    if frequency is None:
        frequency = 'monthly'

    if frequency not in ('daily', 'monthly'):
        raise ValueError("Unsupported frequency '%r'" % frequency)

    input_frequency = detect_frequency(hgt, time_name=time_name)

    if input_frequency not in ('daily', 'monthly'):
        raise RuntimeError('Can only calculate index for daily or monthly data')

    z1 = select_latlon_box(
        hgt, lat_bounds=[15.0, 25.0], lon_bounds=[-180.0, -140.0],
        lat_name=lat_name, lon_name=lon_name).mean(dim=[lat_name, lon_name])
    z2 = select_latlon_box(
        hgt, lat_bounds=[40.0, 50.0], lon_bounds=[-180.0, -140.0],
        lat_name=lat_name, lon_name=lon_name).mean(dim=[lat_name, lon_name])
    z3 = select_latlon_box(
        hgt, lat_bounds=[45.0, 60.0], lon_bounds=[-125.0, -105.0],
        lat_name=lat_name, lon_name=lon_name).mean(dim=[lat_name, lon_name])
    z4 = select_latlon_box(
        hgt, lat_bounds=[25.0, 35.0], lon_bounds=[-90.0, -70.0],
        lat_name=lat_name, lon_name=lon_name).mean(dim=[lat_name, lon_name])

    if input_frequency == 'daily' and frequency == 'daily':

        group_by = 'dayofyear'

    elif input_frequency == 'daily' and frequency == 'monthly':

        z1 = downsample_data(z1, frequency='monthly', time_name=time_name)
        z2 = downsample_data(z2, frequency='monthly', time_name=time_name)
        z3 = downsample_data(z3, frequency='monthly', time_name=time_name)
        z4 = downsample_data(z4, frequency='monthly', time_name=time_name)

        group_by = 'month'

    elif input_frequency == 'monthly' and frequency == 'daily':

        raise ValueError('Cannot calculate daily index from monthly data')

    else:

        group_by = 'month'

    base_period_z1 = z1.where(
        (z1[time_name] >= base_period[0]) &
        (z1[time_name] <= base_period[1]), drop=True)
    base_period_z2 = z2.where(
        (z2[time_name] >= base_period[0]) &
        (z2[time_name] <= base_period[1]), drop=True)
    base_period_z3 = z3.where(
        (z3[time_name] >= base_period[0]) &
        (z3[time_name] <= base_period[1]), drop=True)
    base_period_z4 = z4.where(
        (z4[time_name] >= base_period[0]) &
        (z4[time_name] <= base_period[1]), drop=True)

    if group_by == 'dayofyear':

        z1_clim = base_period_z1.groupby(
            base_period_z1[time_name].dt.dayofyear).mean(time_name)
        z2_clim = base_period_z2.groupby(
            base_period_z2[time_name].dt.dayofyear).mean(time_name)
        z3_clim = base_period_z3.groupby(
            base_period_z3[time_name].dt.dayofyear).mean(time_name)
        z4_clim = base_period_z4.groupby(
            base_period_z4[time_name].dt.dayofyear).mean(time_name)

        z1 = z1.groupby(z1[time_name].dt.dayofyear) - z1_clim
        z2 = z2.groupby(z2[time_name].dt.dayofyear) - z2_clim
        z3 = z3.groupby(z3[time_name].dt.dayofyear) - z3_clim
        z4 = z4.groupby(z4[time_name].dt.dayofyear) - z4_clim

    elif group_by == 'month':

        z1_clim = base_period_z1.groupby(
            base_period_z1[time_name].dt.month).mean(time_name)
        z2_clim = base_period_z2.groupby(
            base_period_z2[time_name].dt.month).mean(time_name)
        z3_clim = base_period_z3.groupby(
            base_period_z3[time_name].dt.month).mean(time_name)
        z4_clim = base_period_z4.groupby(
            base_period_z4[time_name].dt.month).mean(time_name)

        z1 = z1.groupby(z1[time_name].dt.month) - z1_clim
        z2 = z2.groupby(z2[time_name].dt.month) - z2_clim
        z3 = z3.groupby(z3[time_name].dt.month) - z3_clim
        z4 = z4.groupby(z4[time_name].dt.month) - z4_clim

    else:

        z1_clim = base_period_z1.mean(time_name)
        z2_clim = base_period_z2.mean(time_name)
        z3_clim = base_period_z3.mean(time_name)
        z4_clim = base_period_z4.mean(time_name)

        z1 = z1 - z1_clim
        z2 = z2 - z2_clim
        z3 = z3 - z3_clim
        z4 = z4 - z4_clim

    index = ((z1 - z2 + z3 - z4)).rename('pna')

    normalization = index.where(
        (index[time_name] >= base_period[0]) &
        (index[time_name] <= base_period[1]), drop=True).std(time_name)

    return index / normalization


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


def _fix_pna_phase(eofs_ds, pna_mode, lat_name=None, lon_name=None):
    """Fix PNA phase such that negative anomalies occur in Pacific sector."""

    if lat_name is None:
        lat_name = get_coordinate_standard_name(eofs_ds, 'lat')

    if lon_name is None:
        lon_name = get_coordinate_standard_name(eofs_ds, 'lon')

    pna_pattern = eofs_ds['EOFs'].sel(mode=pna_mode, drop=True).squeeze()

    pacific_sector = select_latlon_box(
        pna_pattern, lat_bounds=[35.0, 90.0], lon_bounds=[160.0, 200.0],
        lat_name=lat_name, lon_name=lon_name)

    pacific_max = pacific_sector.max().item()
    flipped_max = (-pacific_sector).max().item()

    if pacific_max > flipped_max:

        eofs_ds['EOFs'] = -eofs_ds['EOFs']
        eofs_ds['PCs'] = -eofs_ds['PCs']

    return eofs_ds


def pc_pna(hgt, frequency='monthly', base_period=None, rotate=True,
           pna_mode=1, n_modes=None, n_rotated_modes=None,
           low_latitude_boundary=20, lat_name=None, lon_name=None,
           time_name=None):
    """Calculate principal component based PNA index.

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

    pna_mode : integer
        Mode to take as corresponding to the PNA mode.

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
        Dataset containing the PNA index and loading pattern.
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
    hgt = hgt.where(hgt[lat_name] >= low_latitude_boundary, drop=True)

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

    if pna_mode is None:
        pna_mode = 1

    if not isinstance(pna_mode, INTEGER_TYPES) or pna_mode < 0:
        raise ValueError('PNA mode must be a positive integer')

    if n_modes is None:
        n_modes = 10

    if not isinstance(n_modes, INTEGER_TYPES) or n_modes <= pna_mode:
        raise ValueError('Number of modes must be an integer greater than %d' %
                         pna_mode)

    if n_rotated_modes is None:
        n_rotated_modes = n_modes

    if not isinstance(n_rotated_modes, INTEGER_TYPES) or n_rotated_modes <= pna_mode:
        raise ValueError('Number of modes must be an integer greater than %d' %
                         pna_mode)

    # Note that EOFs are computed using monthly mean data.
    if input_frequency == 'daily' and frequency == 'daily':

        hgt_anom = standardized_anomalies(
            hgt, base_period=base_period, standardize_by='dayofyear',
            time_name=time_name)

        base_period_hgt = hgt.where(
            (hgt[time_name] >= base_period[0]) &
            (hgt[time_name] <= base_period[1]), drop=True)

        base_period_hgt = downsample_data(
            base_period_hgt, frequency='monthly', time_name=time_name)

        base_period_hgt_anom = standardized_anomalies(
            base_period_hgt, base_period=base_period,
            standardize_by='month', time_name=time_name)

    elif input_frequency == 'daily' and frequency == 'monthly':

        hgt = downsample_data(hgt, frequency='monthly', time_name=time_name)

        hgt_anom = standardized_anomalies(
            hgt, base_period=base_period, standardize_by='month',
            time_name=time_name)

        base_period_hgt_anom = hgt_anom.where(
            (hgt_anom[time_name] >= base_period[0]) &
            (hgt_anom[time_name] <= base_period[1]), drop=True)

    elif input_frequency == 'monthly' and frequency == 'daily':

        raise ValueError('Cannot calculate daily index from monthly data')

    else:

        hgt_anom = standardized_anomalies(
            hgt, base_period=base_period, standardize_by='month',
            time_name=time_name)

        base_period_hgt_anom = hgt_anom.where(
            (hgt_anom[time_name] >= base_period[0]) &
            (hgt_anom[time_name] <= base_period[1]), drop=True)

    # Get square root of cos(latitude) weights.
    scos_weights = np.cos(np.deg2rad(hgt_anom[lat_name])).clip(0., 1.) ** 0.5

    # Loading pattern is computed using boreal winter anomalies.
    winter_hgt_anom = base_period_hgt_anom.where(
        base_period_hgt_anom[time_name].dt.season == 'DJF', drop=True)

    # Calculate loading pattern from monthly anomalies in base period.
    if rotate:
        eofs_ds = reofs(winter_hgt_anom, sample_dim=time_name,
                        weight=scos_weights, n_modes=n_modes,
                        n_rotated_modes=n_rotated_modes)
    else:
        eofs_ds = eofs(winter_hgt_anom, sample_dim=time_name,
                       weight=scos_weights, n_modes=n_modes)

    eofs_ds = _fix_pna_phase(eofs_ds, pna_mode=pna_mode,
                             lat_name=lat_name, lon_name=lon_name)

    # Project weighted anomalies onto PNA mode.
    pcs = _project_onto_subspace(
        hgt_anom, eofs_ds['EOFs'], weight=scos_weights,
        sample_dim=time_name)

    index = pcs.sel(mode=pna_mode, drop=True)

    # Normalize index by base period monthly means and
    # standard deviations.
    if frequency == 'monthly':
        base_period_index = index.where(
            (index[time_name] >= base_period[0]) &
            (index[time_name] <= base_period[1]), drop=True)

    else:

        base_period_index = _project_onto_subspace(
            base_period_hgt_anom, eofs_ds['EOFs'], weight=scos_weights,
            sample_dim=time_name)

        base_period_index = base_period_index.sel(mode=pna_mode, drop=True)

    index_mean = base_period_index.groupby(
        base_period_index[time_name].dt.month).mean(time_name)

    index_std = base_period_index.groupby(
        base_period_index[time_name].dt.month).std(time_name)

    index = xr.apply_ufunc(
        lambda x, m, s: (x - m) / s,
        index.groupby(index[time_name].dt.month),
        index_mean, index_std, dask='allowed')

    pna_pattern = eofs_ds['EOFs'].sel(mode=pna_mode, drop=True)

    data_vars = {'index': index, 'pattern': pna_pattern}

    return xr.Dataset(data_vars)
