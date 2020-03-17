"""
Provides routines for calculating NH teleconnections.
"""


from __future__ import absolute_import

import numbers

import dask.array
import numpy as np
import scipy.linalg
import xarray as xr

from sklearn.cluster import KMeans

from ..utils import (check_base_period, check_data_array,
                     detect_frequency, downsample_data, eofs,
                     get_coordinate_standard_name,
                     get_valid_variables, is_dask_array,
                     select_latlon_box)


INTEGER_TYPES = (numbers.Integral, np.integer)


def _project_onto_subspace(data, basis, weight=None, sample_dim=None,
                           mode_dim='mode', simultaneous=False):
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

    if basis.get_axis_num(mode_dim) != 0:
        basis = basis.transpose(*([mode_dim] + feature_dims))

    n_modes = basis.sizes[mode_dim]

    flat_eofs = basis.values.reshape((n_modes, n_features))
    valid_eofs = flat_eofs[:, valid_features].swapaxes(0, 1)

    if simultaneous:
        if is_dask_array(flat_data):

            projection = dask.array.linalg.lstsq(valid_eofs, valid_data)

        else:

            projection = scipy.linalg.lstsq(valid_eofs, valid_data)[0]

    else:

        projection = valid_eofs.T.dot(valid_data)

    projection = xr.DataArray(
        projection.swapaxes(0, 1),
        coords={sample_dim : data[sample_dim],
                mode_dim : basis[mode_dim]},
        dims=[sample_dim, mode_dim])

    return projection


def calculate_kmeans_pcs_anomalies(hgt, window_length=None, base_period=None,
                                   time_name=None):
    """Calculate anomalies from input data after applying smoothing.

    Parameters
    ----------
    hgt : xarray.DataArray
        Array containing geopotential height values.

    window_length : integer
        Length of moving average window.

    base_period : list
        If given, a two element list containing the earliest and latest
        dates to include in the base period for calculating the climatology.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    hgt_anom : xarray.DataArray
        Array containing geopotential height anomalies.

    base_period : list
        Base period used for climatology.
    """

    hgt = check_data_array(hgt)

    if time_name is None:
        time_name = get_coordinate_standard_name(hgt, 'time')

    input_frequency = detect_frequency(hgt, time_name=time_name)

    if input_frequency == 'daily':

        if window_length is None:
            window_length = 10

    elif input_frequency == 'monthly':

        if window_length is None:
            window_length = 1

    else:

        if window_length is None:
            window_length = 1

    if not isinstance(window_length, INTEGER_TYPES) or window_length < 1:
        raise ValueError('Window length must be a positive integer')

    if window_length > 1:
        hgt = hgt.rolling(
            {time_name : window_length}).mean().dropna(time_name, how='all')

    base_period = check_base_period(hgt, base_period=base_period,
                                    time_name=time_name)

    base_period_hgt = hgt.where(
        (hgt[time_name] >= base_period[0]) &
        (hgt[time_name] <= base_period[1]), drop=True)

    if input_frequency == 'daily':

        clim = base_period_hgt.groupby(
            base_period_hgt[time_name].dt.dayofyear).mean(time_name)

        hgt_anom = hgt.groupby(hgt[time_name].dt.dayofyear) - clim

        if 'dayofyear' not in hgt.coords:
            hgt_anom = hgt_anom.drop('dayofyear')

    elif input_frequency == 'monthly':

        clim = base_period_hgt.groupby(
            base_period_hgt[time_name].dt.month).mean(time_name)

        hgt_anom = hgt.groupby(hgt[time_name].dt.month) - clim

        if 'month' not in hgt.coords:
            hgt_anom = hgt_anom.drop('month')

    else:

        clim = base_period_hgt.mean(time_name)

        hgt_anom = hgt - clim

    return hgt_anom, base_period


def kmeans_pc_clustering(hgt_anom, season=None, n_modes=20, n_clusters=4,
                         base_period=None, lat_name=None, lon_name=None,
                         time_name=None, **kwargs):
    """Perform k-means clustering of leading PCs.

    Parameters
    ----------
    hgt_anom : xarray.DataArray
        Array containing geopotential height anomalies.

    season : None | 'DJF' | 'MAM' | 'JJA' | 'SON' | 'ALL'
        Season to restrict EOF analysis to. If None, all seasons are used.

    n_modes : integer
        Number of EOF modes to compute.

    n_clusters : integer
        Number of clusters.

    base_period : list
        If given, a two element list containing the earliest and latest
        dates to include in the EOF analysis.

    lat_name : str
        Name of the latitude coordinate.

    lon_name : str
        Name of the longitude coordinate.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    result : xarray.Dataset
        Dataset containing the result of the combined EOF and cluster
        analysis.
    """

    hgt_anom = check_data_array(hgt_anom)

    if lat_name is None:
        lat_name = get_coordinate_standard_name(hgt_anom, 'lat')

    if lon_name is None:
        lon_name = get_coordinate_standard_name(hgt_anom, 'lon')

    if time_name is None:
        time_name = get_coordinate_standard_name(hgt_anom, 'time')

    if n_modes is None:
        n_modes = 20

    if not isinstance(n_modes, INTEGER_TYPES) or n_modes < 1:
        raise ValueError('Number of modes must be a positive integer')

    if n_clusters is None:
        n_clusters = 4

    if not isinstance(n_clusters, INTEGER_TYPES) or n_clusters < 1:
        raise ValueError('Number of clusters must be a positive integer')

    if season is None:
        season = 'ALL'

    valid_seasons = ('DJF', 'MAM', 'JJA', 'SON', 'ALL')
    if season not in valid_seasons:
        raise ValueError("Unrecognized season '%r'" % season)

    if season != 'ALL':
        seasonal_anom = hgt_anom.where(hgt_anom[time_name].dt.season == season,
                                       drop=True)
    else:
        seasonal_anom = hgt_anom

    base_period = check_base_period(seasonal_anom, base_period=base_period,
                                    time_name=time_name)

    base_period_anom = seasonal_anom.where(
        (seasonal_anom[time_name] >= base_period[0]) &
        (seasonal_anom[time_name] <= base_period[1]), drop=True)

    # EOF analysis is restricted to North Atlantic region.
    base_period_anom = select_latlon_box(
        base_period_anom, lat_bounds=[20.0, 80.0],
        lon_bounds=[-90.0, 30.0], lat_name=lat_name, lon_name=lon_name)

    scos_weights = np.cos(
        np.deg2rad(seasonal_anom[lat_name])).clip(0., 1.) ** 0.5

    eofs_ds = eofs(base_period_anom, weight=scos_weights, n_modes=n_modes,
                   sample_dim=time_name)

    n_samples = eofs_ds[time_name].size
    if eofs_ds['PCs'].values.shape == (n_samples, n_modes):
        pcs_data = eofs_ds['PCs'].values
    else:
        pcs_data = eofs_ds['PCs'].values.T

    kmeans = KMeans(n_clusters=n_clusters, **kwargs).fit(pcs_data)

    inertia = kmeans.inertia_
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    eofs_ds['centroids'] = xr.DataArray(
        cluster_centers, coords={'cluster': np.arange(n_clusters),
                                 'mode': np.arange(n_modes)},
        dims=['cluster', 'mode'])
    eofs_ds['inertia'] = xr.DataArray(
        inertia, coords={time_name : eofs_ds[time_name]},
        dims=[time_name])
    eofs_ds['labels'] = xr.DataArray(
        labels, coords={time_name : eofs_ds[time_name]}, dims=[time_name])

    return eofs_ds


def kmeans_pcs_composites(hgt_anom, cluster_assignments, n_clusters=None,
                          lat_name=None, time_name=None):
    """Calculate composite fields based on cluster assignments.

    Parameters
    ----------
    hgt_anom : xarray.DataArray)
        Array containing geopotential height anomalies.

    cluster_assignments :  xarray.DataArray
        Array containing the initial cluster assignments.

    n_clusters : integer
        Number of clusters.

    lat_name : str
        Name of the latitude coordinate.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    composites : xarray.DataArray
        Array containing the composited geopotential height anomalies.
    """

    hgt_anom = check_data_array(hgt_anom)

    if lat_name is None:
        lat_name = get_coordinate_standard_name(hgt_anom, 'lat')

    if time_name is None:
        time_name = get_coordinate_standard_name(hgt_anom, 'time')

    # Select time-steps present in initial cluster analysis
    cluster_hgt_anom = hgt_anom.sel(
        {time_name : cluster_assignments[time_name]}, drop=True)

    scos_weights = np.cos(np.deg2rad(
        cluster_hgt_anom[lat_name])).clip(0., 1.) ** 0.5

    # Perform composite over each cluster state
    if n_clusters is None:
        n_clusters = np.size(np.unique(cluster_assignments.values))

    if not isinstance(n_clusters, INTEGER_TYPES) or n_clusters < 1:
        raise ValueError('Number of clusters must be a positive integer')

    composite_shape = cluster_hgt_anom.isel({time_name : 0}).shape
    composite_coords = dict(c for c in cluster_hgt_anom.coords.items()
                            if c != time_name)
    composite_coords['cluster'] = np.arange(n_clusters)

    composite_dims = cluster_hgt_anom.isel({time_name : 0}).dims
    composite_dims = ('cluster',) + composite_dims

    composites = xr.DataArray(np.empty((n_clusters,) + composite_shape),
                              coords=composite_coords,
                              dims=composite_dims)

    for k in range(n_clusters):
        cluster_members = cluster_hgt_anom.where(
            cluster_assignments == k, drop=True)

        cluster_mean = cluster_members.mean(time_name)

        weighted_mean = scos_weights * cluster_mean

        if weighted_mean.shape != cluster_mean.shape:
            weighted_mean = weighted_mean.transpose(*(cluster_mean.dims))

        weighted_mean = weighted_mean / np.sqrt(np.sum(weighted_mean ** 2))

        composites.loc[dict(cluster=k)] = weighted_mean

    return composites


def kmeans_pcs(hgt, frequency='monthly', base_period=None, n_modes=20,
               n_clusters=4, window_length=None, season='DJF',
               lat_name=None, lon_name=None, time_name=None, **kwargs):
    """Calculate k-means based teleconnection indices.

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

    n_modes : integer
        Number of EOF modes to calculate.

    n_clusters : integer
        Number of clusters to use in k-means clustering.

    window_length : integer
        Length of window used for rolling mean. If None and daily data
        is given, a length of 10 days is used. Otherwise, no rolling
        average is performed.

    season : None | 'DJF' | 'MAM' | 'JJA' | 'SON' | 'ALL'
        Season to perform EOF analysis on.

    lat_name : str
        Name of the latitude coordinate.

    lon_name : str
        Name of the longitude coordinate.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    ds : xarray.Dataset
        Dataset containing the indices and loading patterns.
    """

    # Ensure that given data is a data array.
    hgt = check_data_array(hgt)

    if lat_name is None:
        lat_name = get_coordinate_standard_name(hgt, 'lat')

    if lon_name is None:
        lon_name = get_coordinate_standard_name(hgt, 'lon')

    if time_name is None:
        time_name = get_coordinate_standard_name(hgt, 'time')

    if frequency is None:
        frequency = 'monthly'

    if frequency not in ('daily', 'monthly'):
        raise ValueError("Unsupported frequency '%r'" % frequency)

    input_frequency = detect_frequency(hgt, time_name=time_name)

    if input_frequency not in ('daily', 'monthly'):
        raise RuntimeError('Can only calculate index for daily or monthly data')

    if input_frequency == 'monthly' and frequency == 'daily':
        raise ValueError('Cannot calculate daily index from monthly data')

    hgt_anom, base_period = calculate_kmeans_pcs_anomalies(
        hgt, window_length=window_length, base_period=base_period,
        time_name=time_name)

    scos_weights = np.cos(np.deg2rad(hgt_anom[lat_name])).clip(0., 1.) ** 0.5

    cluster_assignments = kmeans_pc_clustering(
        hgt_anom, season=season, n_modes=n_modes, n_clusters=n_clusters,
        base_period=base_period, lat_name=lat_name, lon_name=lon_name,
        time_name=time_name, **kwargs)

    composites = kmeans_pcs_composites(
        hgt_anom, cluster_assignments['labels'],
        lat_name=lat_name, time_name=time_name)

    if input_frequency == 'daily' and frequency == 'monthly':

        hgt = downsample_data(hgt, frequency='monthly',
                              time_name=time_name)

        base_period_hgt = hgt.where(
            (hgt[time_name] >= base_period[0]) &
            (hgt[time_name] <= base_period[1]), drop=True)

        clim = base_period_hgt.groupby(
            base_period_hgt[time_name].dt.month).mean(time_name)

        hgt_anom = hgt.groupby(hgt[time_name].dt.month) - clim

    indices = _project_onto_subspace(
        hgt_anom, composites, weight=scos_weights,
        sample_dim=time_name, mode_dim='cluster')

    # Standardize indices by monthly means and standard deviations
    # within base period.
    if frequency == 'monthly':

        base_period_indices = indices.where(
            (indices[time_name] >= base_period[0]) &
            (indices[time_name] <= base_period[1]), drop=True)

    else:

        hgt = downsample_data(hgt, frequency='monthly',
                              time_name=time_name)

        base_period_hgt = hgt.where(
            (hgt[time_name] >= base_period[0]) &
            (hgt[time_name] <= base_period[1]), drop=True)

        clim = base_period_hgt.groupby(
            base_period_hgt[time_name].dt.month).mean(time_name)

        hgt_anom = hgt.groupby(hgt[time_name].dt.month) - clim

        base_period_indices = _project_onto_subspace(
            hgt_anom, composites, weight=scos_weights,
            sample_dim=time_name, mode_dim='cluster')

        base_period_indices = indices.where(
            (indices[time_name] >= base_period[0]) &
            (indices[time_name] <= base_period[1]), drop=True)

    indices_mean = base_period_indices.groupby(
        base_period_indices[time_name].dt.month).mean(time_name)

    indices_std = base_period_indices.groupby(
        base_period_indices[time_name].dt.month).std(time_name)

    indices = xr.apply_ufunc(
        lambda x, m, s: (x - m) / s,
        indices.groupby(indices[time_name].dt.month),
        indices_mean, indices_std, dask='allowed')

    data_vars = {'composites' : composites, 'indices' : indices}

    return xr.Dataset(data_vars)
