import numpy as np
import xarray as xr

from sklearn.cluster import KMeans

from .eofs import calc_eofs


DEFAULT_TIME_FIELD = 'time'
DEFAULT_LAT_FIELD = 'lat'
DEFAULT_LON_FIELD = 'lon'
DEFAULT_HGT_FIELD = 'hgt'

DEFAULT_SEASON = 'DJF'
DEFAULT_LAT_BOUNDS = None
DEFAULT_LON_BOUNDS = None

VALID_SEASONS = ['ALL', 'DJF', 'MAM', 'JJA', 'SON']
VALID_WEIGHTS = ['none', 'cos', 'scos']

DEFAULT_N_EOFS = 6
DEFAULT_LAT_WEIGHTS = 'scos'
EOF_DIM_NAME = 'mode'

CLUSTER_DIM_NAME = 'cluster'
DEFAULT_N_CLUSTERS = 4


def _check_valid_lat_weights(weights):
    if weights not in VALID_WEIGHTS:
        raise ValueError("unrecognized latitude weights '%r'" % weights)


def _get_lat_weights(lats, lat_weights='scos'):
    _check_valid_lat_weights(lat_weights)

    if lat_weights is None or lat_weights == 'none':
        return np.ones(lats.shape, lats.dtype)
    elif lat_weights == 'cos':
        return np.cos(np.deg2rad(lats))
    else:
        return np.sqrt(np.cos(np.deg2rad(lats)).clip(0., 1.))


def _get_valid_data(data, season=DEFAULT_SEASON,
                    lat_bounds=DEFAULT_LAT_BOUNDS,
                    lon_bounds=DEFAULT_LON_BOUNDS,
                    time_field=DEFAULT_TIME_FIELD,
                    lat_field=DEFAULT_LAT_FIELD,
                    lon_field=DEFAULT_LON_FIELD):
    if season == 'ALL':
        valid_data = data
    else:
        valid_data = data.where(data[time_field].dt.season == season,
                                drop=True)

    if lat_bounds is not None:
        if lat_bounds.ndim == 1:
            valid_data = valid_data.where(
                (valid_data[lat_field] >= lat_bounds[0]) &
                (valid_data[lat_field] <= lat_bounds[1]),
                drop=True)
        else:
            n_intervals = lat_bounds.shape[0]
            lat_vals = valid_data[lat_field]
            mask = np.zeros(lat_vals.shape, dtype=bool)
            for i in range(n_intervals):
                mask[np.logical_and(lat_vals >= lat_bounds[i, 0],
                                    lat_vals <= lat_bounds[i, 1])] = True
            valid_lat_vals = lat_vals[mask]
            valid_data = valid_data.where(
                valid_data[lat_field].isin(valid_lat_vals), drop=True)

    if lon_bounds is not None:
        if lon_bounds.ndim == 1:
            valid_data = valid_data.where(
                (valid_data[lon_field] >= lon_bounds[0]) &
                (valid_data[lon_field] <= lon_bounds[1]),
                drop=True)
        else:
            n_intervals = lon_bounds.shape[0]
            lon_vals = valid_data[lon_field]
            mask = np.zeros(lon_vals.shape, dtype=bool)
            for i in range(n_intervals):
                mask[np.logical_and(lon_vals >= lon_bounds[i, 0],
                                    lon_vals <= lon_bounds[i, 1])] = True
            valid_lon_vals = lon_vals[mask]
            valid_data = valid_data.where(
                valid_data[lon_field].isin(valid_lon_vals), drop=True)

    return valid_data


def calculate_seasonal_eofs(anom_data, season=DEFAULT_SEASON,
                            lat_bounds=DEFAULT_LAT_BOUNDS,
                            lon_bounds=DEFAULT_LON_BOUNDS,
                            n_eofs=DEFAULT_N_EOFS,
                            lat_weights=DEFAULT_LAT_WEIGHTS,
                            time_field=DEFAULT_TIME_FIELD,
                            lat_field=DEFAULT_LAT_FIELD,
                            lon_field=DEFAULT_LON_FIELD,
                            var_field=DEFAULT_HGT_FIELD,
                            random_state=None):
    valid_data = _get_valid_data(
        anom_data, season=season, lat_bounds=lat_bounds,
        lon_bounds=lon_bounds)
    print(valid_data)
    lat_data = valid_data[lat_field]

    weights = _get_lat_weights(lat_data, lat_weights=lat_weights)
    weights = xr.broadcast(valid_data.isel({time_field: 0}), weights)[1]

    eofs_results = calc_eofs(
        valid_data.values, weights=weights.values, n_eofs=n_eofs,
        random_state=random_state)

    eofs_data = eofs_results['eofs']
    eofs_dims = ([EOF_DIM_NAME] +
                 [d for d in valid_data.dims if d != time_field])
    eofs_coords = valid_data.coords.to_dataset().drop(
        time_field).reset_coords(drop=True)
    eofs_coords = eofs_coords.expand_dims(EOF_DIM_NAME, axis=0)
    eofs_coords.coords[EOF_DIM_NAME] = (EOF_DIM_NAME, np.arange(n_eofs))

    eofs_da = xr.DataArray(eofs_data, dims=eofs_dims,
                           coords=eofs_coords.coords)

    pcs_data = eofs_results['pcs']
    pcs_dims = [time_field, EOF_DIM_NAME]
    pcs_coords = {time_field: valid_data[time_field].values,
                  EOF_DIM_NAME: np.arange(n_eofs)}
    pcs_da = xr.DataArray(pcs_data, dims=pcs_dims, coords=pcs_coords)

    ev = eofs_results['explained_variance']
    evr = eofs_results['explained_variance_ratio']
    sv = eofs_results['singular_values']

    result = {
        'eofs': eofs_da,
        'pcs': pcs_da,
        'explained_variance': ev,
        'explained_variance_ratio': evr,
        'singular_values': sv
    }

    return result


def cluster_pcs(pcs_da, time_field=DEFAULT_TIME_FIELD, **kwargs):
    pcs_data = pcs_da.values

    kmeans = KMeans(**kwargs).fit(pcs_data)

    inertia = kmeans.inertia_
    n_iter = kmeans.n_iter_

    cluster_centers = kmeans.cluster_centers_
    n_clusters, n_eofs = cluster_centers.shape

    labels = kmeans.labels_
    n_samples = labels.shape[0]

    center_dims = [CLUSTER_DIM_NAME, EOF_DIM_NAME]
    center_coords = {CLUSTER_DIM_NAME: np.arange(n_clusters),
                     EOF_DIM_NAME: np.arange(n_eofs)}
    cluster_centers_da = xr.DataArray(
        cluster_centers, dims=center_dims,
        coords=center_coords)

    labels_data = np.zeros((n_samples, n_clusters))
    for i in range(n_samples):
        labels_data[i, labels[i]] = 1
    label_dims = [time_field, CLUSTER_DIM_NAME]
    label_coords = {time_field: pcs_da[time_field],
                    CLUSTER_DIM_NAME: np.arange(n_clusters)}
    labels_da = xr.DataArray(
        labels_data, dims=label_dims,
        coords=label_coords)

    return {'n_iter': n_iter, 'labels': labels_da,
            'inertia': inertia,
            'cluster_centers': cluster_centers_da}


def calculate_composites(anom_data, labels, season=DEFAULT_SEASON,
                         lat_bounds=DEFAULT_LAT_BOUNDS,
                         lon_bounds=DEFAULT_LON_BOUNDS,
                         time_field=DEFAULT_TIME_FIELD):
    valid_data = _get_valid_data(
        anom_data, season=season, lat_bounds=lat_bounds,
        lon_bounds=lon_bounds)

    n_samples, n_clusters = labels.shape
    labels_data = labels.values

    clusters = np.arange(n_clusters)
    composites_data = np.empty((n_clusters,) +
                               valid_data.isel({time_field: 0}).shape)

    time_axis = valid_data.get_axis_num(time_field)

    for i, k in enumerate(clusters):
        mask = labels_data[:, k] == 1
        cluster_data = valid_data.values[mask]
        cluster_mean = cluster_data.mean(axis=time_axis)
        composites_data[i] = cluster_mean

    composites_dims = ([CLUSTER_DIM_NAME] +
                       [d for d in valid_data.dims if d != time_field])

    composites_coords = valid_data.coords.to_dataset().drop(
        time_field).reset_coords(drop=True)
    composites_coords = composites_coords.expand_dims(CLUSTER_DIM_NAME, axis=0)
    composites_coords.coords[CLUSTER_DIM_NAME] = (
        CLUSTER_DIM_NAME, np.arange(n_clusters))

    composites_da = xr.DataArray(composites_data, dims=composites_dims,
                                 coords=composites_coords.coords)

    return composites_da
