import numpy as np
import xarray as xr

from .eofs import calc_eofs


DEFAULT_TIME_FIELD = 'time'
DEFAULT_LAT_FIELD = 'lat'
DEFAULT_HGT_FIELD = 'hgt'

MIN_LATITUDE = 20.0
MAX_LATITUDE = 90.0

VALID_WEIGHTS = ['none', 'cos', 'scos']

DEFAULT_AO_MODE = 0
N_EOFS = 1
EOF_DIM_NAME = 'mode'


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


def _project_data(X, eofs, lat_weights='scos',
                  time_field=DEFAULT_TIME_FIELD,
                  lat_field=DEFAULT_LAT_FIELD):
    n_samples = X.shape[0]
    n_eofs = eofs.shape[0]
    n_features = np.product(X.shape[1:])

    lat_data = X[lat_field]

    weights = _get_lat_weights(lat_data, lat_weights=lat_weights)
    weights = xr.broadcast(X.isel({time_field: 0}), weights)[1]

    weighted_data = weights.values * X.values

    flat_data = np.reshape(weighted_data, (n_samples, n_features))
    flat_eofs = np.reshape(eofs.values, (n_eofs, n_features))

    a = np.dot(flat_eofs, flat_eofs.T)
    b = np.dot(flat_eofs, flat_data.T)

    sol = np.linalg.lstsq(a, b, rcond=None)[0]

    return sol.T


def calculate_daily_region_anomalies(hgt_data, climatology=None,
                                     time_field=DEFAULT_TIME_FIELD,
                                     lat_field=DEFAULT_LAT_FIELD):
    valid_data = hgt_data.where(
        (hgt_data[lat_field] >= MIN_LATITUDE) &
        (hgt_data[lat_field] <= MAX_LATITUDE), drop=True)

    if climatology is None:
        climatology = valid_data.groupby(
            valid_data[time_field].dt.dayofyear).mean(time_field)

    anom = valid_data.groupby(
        valid_data[time_field].dt.dayofyear) - climatology

    return anom, climatology


def calculate_monthly_region_anomalies(hgt_data, climatology=None,
                                       time_field=DEFAULT_TIME_FIELD,
                                       lat_field=DEFAULT_LAT_FIELD):
    valid_data = hgt_data.where(
        (hgt_data[lat_field] >= MIN_LATITUDE) &
        (hgt_data[lat_field] <= MAX_LATITUDE), drop=True)

    monthly_data = valid_data.resample({time_field: '1M'}).mean(time_field)

    if climatology is None:
        climatology = monthly_data.groupby(
            monthly_data[time_field].dt.month).mean(time_field)

    anom = monthly_data.groupby(
        monthly_data[time_field].dt.month) - climatology

    return anom, climatology


def calculate_annual_eof(anom_data, ao_mode=DEFAULT_AO_MODE,
                         time_field=DEFAULT_TIME_FIELD,
                         lat_weights='scos', lat_field=DEFAULT_LAT_FIELD,
                         hgt_field=DEFAULT_HGT_FIELD,
                         n_eofs=N_EOFS, random_state=None):
    valid_data = anom_data.where(
        (anom_data[lat_field] >= MIN_LATITUDE) &
        (anom_data[lat_field] <= MAX_LATITUDE), drop=True)

    lat_data = valid_data[lat_field]

    weights = _get_lat_weights(lat_data, lat_weights=lat_weights)
    weights = xr.broadcast(valid_data.isel({time_field: 0}), weights)[1]

    eofs_result = calc_eofs(
        valid_data.values, weights=weights.values, n_eofs=n_eofs,
        random_state=random_state)

    eofs_data = eofs_result['eofs'][ao_mode][np.newaxis, ...]
    eofs_dims = ([EOF_DIM_NAME] +
                 [d for d in valid_data.dims if d != time_field])
    eofs_coords = valid_data.coords.to_dataset().drop(
        time_field).reset_coords(drop=True)
    eofs_coords = eofs_coords.expand_dims(EOF_DIM_NAME, axis=0)
    eofs_coords.coords[EOF_DIM_NAME] = (EOF_DIM_NAME, np.arange(1))

    eofs_da = xr.DataArray(eofs_data, dims=eofs_dims,
                           coords=eofs_coords.coords)

    pcs_data = eofs_result['pcs'][:, ao_mode][:, np.newaxis]
    pcs_dims = [time_field, EOF_DIM_NAME]
    pcs_coords = {time_field: valid_data[time_field].values,
                  EOF_DIM_NAME: np.arange(1)}
    pcs_da = xr.DataArray(pcs_data, dims=pcs_dims, coords=pcs_coords)

    ev = eofs_result['explained_variance'][ao_mode]
    evr = eofs_result['explained_variance_ratio'][ao_mode]
    sv = eofs_result['singular_values'][ao_mode]

    annual_eofs = {
        'explained_variance': ev,
        'explained_variance_ratio': evr,
        'singular_values': sv,
        'eofs': eofs_da,
        'pcs': pcs_da}

    return annual_eofs


def calculate_ao_pc_index(anom_data, eofs_data,
                          ddof=0,
                          lat_weights='scos',
                          time_field=DEFAULT_TIME_FIELD,
                          lat_field=DEFAULT_LAT_FIELD,
                          normalization=1):
    n_eofs = eofs_data.shape[0]

    pcs = _project_data(anom_data, eofs_data, lat_weights=lat_weights,
                        time_field=time_field, lat_field=lat_field)

    pcs_dims = [time_field, EOF_DIM_NAME]
    pcs_coords = {time_field: anom_data[time_field].values,
                  EOF_DIM_NAME: np.arange(n_eofs)}
    pcs_da = xr.DataArray(pcs, dims=pcs_dims, coords=pcs_coords)

    pcs_da = pcs_da / normalization

    return pcs_da


__all__ = ['calculate_daily_region_anomalies',
           'calculate_monthly_region_anomalies',
           'calculate_annual_eof',
           'calculate_ao_pc_index']
