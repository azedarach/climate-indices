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


def calculate_annual_eofs(anom_data,
                          time_field=DEFAULT_TIME_FIELD,
                          lat_weights='scos', lat_field=DEFAULT_LAT_FIELD,
                          hgt_field=DEFAULT_HGT_FIELD,
                          n_eofs=N_EOFS, random_state=None, **kwargs):
    valid_data = anom_data.where(
        (anom_data[lat_field] >= MIN_LATITUDE) &
        (anom_data[lat_field] <= MAX_LATITUDE), drop=True)

    lat_data = valid_data[lat_field]
    weights = _get_lat_weights(lat_data, lat_weights=lat_weights)

    weighted_data = weights * valid_data
    weighted_data = weighted_data.transpose(*[d for d in valid_data.dims])

    eofs, pcs, ev, evr = calc_eofs(
        weighted_data.values, n_components=n_eofs, rowvar=False,
        random_state=random_state, **kwargs)

    eofs_dims = ([EOF_DIM_NAME] +
                 [d for d in valid_data.dims if d != time_field])
    eofs_coords = valid_data.coords.to_dataset().drop(
        time_field).reset_coords(drop=True)
    eofs_coords = eofs_coords.expand_dims(EOF_DIM_NAME, axis=0)
    eofs_coords.coords[EOF_DIM_NAME] = (EOF_DIM_NAME, np.arange(n_eofs))

    eofs_da = xr.DataArray(eofs, dims=eofs_dims,
                           coords=eofs_coords.coords)

    pcs_dims = [time_field, EOF_DIM_NAME]
    pcs_coords = {time_field: valid_data[time_field].values,
                  EOF_DIM_NAME: np.arange(n_eofs)}
    pcs_da = xr.DataArray(pcs, dims=pcs_dims, coords=pcs_coords)

    ev_dims = [EOF_DIM_NAME]
    ev_coords = {EOF_DIM_NAME: np.arange(n_eofs)}
    ev_da = xr.DataArray(ev, dims=ev_dims, coords=ev_coords)

    evr_dims = [EOF_DIM_NAME]
    evr_coords = {EOF_DIM_NAME: np.arange(n_eofs)}
    evr_da = xr.DataArray(evr, dims=evr_dims, coords=evr_coords)

    annual_eofs = {
        'explained_variance': ev_da,
        'explained_variance_ratio': evr_da,
        'eofs': eofs_da,
        'pcs': pcs_da}

    return annual_eofs


def _fix_phase(eofs_data, time_field=DEFAULT_TIME_FIELD,
               lat_field=DEFAULT_LAT_FIELD):
    lat_max = np.asscalar(eofs_data.where(eofs_data == eofs_data.max(),
                                          drop=True)[lat_field])
    lat_min = np.asscalar(eofs_data.where(eofs_data == eofs_data.min(),
                                          drop=True)[lat_field])

    if lat_max > lat_min:
        return -eofs_data
    else:
        return eofs_data


def calculate_ao_pc_index(anom_data, eofs_data,
                          ao_mode=DEFAULT_AO_MODE, ddof=0,
                          lat_weights='scos',
                          time_field=DEFAULT_TIME_FIELD,
                          lat_field=DEFAULT_LAT_FIELD,
                          normalization=1):
    n_eofs = 1

    eof_data = eofs_data.where(eofs_data[EOF_DIM_NAME] == ao_mode,
                               drop=True)

    pos_phase_pattern = _fix_phase(eof_data, time_field=time_field,
                                   lat_field=lat_field)
    pcs = _project_data(anom_data, pos_phase_pattern, lat_weights=lat_weights,
                        time_field=time_field, lat_field=lat_field)

    pcs_dims = [time_field, EOF_DIM_NAME]
    pcs_coords = {time_field: anom_data[time_field].values,
                  EOF_DIM_NAME: np.arange(n_eofs)}
    pcs_da = xr.DataArray(pcs, dims=pcs_dims, coords=pcs_coords)

    pcs_da = pcs_da / normalization

    return pcs_da


__all__ = ['calculate_daily_region_anomalies',
           'calculate_monthly_region_anomalies',
           'calculate_annual_eofs',
           'calculate_ao_pc_index']
