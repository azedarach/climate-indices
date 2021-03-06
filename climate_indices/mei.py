import datetime
import numpy as np
import xarray as xr

from .eofs import calc_eofs


DEFAULT_TIME_FIELD = 'time'
DEFAULT_LAT_FIELD = 'lat'
DEFAULT_LON_FIELD = 'lon'
DEFAULT_SLP_FIELD = 'PRMSL_GDS0_MSL'
DEFAULT_SST_FIELD = 'BRTMP_GDS0_SFC'
DEFAULT_UWND_FIELD = 'UGRD_GDS0_HTGL'
DEFAULT_VWND_FIELD = 'VGRD_GDS0_HTGL'
DEFAULT_OLR_FIELD = 'olr'

DEFAULT_VARIABLES = [
    DEFAULT_SLP_FIELD,
    DEFAULT_SST_FIELD,
    DEFAULT_UWND_FIELD,
    DEFAULT_VWND_FIELD,
    DEFAULT_OLR_FIELD]

MONTHS_PER_SEASON = 2

VALID_WEIGHTS = ['none', 'cos', 'scos']
N_EOFS = 5

EOF_DIM_NAME = 'mode'
SEASON_DIM_NAME = 'season'

SEASONS = [
    ('DJ', (12, 1)),
    ('JF', (1, 2)),
    ('FM', (2, 3)),
    ('MA', (3, 4)),
    ('AM', (4, 5)),
    ('MJ', (5, 6)),
    ('JJ', (6, 7)),
    ('JA', (7, 8)),
    ('AS', (8, 9)),
    ('SO', (9, 10)),
    ('ON', (10, 11)),
    ('ND', (11, 12))
]


def get_season_name(s):
    n_seasons = len(SEASONS)
    if s >= 1 and s <= n_seasons:
        return SEASONS[s - 1][0]
    else:
        raise ValueError('invalid season index %d' % s)


def calculate_daily_anomalies(ds, climatology=None,
                              time_field=DEFAULT_TIME_FIELD):
    if climatology is None:
        climatology = ds.groupby(
            ds[time_field].dt.dayofyear).mean(time_field)

    anom = ds.groupby(ds[time_field].dt.dayofyear) - climatology

    return anom, climatology


def calculate_monthly_anomalies(ds, climatology=None,
                                time_field=DEFAULT_TIME_FIELD):
    monthly_ds = ds.resample({time_field: '1M'}).mean(time_field)

    if climatology is None:
        climatology = monthly_ds.groupby(
            monthly_ds[time_field].dt.month).mean(time_field)

    anom = monthly_ds.groupby(
        monthly_ds[time_field].dt.month) - climatology

    return anom, climatology


def get_seasonal_data(ds, time_field=DEFAULT_TIME_FIELD):
    monthly_ds = ds.resample({time_field: '1MS'}).mean(time_field)

    # NB the date assigned to each sample corresponds to
    # the first day of the second month of the season (for
    # bimonthly seasons)
    seasonal_ds = monthly_ds.rolling(
        {time_field: MONTHS_PER_SEASON},
        center=True, min_periods=1).mean()

    return seasonal_ds


def calculate_seasonal_anomalies(seasonal_ds, climatology=None,
                                 time_field=DEFAULT_TIME_FIELD,):
    if climatology is None:
        climatology = seasonal_ds.groupby(
            seasonal_ds[time_field].dt.month).mean(time_field)

    anom = seasonal_ds.groupby(
        seasonal_ds[time_field].dt.month) - climatology

    return anom, climatology


def standardize_seasonal_values(seasonal_ds,
                                time_field=DEFAULT_TIME_FIELD,
                                clim_start_year=None,
                                clim_end_year=None,
                                skipna=True, ddof=0):
    if clim_start_year is None:
        clim_start_year = int(seasonal_ds[time_field].dt.year.min())
    if clim_end_year is None:
        clim_end_year = int(seasonal_ds[time_field].dt.year.max())

    ref_ds = seasonal_ds.where(
        (seasonal_ds[time_field].dt.year >= clim_start_year) &
        (seasonal_ds[time_field].dt.year <= clim_end_year),
        drop=True)

    means = ref_ds.groupby(ref_ds[time_field].dt.month).mean(
        time_field, skipna=skipna)
    stds = ref_ds.groupby(ref_ds[time_field].dt.month).std(
        time_field, ddof=ddof, skipna=skipna)

    return xr.apply_ufunc(
        lambda x, m, s: (x - m) / s,
        seasonal_ds.groupby(seasonal_ds[time_field].dt.month),
        means, stds, dask='allowed')


def standardize_values(ds, time_field=DEFAULT_TIME_FIELD,
                       clim_start_year=None, clim_end_year=None,
                       skipna=True, ddof=0):
    if clim_start_year is None:
        clim_start_year = int(ds[time_field].dt.year.min())
    if clim_end_year is None:
        clim_end_year = int(ds[time_field].dt.year.max())

    ref_ds = ds.where((ds[time_field].dt.year >= clim_start_year) &
                      (ds[time_field].dt.year <= clim_end_year),
                      drop=True)

    means = ref_ds.mean(time_field, skipna=skipna)
    stds = ref_ds.std(time_field, skipna=skipna, ddof=ddof)

    std_ds = (ds - means) / stds

    return std_ds


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


def _get_combined_data(ds, variables=DEFAULT_VARIABLES,
                       time_field=DEFAULT_TIME_FIELD):
    n_variables = len(variables)
    var_data = {v: ds[v] for v in variables}

    if ds[time_field].shape:
        n_samples = ds[time_field].shape[0]
    else:
        n_samples = 1

    flat_var_data = [None] * n_variables
    for i, v in enumerate(variables):
        if n_samples == 1:
            var_shape = var_data[v].shape
        else:
            var_shape = var_data[v].isel({time_field: 0}).shape
        n_features = np.product(var_shape)
        flat_var_data[i] = np.reshape(
            var_data[v].values, (n_samples, n_features))

    return np.hstack(flat_var_data)


def _get_separated_data(x, ds, variables=DEFAULT_VARIABLES,
                        time_field=DEFAULT_TIME_FIELD):
    separated_data = {v: None for v in variables}

    n_samples = x.shape[0]

    pos = 0
    for v in variables:
        var_data = ds[v]

        var_shape = var_data.isel({time_field: 0}).shape
        n_features = np.product(var_shape)

        flat_var_data = x[:, pos:pos + n_features]
        separated_data[v] = np.reshape(flat_var_data, (n_samples,) + var_shape)

        pos += n_features

    return separated_data


def calculate_seasonal_eofs(anom_ds, time_field=DEFAULT_TIME_FIELD,
                            lat_field=DEFAULT_LAT_FIELD,
                            lon_field=DEFAULT_LON_FIELD,
                            variables=DEFAULT_VARIABLES,
                            lat_weights='scos',
                            n_eofs=N_EOFS, random_state=None,
                            **kwargs):
    present_seasons = np.unique(anom_ds[time_field].dt.month.values)
    n_seasons = np.size(present_seasons)

    lat_data = anom_ds[lat_field]

    n_samples = anom_ds.sizes[time_field]
    variable_dims = {}
    eofs_results = {}
    pcs_values = np.empty((n_samples, n_eofs))
    for v in variables:
        variable_dims[v] = [d for d in anom_ds[v].dims if d != time_field]
        eofs_results[v] = np.empty(
            (n_seasons, n_eofs) +
            anom_ds[v].isel({time_field: 0}).shape)

    for i, s in enumerate(present_seasons):
        pc_mask = anom_ds[time_field].dt.month == s
        season_ds = anom_ds.where(anom_ds[time_field].dt.month == s,
                                  drop=True)

        weights = _get_lat_weights(lat_data, lat_weights=lat_weights)
        weighted_ds = season_ds.coords.to_dataset()
        for v in variables:
            var_data = season_ds[v]
            weighted_ds[v] = weights * var_data
            weighted_ds[v] = weighted_ds[v].transpose(
                *[d for d in var_data.dims])

        weighted_data = _get_combined_data(weighted_ds, variables,
                                           time_field=time_field)

        eofs, pcs, _, _ = calc_eofs(
            weighted_data, n_components=n_eofs,
            rowvar=False, random_state=random_state, **kwargs)

        var_eofs = _get_separated_data(
            eofs, season_ds, variables=variables,
            time_field=time_field)

        for v in variables:
            eofs_results[v][i] = var_eofs[v]

        pcs_values[pc_mask] = pcs

    var_data = {v: ([SEASON_DIM_NAME, EOF_DIM_NAME] + variable_dims[v],
                    eofs_results[v]) for v in variables}

    eofs_coords = anom_ds.coords.to_dataset().drop(
        time_field).reset_coords(drop=True)
    eofs_coords = eofs_coords.expand_dims(EOF_DIM_NAME, axis=0)
    eofs_coords = eofs_coords.expand_dims(SEASON_DIM_NAME, axis=0)

    eofs_coords.coords[EOF_DIM_NAME] = (EOF_DIM_NAME, np.arange(n_eofs))
    eofs_coords.coords[SEASON_DIM_NAME] = (SEASON_DIM_NAME, present_seasons)

    eofs_ds = xr.Dataset(var_data, coords=eofs_coords.coords)

    pcs_dims = [time_field, EOF_DIM_NAME]
    pcs_coords = {time_field: anom_ds[time_field].values,
                  EOF_DIM_NAME: np.arange(n_eofs)}
    pcs_da = xr.DataArray(pcs_values, dims=pcs_dims, coords=pcs_coords)

    return eofs_ds, pcs_da


def fix_phases(eofs_ds, pcs_da, time_field=DEFAULT_TIME_FIELD,
               lon_field=DEFAULT_LON_FIELD, lat_field=DEFAULT_LAT_FIELD,
               variables=DEFAULT_VARIABLES,
               sst_field=DEFAULT_SST_FIELD):
    present_seasons = np.unique(pcs_da[time_field].dt.month.values)

    for i, s in enumerate(present_seasons):
        season_eofs = eofs_ds.where(eofs_ds[SEASON_DIM_NAME] == s, drop=True)
        sst_eof = season_eofs[sst_field].isel({EOF_DIM_NAME: 0})

        min_lon = sst_eof[lon_field].min()
        max_lon = sst_eof[lon_field].max()

        if min_lon < 0 and max_lon <= 180:
            sst_eof = sst_eof.where((sst_eof[lon_field] <= -120) &
                                    (sst_eof[lon_field] >= -170) &
                                    (sst_eof[lat_field] >= -5) &
                                    (sst_eof[lat_field] <= 5), drop=True)
        else:
            sst_eof = sst_eof.where((sst_eof[lon_field] >= 190) &
                                    (sst_eof[lon_field] <= 240) &
                                    (sst_eof[lat_field] >= -5) &
                                    (sst_eof[lat_field] <= 5), drop=True)

        flipped_sst_eof = -sst_eof.copy()
        nino34_max_anom = sst_eof.max()
        flipped_nino34_max_anom = flipped_sst_eof.max()

        if flipped_nino34_max_anom > nino34_max_anom:
            for v in variables:
                eofs_ds[v] = xr.where(
                    eofs_ds.coords[SEASON_DIM_NAME] == s,
                    -eofs_ds[v], eofs_ds[v])
            pcs_da = xr.where(
                pcs_da[time_field].dt.month == s,
                -pcs_da, pcs_da)

    return eofs_ds, pcs_da


def _check_constant_missing_values(X):
    nan_mask = np.isnan(X)
    if not (nan_mask.any(axis=0) == nan_mask.all(axis=0)).all():
        raise ValueError(
            'array has missing values in variable locations')


def _project_data(anom_ds, eofs_ds, lat_weights='scos',
                  time_field=DEFAULT_TIME_FIELD,
                  lat_field=DEFAULT_LAT_FIELD,
                  variables=DEFAULT_VARIABLES):
    combined_data = _get_combined_data(anom_ds, variables,
                                       time_field=time_field)

    weights = _get_lat_weights(anom_ds[lat_field], lat_weights=lat_weights)
    weights_ds = anom_ds.coords.to_dataset()
    for v in variables:
        var_data = anom_ds[v]
        var_weights = xr.broadcast(var_data, weights)[1]
        weights_ds[v] = var_weights

    weights = _get_combined_data(weights_ds, variables,
                                 time_field=time_field)

    weighted_data = combined_data * weights

    _check_constant_missing_values(weighted_data)
    nonnan_idx = np.where(np.logical_not(np.isnan(weighted_data[0])))[0]
    valid_data = weighted_data[:, nonnan_idx]

    combined_eofs = _get_combined_data(eofs_ds, variables,
                                       time_field=EOF_DIM_NAME)

    _check_constant_missing_values(combined_eofs)
    nonnan_idx = np.where(np.logical_not(np.isnan(combined_eofs[0])))[0]
    valid_eofs = combined_eofs[:, nonnan_idx]

    return np.dot(valid_data, valid_eofs.T)


def calculate_seasonal_mei(seasonal_anom_ds, eofs_ds, ref_pcs_da,
                           time_field=DEFAULT_TIME_FIELD,
                           lat_field=DEFAULT_LAT_FIELD, lat_weights='scos',
                           variables=DEFAULT_VARIABLES):
    present_seasons = np.unique(seasonal_anom_ds[time_field].dt.month.values)

    n_times = seasonal_anom_ds.sizes[time_field]

    pcs_values = np.empty((n_times,))

    seasons = seasonal_anom_ds[time_field].dt.month
    for s in present_seasons:
        mask = seasons == s
        season_ds = seasonal_anom_ds.where(
            seasonal_anom_ds[time_field].dt.month == s, drop=True)

        season_eofs_ds = eofs_ds.where(eofs_ds[SEASON_DIM_NAME] == s,
                                       drop=True)
        season_ref_pcs = ref_pcs_da.where(ref_pcs_da[time_field].dt.month == s,
                                          drop=True)

        season_pcs = _project_data(season_ds, season_eofs_ds,
                                   lat_weights=lat_weights,
                                   time_field=time_field,
                                   lat_field=lat_field,
                                   variables=variables)

        pcs_values[mask] = season_pcs[:, 0]
        normalization = season_ref_pcs.std(time_field).values[0]

        pcs_values[mask] /= normalization

    pcs_dims = [time_field]
    pcs_coords = {time_field: seasonal_anom_ds[time_field].values}
    pcs_da = xr.DataArray(pcs_values, dims=pcs_dims, coords=pcs_coords)

    return pcs_da


def calculate_daily_mei(anom_ds, eofs_ds, ref_pcs_da,
                        time_field=DEFAULT_TIME_FIELD,
                        lat_field=DEFAULT_LAT_FIELD, lat_weights='scos',
                        variables=DEFAULT_VARIABLES, interpolate=True):
    n_times = anom_ds.sizes[time_field]

    pcs_values = np.empty((n_times,))

    for i in range(n_times):
        month = anom_ds[time_field].dt.month.values[i]
        data = anom_ds.isel({time_field: i})
        if interpolate and i != n_times - 1:
            data_date = datetime.datetime(
                int(data[time_field].dt.year.values),
                month, int(data[time_field].dt.day.values))
            season_start_date = datetime.datetime(
                int(data[time_field].dt.year.values),
                month, 1)

            if month == 12:
                next_month = 1
                next_season_date = datetime.datetime(
                    int(data[time_field].dt.year.values + 1),
                    next_month, 1)
            else:
                next_month = month + 1
                next_season_date = datetime.datetime(
                    int(data[time_field].dt.year.values),
                    next_month, 1)

            frac = ((data_date - season_start_date) /
                    (next_season_date - season_start_date))

            current_season_eofs_ds = eofs_ds.sel({SEASON_DIM_NAME: month})
            next_season_eofs_ds = eofs_ds.sel({SEASON_DIM_NAME: next_month})

            season_eofs_ds = (frac * current_season_eofs_ds +
                              (1 - frac) * next_season_eofs_ds)

            season_ref_pcs = ref_pcs_da.where(
                ref_pcs_da[time_field].dt.month == month, drop=True)
            normalization = season_ref_pcs.std(time_field).values[0]
        else:
            season_eofs_ds = current_season_eofs_ds
            season_ref_pcs = ref_pcs_da.where(
                ref_pcs_da[time_field].dt.month == month, drop=True)
            normalization = season_ref_pcs.std(time_field).values[0]

        season_pcs = _project_data(data, season_eofs_ds,
                                   lat_weights=lat_weights,
                                   time_field=time_field,
                                   lat_field=lat_field,
                                   variables=variables)

        pcs_values[i] = season_pcs[:, 0] / normalization

    pcs_dims = [time_field]
    pcs_coords = {time_field: anom_ds[time_field].values}
    pcs_da = xr.DataArray(pcs_values, dims=pcs_dims, coords=pcs_coords)

    return pcs_da


__all__ = ['EOF_DIM_NAME', 'SEASON_DIM_NAME',
           'calculate_seasonal_eofs',
           'calculate_daily_mei',
           'calculate_seasonal_mei',
           'calculate_daily_anomalies',
           'calculate_monthly_anomalies',
           'calculate_seasonal_anomalies',
           'get_season_name',
           'standardize_seasonal_values',
           'standardize_values']
