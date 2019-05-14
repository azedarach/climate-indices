import datetime
import numpy as np
import xarray as xr


DEFAULT_TIME_FIELD = 'time'
DEFAULT_LAT_FIELD = 'lat'
DEFAULT_LON_FIELD = 'lon'
DEFAULT_HGT_FIELD = 'hgt'

NH_PHI_N = 80.0
NH_PHI_0 = 60.0
NH_PHI_S = 40.0

SH_PHI_N = -35.0
SH_PHI_0 = -50.0
SH_PHI_S = -65.0


def calc_tibaldi(ds, time_field=DEFAULT_TIME_FIELD,
                 lat_field=DEFAULT_LAT_FIELD,
                 lon_field=DEFAULT_LON_FIELD, hgt_field=DEFAULT_HGT_FIELD,
                 southern_hemisphere=False, delta=5, window_length=0,
                 method='nearest'):
    offsets = np.array([-delta, 0, delta])

    if window_length == 0:
        n_samples = ds[time_field].shape[0]
    else:
        ds_t = ds.rolling(
            {time_field: window_length}).mean().dropna(time_field)
        n_samples = ds_t[time_field].shape[0]

    n_lon = ds[lon_field].shape[0]
    n_offsets = offsets.shape[0]

    ghgn = np.empty((n_samples, n_offsets, n_lon), dtype=ds[hgt_field].dtype)
    ghgs = np.empty((n_samples, n_offsets, n_lon), dtype=ds[hgt_field].dtype)

    for i, o in enumerate(offsets):
        if southern_hemisphere:
            phi_n = SH_PHI_N + o
            phi_0 = SH_PHI_0 + o
            phi_s = SH_PHI_S + o
        else:
            phi_n = NH_PHI_N + o
            phi_0 = NH_PHI_0 + o
            phi_s = NH_PHI_S + o

        ds_phi_n = ds.sel({lat_field: phi_n}, method=method)
        ds_phi_0 = ds.sel({lat_field: phi_0}, method=method)
        ds_phi_s = ds.sel({lat_field: phi_s}, method=method)

        if window_length > 0:
            ds_phi_n = ds_phi_n.rolling(
                {time_field: window_length}).mean().dropna(time_field)
            ds_phi_0 = ds_phi_0.rolling(
                {time_field: window_length}).mean().dropna(time_field)
            ds_phi_s = ds_phi_s.rolling(
                {time_field: window_length}).mean().dropna(time_field)

        z_phi_n = np.squeeze(ds_phi_n[hgt_field].values)
        z_phi_0 = np.squeeze(ds_phi_0[hgt_field].values)
        z_phi_s = np.squeeze(ds_phi_s[hgt_field].values)

        if southern_hemisphere:
            ghgs[:, i, :] = (z_phi_s - z_phi_0) / (phi_0 - phi_s)
            ghgn[:, i, :] = (z_phi_0 - z_phi_n) / (phi_n - phi_0)
        else:
            ghgs[:, i, :] = (z_phi_0 - z_phi_s) / (phi_0 - phi_s)
            ghgn[:, i, :] = (z_phi_n - z_phi_0) / (phi_n - phi_0)

    var_data = {'GHGN': ([time_field, 'lat_offset', lon_field], ghgn),
                'GHGS': ([time_field, 'lat_offset', lon_field], ghgs)}

    if window_length == 0:
        coords = {time_field: ds[time_field],
                  'lat_offset': (['lat_offset'], offsets),
                  lon_field: ds[lon_field]}
    else:
        coords = {time_field: ds_t[time_field],
                  'lat_offset': (['lat_offset'], offsets),
                  lon_field: ds[lon_field]}
    return xr.Dataset(var_data, coords=coords)


def _to_datetime(t):
    years = t.year
    months = t.month
    days = t.day

    n_samples = years.shape[0]

    return np.array([datetime.datetime(years[i], months[i], days[i])
                     for i in range(n_samples)])


def tibaldi_index_1d(tibaldi_ds, time_field=DEFAULT_TIME_FIELD,
                     lon_field=DEFAULT_LON_FIELD,
                     clim_start_year=None, clim_end_year=None,
                     southern_hemisphere=False):
    """Calculate the 1D Tibaldi index as the zonal mean maximum index value."""
    if southern_hemisphere:
        index_field = 'GHGN'
    else:
        index_field = 'GHGS'

    if clim_start_year is None:
        clim_start_year = int(tibaldi_ds[time_field].dt.year.min())
    if clim_end_year is None:
        clim_end_year = int(tibaldi_ds[time_field].dt.year.max())

    max_ds = tibaldi_ds.max('lat_offset')

    clim_ds = max_ds.where(
        (max_ds[time_field].dt.year >= clim_start_year) &
        (max_ds[time_field].dt.year <= clim_end_year), drop=True)

    clim = clim_ds.groupby(clim_ds[time_field].dt.dayofyear).mean(time_field)

    anom_ds = max_ds.groupby(max_ds[time_field].dt.dayofyear) - clim

    times = _to_datetime(anom_ds[time_field].dt)
    index = anom_ds[index_field].mean(lon_field).values

    return times, index
