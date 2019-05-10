import numpy as np

from .anomalies import monthly_anomalies
from .eofs import calc_eofs
from .timeavg import multiyear_monthly_means

from .utils._validation import _check_array_shape, _check_matching_lengths


MIN_LATITUDE = -90
MAX_LATITUDE = -20
N_EOFS = 1


def _get_time_mask(time, start_year=None, end_year=None):
    years = np.array([t.year for t in time], dtype='i8')
    if start_year is None and end_year is None:
        return np.ones(np.shape(time), dtype=bool)
    elif start_year is None and end_year is not None:
        return years <= end_year
    elif start_year is not None and end_year is None:
        return years >= start_year
    else:
        return np.logical_and(years >= start_year, years <= end_year)


def _get_latitude_mask(lat):
    return np.logical_and(lat >= MIN_LATITUDE, lat <= MAX_LATITUDE)


def _get_analysis_masks(time, lat, lon, start_year=None, end_year=None):
    time_mask = _get_time_mask(time, start_year=start_year, end_year=end_year)
    if np.sum(time_mask) == 0:
        raise ValueError('no valid time data found')

    lat_mask = _get_latitude_mask(lat)
    if np.sum(lat_mask) == 0:
        raise ValueError('no valid latitude data found')

    lon_mask = np.ones(np.shape(lon), dtype=bool)

    return time_mask, lat_mask, lon_mask


def _check_input_data(time, lat, lon, data, whom):
    _check_matching_lengths(time, data, whom)

    n_times = np.size(time)
    n_lat = np.size(lat)
    n_lon = np.size(lon)

    _check_array_shape(data, (n_times, n_lat, n_lon), whom)


def _get_scos_weights(lat, data):
    clat = np.cos(np.deg2rad(lat)).clip(0., 1.)
    weights = np.sqrt(clat)[..., np.newaxis]

    weights = np.broadcast_arrays(
        data[0:1], weights)[1][0]

    return weights


def _project_data(data, eof):
    n_samples = data.shape[0]
    expected_shape = data.shape[1:]
    n_features = np.product(expected_shape)

    _check_array_shape(eof, expected_shape, '_project_data')

    flat_data = np.reshape(data, (n_samples, n_features))
    flat_eof = np.reshape(eof, (n_features,))

    projection = np.dot(flat_data, flat_eof)

    return projection


def calc_aao_from_anomalies(time, lat, lon, monthly_z700_anom,
                            start_year=None, end_year=None,
                            random_state=None):
    """Calculate AAO EOF pattern and PC time series from given anomalies.

    Parameters
    ----------
    time : array-like, shape (n_samples)
        Array containing time points (i.e., months) at which data is given.

    lat : array-like, shape (n_lat)
        Array containing the latitude points at which the data is given.

    lon : array-like, shape (n_lon)
        Array containing the longitude points at which the data is given.

    monthly_z700_anom : array-like, shape (n_samples, n_lat, n_lon)
        Monthly mean 700 hPa geopotential height anomalies data.

    start_year : integer or None
        If an integer, the first year to include in calculating the pattern.
        If None, all years in the given data are included.

    end_year : integer or None
        If an integer, the last year to include in calculating the pattern.
        If None, all years in the given data are included.

    min_records : integer or None
        If given, the minimum number of records to be required in
        computing the climatology.

    random_state : integer, RandomState, or None
        If an integer, random_state is the seed used by the
        random number generator. If a RandomState instance,
        random_state is the random number generator. If
        None, the random number generator is the
        RandomState instance used by `np.random`.

    Returns
    -------
    pcs : array-like, shape (n_samples)
        Array containing the PC timeseries of the first EOF mode.

    eof : array-like, shape (n_lat, n_lon)
        Array containing the first EOF mode on the given latitude-longitude
        grid.

    References
    ----------
    See description of procedure at
    https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/history/method.shtml
    """
    _check_input_data(time, lat, lon, monthly_z700_anom,
                      'calc_aao_from_anomalies')

    time_mask, lat_mask, _ = _get_analysis_masks(
        time, lat, lon, start_year=start_year, end_year=end_year)

    valid_lat = lat[lat_mask]
    valid_data = monthly_z700_anom[time_mask, :, :]
    valid_data = valid_data[:, lat_mask, :]

    weights = _get_scos_weights(valid_lat, valid_data)

    pcs, eofs, _, _, _ = calc_eofs(
        valid_data, weights=weights, n_eofs=N_EOFS,
        random_state=random_state)

    return pcs, np.squeeze(eofs), valid_lat, lon


def calc_aao(time, lat, lon, monthly_z700, start_year=None, end_year=None,
             ignore_nan=True, dtype=None, min_records=None,
             random_state=None):
    """Calculate AAO EOF pattern and PC time series.

    The multiyear monthly climatology is first calculated from the
    given monthly data, from which the monthly mean anomalies are
    calculated. The resulting anomalies are then used to compute
    the AO pattern and the corresponding PC timeseries.

    Parameters
    ----------
    time : array-like, shape (n_samples)
        Array containing time points (i.e., months) at which data is given.

    lat : array-like, shape (n_lat)
        Array containing the latitude points at which the data is given.

    lon : array-like, shape (n_lon)
        Array containing the longitude points at which the data is given.

    monthly_z700 : array-like, shape (n_samples, n_lat, n_lon)
        Monthly mean 700 hPa geopotential height data.

    start_year : integer or None
        If an integer, the first year to include in calculating the pattern.
        If None, all years in the given data are included.

    end_year : integer or None
        If an integer, the last year to include in calculating the pattern.
        If None, all years in the given data are included.

    ignore_nan : bool, default: True
        If True, ignore NaNs when computing means.

    dtype : optional
        If given, the dtype used in computing means.

    min_records : integer or None
        If given, the minimum number of records to be required in
        computing the climatology.

    random_state : integer, RandomState, or None
        If an integer, random_state is the seed used by the
        random number generator. If a RandomState instance,
        random_state is the random number generator. If
        None, the random number generator is the
        RandomState instance used by `np.random`.

    Returns
    -------
    pcs : array-like, shape (n_samples)
        Array containing the PC timeseries of the first EOF mode.

    eof : array-like, shape (n_lat, n_lon)
        Array containing the first EOF mode on the given latitude-longitude
        grid.

    References
    ----------
    See description of procedure at
    https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/history/method.shtml
    """
    _check_input_data(time, lat, lon, monthly_z700, 'calc_aao')
    time_mask, lat_mask, _ = _get_analysis_masks(
        time, lat, lon, start_year=start_year, end_year=end_year)

    valid_time = time[time_mask]
    valid_lat = lat[lat_mask]
    valid_data = monthly_z700[time_mask, :, :]
    valid_data = valid_data[:, lat_mask, :]

    monthly_clim = multiyear_monthly_means(
        valid_time, valid_data, ignore_nan=ignore_nan, dtype=dtype,
        min_records=min_records)

    anom = monthly_anomalies(valid_time, valid_data, monthly_clim)

    return calc_aao_from_anomalies(valid_time, valid_lat, lon, anom,
                                   start_year=start_year, end_year=end_year,
                                   random_state=random_state)


def calc_aao_index(time, lat, lon, z700_anom, aao_eof,
                   start_year=None, end_year=None,
                   normalization=None):
    time_mask, lat_mask, _ = _get_analysis_masks(
        time, lat, lon, start_year=start_year, end_year=end_year)

    valid_time = time[time_mask]
    valid_lat = lat[lat_mask]
    valid_data = z700_anom[time_mask, :, :]
    valid_data = valid_data[:, lat_mask, :]

    weights = _get_scos_weights(valid_lat, valid_data)
    weighted_data = weights * valid_data

    index = _project_data(weighted_data, aao_eof)

    if normalization is None:
        normalization = index.std(ddof=1)

    print('normalization = ', normalization)

    return valid_time, index / normalization
