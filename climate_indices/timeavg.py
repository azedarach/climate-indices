import datetime
import numpy as np
import warnings

from .utils._dates import (_central_month_to_season,
                           _season_to_central_month,
                           _get_months_in_season,
                           _get_all_days,
                           _get_month_mask, _get_month_day_mask,
                           _get_year_month_mask,
                           _get_year_month_day_mask, _get_middle_date_of_month,
                           _get_middle_day_of_months)
from .utils._validation import _check_matching_lengths


def _count_valid(data, ignore_nan=True):
    if not ignore_nan:
        n_samples = data.shape[0]
        return np.full(data.shape[1:], n_samples)
    else:
        return np.sum(~np.isnan(data), axis=0)


def _masked_time_mean(time_mask, data, ignore_nan=True, dtype=None,
                      min_records=None):
    masked_data = data[time_mask]

    valid_counts = _count_valid(masked_data, ignore_nan=ignore_nan)

    if ignore_nan:
        calc_mean = np.nanmean
    else:
        calc_mean = np.mean

    result = calc_mean(masked_data, axis=0, dtype=dtype)

    if min_records is not None:
        if np.any(valid_counts < min_records):
            warnings.warn('number of valid records fewer than min_records',
                          warnings.UserWarning)
        result[valid_counts < min_records] = np.NaN

    return result


def _masked_time_std(time_mask, data, ignore_nan=True, dtype=None,
                     min_records=None, ddof=0):
    masked_data = data[time_mask]

    valid_counts = _count_valid(masked_data, ignore_nan=ignore_nan)

    if ignore_nan:
        calc_std = np.nanstd
    else:
        calc_std = np.std

    result = calc_std(masked_data, axis=0, dtype=dtype, ddof=ddof)

    if min_records is not None:
        if np.any(valid_counts < min_records):
            warnings.warn('number of valid records fewer than min_records',
                          warnings.UserWarning)
        result[valid_counts < min_records] = np.NaN

    return result


def daily_means(time, data, ignore_nan=True, dtype=None,
                min_records=None, dummy_timedelta=None):
    """Calculate daily mean for each date in the given data."""
    _check_matching_lengths(time, data, 'daily_means')

    days = np.array([[t.year, t.month, t.day] for t in time],
                    dtype='i8')
    unique_days = np.unique(days, axis=0)

    n_days = unique_days.shape[0]
    means = np.empty((n_days,) + data.shape[1:])
    for i, d in enumerate(unique_days):
        mask = _get_year_month_day_mask(
            time, valid_years=[d[0]], valid_months=[d[1]],
            valid_days=[d[2]])
        means[i] = _masked_time_mean(
            mask, data, ignore_nan=ignore_nan, dtype=dtype,
            min_records=min_records)

    if dummy_timedelta is None:
        offset = datetime.timedelta()
    else:
        offset = dummy_timedelta

    day_times = np.array([datetime.datetime(*d) + offset
                          for d in unique_days])

    return day_times, means


def multiyear_daily_means(time, data, ignore_leap=False,
                          ignore_nan=True, dtype=np.float64,
                          min_records=None, min_daily_records=None):
    """Calculate daily means across multiple years."""
    _check_matching_lengths(time, data, 'multiyear_daily_means')

    daily_time, daily_data = daily_means(
        time, data, ignore_nan=ignore_nan, dtype=dtype,
        min_records=min_daily_records)

    days = _get_all_days(ignore_leap=ignore_leap)
    n_days = days.shape[0]
    means = np.empty((n_days,) + data.shape[1:])
    for i, d in enumerate(days):
        mask = _get_month_day_mask(
            daily_time, valid_months=[d[0]], valid_days=[d[1]])
        n_valid_times = np.sum(mask)
        if n_valid_times == 0:
            warnings.warn(
                'no data for (day %d, month %d)' % (d[1], d[0]))
            means[i] = np.NaN
        else:
            means[i] = _masked_time_mean(
                mask, daily_data, ignore_nan=ignore_nan, dtype=dtype,
                min_records=min_records)

    return days, means


def _add_month_time_offset(dt, offset=None):
    if offset is None:
        return _get_middle_day_of_months(dt)
    else:
        return dt + offset


def _get_present_months(dt):
    months = np.array([t.month for t in dt], dtype='i8')
    return np.unique(months)


def monthly_means(time, data, ignore_nan=True, dtype=None,
                  min_records=None, dummy_timedelta=None):
    """Calculate monthly mean for each month and year in the given data."""
    _check_matching_lengths(time, data, 'monthly_means')

    months = np.array([[t.year, t.month, 1] for t in time], dtype='i8')
    unique_months = np.unique(months, axis=0)

    n_months = unique_months.shape[0]
    means = np.empty((n_months,) + data.shape[1:])
    for i, m in enumerate(unique_months):
        mask = _get_year_month_mask(
            time, valid_years=[m[0]], valid_months=[m[1]])
        means[i] = _masked_time_mean(
            mask, data, ignore_nan=ignore_nan, dtype=dtype,
            min_records=min_records)

    month_times = np.array([datetime.datetime(*d) for d in unique_months])
    month_times = _add_month_time_offset(month_times, offset=dummy_timedelta)

    return month_times, means


def multiyear_monthly_means(time, data, ignore_nan=True, dtype=None,
                            min_records=None, min_monthly_records=None):
    """Calculate monthly means across multiple years."""
    _check_matching_lengths(time, data, 'multiyear_monthly_means')

    monthly_time, monthly_data = monthly_means(
        time, data, ignore_nan=ignore_nan, dtype=dtype,
        min_records=min_monthly_records)

    months_in_data = _get_present_months(time)
    n_months = np.size(months_in_data)

    means = np.empty((n_months,) + monthly_data.shape[1:])
    for i in range(n_months):
        month = months_in_data[i]
        mask = _get_month_mask(monthly_time, valid_months=[month])
        n_valid_times = np.sum(mask)
        if n_valid_times == 0:
            warnings.warn(
                'no data for month %d' % month,
                warnings.UserWarning)
            means[i] = np.NaN
        else:
            means[i] = _masked_time_mean(
                mask, monthly_data, ignore_nan=ignore_nan, dtype=dtype,
                min_records=min_records)

    return months_in_data, means


def _get_present_years_and_seasons(dt):
    time_points = np.array([[t.year, t.month] for t in dt], dtype='i8')
    unique_time_points = np.unique(time_points, axis=0)
    return [[t[0], _central_month_to_season(t[1])] for t in unique_time_points]


def _get_present_seasons(dt):
    ys = _get_present_years_and_seasons(dt)
    central_months = np.array([_season_to_central_month(t[1]) for t in ys],
                              dtype='i8')
    unique_months = np.unique(central_months)
    return [_central_month_to_season(m) for m in unique_months]


def _get_season_time(time_points, offset=None):
    ym = np.array([[tp[0], _season_to_central_month(tp[1])]
                   for tp in time_points], dtype='i8')
    base_times = np.array([_get_middle_date_of_month(
        datetime.datetime(d[0], d[1], 1)) for d in ym])
    if offset is not None:
        for i in range(base_times.shape[0]):
            base_times[i] += offset
    return base_times


def _get_season_mask(dt, valid_seasons=None):
    if valid_seasons is None:
        return np.ones(dt.shape, dtype=bool)

    # note: for each element, the value of the month
    # is assumed to give the central month of the season
    months = np.array([t.month for t in dt], dtype='i8')
    valid_central_months = [_season_to_central_month(s)
                            for s in valid_seasons]
    n_valid_months = len(valid_central_months)

    mask = np.zeros(dt.shape, dtype=bool)
    for i in range(n_valid_months):
        mask = np.logical_or(mask, months == valid_central_months[i])

    return mask


def seasonal_means(time, data, ignore_nan=True, dtype=None,
                   min_records=None, dummy_timedelta=None):
    """Calculate seasonal mean for each season and year in the given data."""
    _check_matching_lengths(time, data, 'seasonal_means')

    time_points = _get_present_years_and_seasons(time)
    n_time_points = len(time_points)

    means = np.empty((n_time_points,) + data.shape[1:])
    for i, tp in enumerate(time_points):
        valid_years = [tp[0]]
        valid_months = _get_months_in_season(tp[1])
        mask = _get_year_month_mask(
            time, valid_years=valid_years, valid_months=valid_months)
        means[i] = _masked_time_mean(
            mask, data, ignore_nan=ignore_nan, dtype=dtype,
            min_records=min_records)

    season_times = _get_season_time(time_points, offset=dummy_timedelta)

    return season_times, means


def multiyear_seasonal_means(time, data, ignore_nan=True, dtype=None,
                             min_records=None, min_seasonal_records=None):
    """Calculate seasonal means across multiple years."""
    _check_matching_lengths(time, data, 'multiyear_seasonal_means')

    seasonal_time, seasonal_data = seasonal_means(
        time, data, ignore_nan=ignore_nan, dtype=dtype,
        min_records=min_seasonal_records)

    seasons_in_data = _get_present_seasons(time)
    n_seasons = np.size(seasons_in_data)

    means = np.empty((n_seasons,) + seasonal_data.shape[1:])
    for i in range(n_seasons):
        season = seasons_in_data[i]
        mask = _get_season_mask(seasonal_time, valid_seasons=[season])
        means[i] = _masked_time_mean(
            mask, seasonal_data, ignore_nan=ignore_nan, dtype=dtype,
            min_records=min_records)

    return seasons_in_data, means


def multiyear_seasonal_mean_std(time, data, ignore_nan=True, dtype=None,
                                min_records=None, min_seasonal_records=None,
                                ddof=1):
    """Calculate seasonal means and standard deviations for multiple years."""
    _check_matching_lengths(time, data, 'multiyear_seasonal_means_std')

    seasonal_time, seasonal_data = seasonal_means(
        time, data, ignore_nan=ignore_nan, dtype=dtype,
        min_records=min_seasonal_records)

    seasons_in_data = _get_present_seasons(time)
    n_seasons = np.size(seasons_in_data)

    means = np.empty((n_seasons,) + seasonal_data.shape[1:])
    stds = np.empty((n_seasons,) + seasonal_data.shape[1:])
    for i in range(n_seasons):
        season = seasons_in_data[i]
        mask = _get_season_mask(seasonal_time, valid_seasons=[season])
        means[i] = _masked_time_mean(
            mask, seasonal_data, ignore_nan=ignore_nan, dtype=dtype,
            min_records=min_records)
        stds[i] = _masked_time_std(
            mask, seasonal_data, ignore_nan=ignore_nan, dtype=dtype,
            min_records=min_records, ddof=ddof)

    return seasons_in_data, means, stds


__all__ = ['daily_means', 'multiyear_daily_means',
           'monthly_means', 'multiyear_monthly_means',
           'seasonal_means', 'multiyear_seasonal_means',
           'multiyear_seasonal_mean_std']
