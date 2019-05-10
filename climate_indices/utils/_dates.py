import datetime
import numpy as np


NUMBER_OF_DAYS = 366
NUMBER_OF_MONTHS = 12

SEASONS = (
    ('DJF', (12, 1, 2)),
    ('JFM', (1, 2, 3)),
    ('FMA', (2, 3, 4)),
    ('MAM', (3, 4, 5)),
    ('AMJ', (4, 5, 6)),
    ('MJJ', (5, 6, 7)),
    ('JJA', (6, 7, 8)),
    ('JAS', (7, 8, 9)),
    ('ASO', (8, 9, 10)),
    ('SON', (9, 10, 11)),
    ('OND', (10, 11, 12)),
    ('NDJ', (11, 12, 1))
)


def _central_month_to_season(month):
    if month == 1:
        return 'DJF'
    elif month == 2:
        return 'JFM'
    elif month == 3:
        return 'FMA'
    elif month == 4:
        return 'MAM'
    elif month == 5:
        return 'AMJ'
    elif month == 6:
        return 'MJJ'
    elif month == 7:
        return 'JJA'
    elif month == 8:
        return 'JAS'
    elif month == 9:
        return 'ASO'
    elif month == 10:
        return 'SON'
    elif month == 11:
        return 'OND'
    elif month == 12:
        return 'NDJ'
    else:
        raise ValueError('invalid month %d' % month)


def _season_to_central_month(season):
    if season == 'DJF':
        return 1
    elif season == 'JFM':
        return 2
    elif season == 'FMA':
        return 3
    elif season == 'MAM':
        return 4
    elif season == 'AMJ':
        return 5
    elif season == 'MJJ':
        return 6
    elif season == 'JJA':
        return 7
    elif season == 'JAS':
        return 8
    elif season == 'ASO':
        return 9
    elif season == 'SON':
        return 10
    elif season == 'OND':
        return 11
    elif season == 'NDJ':
        return 12
    else:
        raise ValueError("invalid season '%r'" % season)


def _get_months_in_season(season):
    season_names = [s[0] for s in SEASONS]

    if season not in season_names:
        raise ValueError("unrecognized season '%r'" % season)

    season_index = season_names.index(season)

    return SEASONS[season_index][1]


def _get_number_of_days_in_month(m, ignore_leap=False):
    if m in (1, 3, 5, 7, 8, 10, 12):
        return 31
    elif m in (4, 6, 9, 11):
        return 30
    elif m == 2:
        return 28 if ignore_leap else 29
    else:
        raise ValueError('invalid month %r' % m)


def _get_all_days(ignore_leap=False, dummy_year=1):
    days = []
    for m in range(1, NUMBER_OF_MONTHS + 1):
        n_days_in_month = _get_number_of_days_in_month(
            m, ignore_leap=ignore_leap)
        days += [[m, d]
                 for d in range(1, n_days_in_month + 1)]

    return np.asarray(days, dtype='i8')


def _get_middle_date_of_month(t):
    t1 = t.replace(day=1)
    if t1.month < 12:
        delta = t1.replace(month=(t.month + 1)) - t1
    else:
        delta = t1.replace(year=(t1.year + 1), month=1) - t1
    delta = datetime.timedelta(days=(delta / 2).days - 1)

    return t1 + delta


def _get_middle_day_of_months(dt):
    middle_days = dt.copy()

    for i in np.ndindex(dt.shape):
        middle_days[i] = _get_middle_date_of_month(dt[i])

    return middle_days


def _is_leap_day(date):
    return date.day == 29 and date.month == 2


def _contains_leap_days(dt):
    is_leap_day = np.array([_is_leap_day(d) for d in dt], dtype=bool)
    return np.any(is_leap_day)


def _get_day_of_year(day):
    return day.timetuple().tm_yday


def _get_day_mask(dt, valid_days=None):
    if valid_days is None:
        return np.ones(dt.shape, dtype=bool)

    days = np.array([t.day for t in dt], dtype='i8')

    mask = np.zeros(dt.shape, dtype=bool)
    for d in valid_days:
        mask = np.logical_or(mask, days == d)

    return mask


def _get_month_mask(dt, valid_months=None):
    if valid_months is None:
        return np.ones(dt.shape, dtype=bool)

    months = np.array([t.month for t in dt], dtype='i8')

    mask = np.zeros(dt.shape, dtype=bool)
    for m in valid_months:
        mask = np.logical_or(mask, months == m)

    return mask


def _get_season_mask(dt, valid_seasons=None):
    if valid_seasons is None:
        return np.ones(dt.shape, dtype=bool)

    central_months = np.array([t.month for t in dt], dtype='i8')

    mask = np.zeros(dt.shape, dtype=bool)
    for s in valid_seasons:
        m = _season_to_central_month(s)
        mask = np.logical_or(mask, central_months == m)

    return mask


def _get_year_mask(dt, valid_years=None):
    if valid_years is None:
        return np.ones(dt.shape, dtype=bool)

    years = np.array([t.year for t in dt], dtype='i8')
    mask = np.zeros(dt.shape, dtype=bool)
    for y in valid_years:
        mask = np.logical_or(mask, years == y)

    return mask


def _get_year_month_mask(dt, valid_years=None, valid_months=None):
    return np.logical_and(_get_year_mask(dt, valid_years=valid_years),
                          _get_month_mask(dt, valid_months=valid_months))


def _get_month_day_mask(dt, valid_months=None, valid_days=None):
    return np.logical_and(_get_month_mask(dt, valid_months=valid_months),
                          _get_day_mask(dt, valid_days=valid_days))


def _get_year_month_day_mask(dt, valid_years=None, valid_months=None,
                             valid_days=None):
    return np.logical_and(
        _get_year_month_mask(
            dt, valid_years=valid_years, valid_months=valid_months),
        _get_day_mask(dt, valid_days=valid_days))
