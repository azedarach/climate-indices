import xarray as xr
import numpy as np

from math import pi

EARTH_RADIUS = 6.37e6
GRAV_ACCEL = 9.81


DEFAULT_TIME_FIELD = 'time'
DEFAULT_LAT_FIELD = 'lat'
DEFAULT_LEVEL_FIELD = 'level'


DEFAULT_EXTENT_MIN_LEVEL_HPA = 400
DEFAULT_EXTENT_MAX_LEVEL_HPA = 600


ZMSF_NAME = 'zmsf'
ZERO_CROSSING_LOWER_NAME = 'lower'
ZERO_CROSSING_UPPER_NAME = 'upper'


def _get_zero_crossings_mask(X, axis=0, **kwargs):
    """Return masked array indicating location of sign changes.

    The result is a masked array, with True values indicating
    elements that differ in sign from the element preceding
    it along the requested axis.
    """
    mask = np.zeros(X.shape, dtype=bool)
    axis_dim = X.shape[axis]

    reordered = np.moveaxis(X, axis, 0)
    reordered_mask = np.moveaxis(mask, axis, 0)
    for i in range(1, axis_dim):
        x_prev = reordered[i - 1]
        x_cur = reordered[i]

        flags = np.sign(x_prev) != np.sign(x_cur)

        reordered_mask[i] = flags

    return mask


def _bound_zero_crossings(zmsf, level_field=DEFAULT_LEVEL_FIELD,
                          lat_field=DEFAULT_LAT_FIELD, **kwargs):
    bounds = {level_field: zmsf[level_field], 'lower_bounds': [],
              'upper_bounds': []}

    lat_vals = zmsf[lat_field].values
    zmsf_vals = zmsf.values

    if zmsf_vals.ndim != 2:
        raise ValueError('data must be provided two dimensional')

    n_levels = zmsf[level_field].shape[0]

    level_axis = zmsf.get_axis_num(level_field)
    lat_axis = zmsf.get_axis_num(lat_field)

    mask = _get_zero_crossings_mask(zmsf_vals, axis=lat_axis)

    reordered_mask = np.moveaxis(mask, level_axis, 0)
    reordered_vals = np.moveaxis(zmsf_vals, level_axis, 0)
    for i in range(n_levels):
        lat_mask = reordered_mask[i]
        vals = reordered_vals[i]
        n_crossings = np.sum(lat_mask)

        upper_lat_bounds = lat_vals[lat_mask]
        lower_lat_bounds = lat_vals[np.roll(lat_mask, -1)]

        upper_zmsf_bounds = vals[lat_mask]
        lower_zmsf_bounds = vals[np.roll(lat_mask, -1)]

        bounds['lower_bounds'].append(
            np.hstack([np.reshape(lower_lat_bounds, (n_crossings, 1)),
                       np.reshape(lower_zmsf_bounds, (n_crossings, 1))]))
        bounds['upper_bounds'].append(
            np.hstack([np.reshape(upper_lat_bounds, (n_crossings, 1)),
                       np.reshape(upper_zmsf_bounds, (n_crossings, 1))]))

    return bounds


def _check_consistent_bounds(bounds, level_field=DEFAULT_LEVEL_FIELD):
    n_levels = len(bounds[level_field])
    for i in range(n_levels):
        lb = bounds['lower_bounds'][i]
        ub = bounds['upper_bounds'][i]

        n_lb = lb.shape[0]
        n_ub = ub.shape[0]

        if n_lb != n_ub:
            raise ValueError('numbers of lower and upper bounds do not match')


def _interpolate_zero_intervals(bounds, level_field=DEFAULT_LEVEL_FIELD,
                                lat_field=DEFAULT_LAT_FIELD):
    _check_consistent_bounds(bounds)

    zero_crossings = {level_field: bounds[level_field],
                      lat_field: []}
    n_levels = bounds[level_field].shape[0]

    for i in range(n_levels):
        lb = bounds['lower_bounds'][i]
        ub = bounds['upper_bounds'][i]
        n_crossings = lb.shape[0]
        lats = np.empty((n_crossings,))
        for j in range(n_crossings):
            lz = lb[j, 1]
            uz = ub[j, 1]
            if lz == uz:
                lats[j] = lb[j, 0]
            else:
                t = lz / (lz - uz)
                lats[j] = lb[j, 0] * (1 - t) + ub[j, 0] * t
        zero_crossings[lat_field].append(lats)

    return zero_crossings


def _get_nearest_endpoint(bounds, level_field=DEFAULT_LEVEL_FIELD,
                          lat_field=DEFAULT_LAT_FIELD):
    _check_consistent_bounds(bounds)

    zero_crossings = {level_field: bounds[level_field],
                      lat_field: []}
    n_levels = bounds[level_field].shape[0]

    for i in range(n_levels):
        lb = bounds['lower_bounds'][i]
        ub = bounds['upper_bounds'][i]
        n_crossings = lb.shape[0]
        lats = np.empty((n_crossings,))
        for j in range(n_crossings):
            lz = np.abs(lb[j, 1])
            uz = np.abs(ub[j, 1])
            if lz < uz:
                lats[j] = lb[j, 0]
            else:
                lats[j] = ub[j, 0]
        zero_crossings[lat_field].append(lats)

    return zero_crossings


def _find_zero_crossings_at_time(single_time_zmsf,
                                 level_field=DEFAULT_LEVEL_FIELD,
                                 lat_field=DEFAULT_LAT_FIELD,
                                 interpolate=True):
    bounds = _bound_zero_crossings(
        single_time_zmsf, level_field=level_field, lat_field=lat_field)
    if interpolate:
        return _interpolate_zero_intervals(
            bounds, level_field=level_field, lat_field=lat_field)
    else:
        return _get_nearest_endpoint(
            bounds, level_field=level_field, lat_field=lat_field)


def find_zero_crossings(zmsf, time_field=DEFAULT_TIME_FIELD,
                        lat_field=DEFAULT_LAT_FIELD,
                        level_field=DEFAULT_LEVEL_FIELD,
                        interpolate=True):
    time_vals = zmsf[time_field]
    level_vals = zmsf[level_field].values

    crossing_data = []
    for t in time_vals:
        zero_crossings = _find_zero_crossings_at_time(
            zmsf.sel({time_field: t}), level_field=level_field,
            lat_field=lat_field, interpolate=interpolate)
        crossing_data.append(zero_crossings)

    return {time_field: time_vals, 'zero_crossings': crossing_data}


def calculate_zmsf(zm_vwnd_da, lat_field=DEFAULT_LAT_FIELD,
                   level_field=DEFAULT_LEVEL_FIELD):
    level_vals = zm_vwnd_da[level_field]
    lat_vals = zm_vwnd_da[lat_field]

    zmsf_da = xr.zeros_like(zm_vwnd_da)

    prefactor = (2.0 * pi * EARTH_RADIUS * np.cos(np.deg2rad(lat_vals)) /
                 GRAV_ACCEL)
    for p in zm_vwnd_da[level_field]:
        integrand = zm_vwnd_da.where(zm_vwnd_da[level_field] >= p, drop=True)
        zmsf_da.loc[{level_field: p}] = (integrand.integrate(level_field) *
            prefactor)

    return zmsf_da
