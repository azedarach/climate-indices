"""
Provides default coordinate names and conventions.
"""


def get_coordinate_standard_name(obj, coord):
    """Return standard name for coordinate.

    Parameters
    ----------
    obj : xarray object
        Object to search for coordinate name in.

    coord : 'lat' | 'lon' | 'level' | 'time'
        Coordinate to search for.

    Returns
    -------
    name : str
        Standard name of coordinate if found
    """

    valid_keys = ['lat', 'lon', 'level', 'time']

    if coord not in valid_keys:
        raise ValueError("Unrecognized coordinate key '%r'" % coord)

    if coord == 'lat':

        candidates = ['lat', 'g0_lat_1', 'g0_lat_2', 'lat_2', 'yt_ocean']

    elif coord == 'lon':

        candidates = ['lon', 'g0_lon_2', 'g0_lon_3', 'lon_2', 'xt_ocean']

    elif coord == 'level':

        candidates = ['level']

    elif coord == 'time':

        candidates = ['time', 'initial_time0_hours']

    for c in candidates:
        if c in obj.dims:
            return c

    raise ValueError("Unable to find coordinate '%s'" % coord)
