"""Provides routines for providing defaults."""

# License: MIT

from typing import Any, List


_KNOWN_COORDINATE_NAMES = {
    'depth': ['depth'],
    'latitude': ['lat', 'latitude', 'g0_lat_1', 'g0_lat_2',
                 'lat_2', 'yt_ocean'],
    'level': ['level'],
    'longitude': ['lon', 'longitude', 'g0_lon_2', 'g0_lon_3',
                  'lon_2', 'xt_ocean'],
    'time': ['time', 'initial_time0_hours']
}


def _get_coordinate_candidate_names(coord: str) -> List[str]:
    """Return candidate standard names for coordinate.

    Parameters
    ----------
    coord : str
        Coordinate to get candidate names for.

    Returns
    -------
    candidates : list
        List of candidate standard names.
    """
    for category in _KNOWN_COORDINATE_NAMES:
        if coord in _KNOWN_COORDINATE_NAMES[category]:
            return _KNOWN_COORDINATE_NAMES[category].copy()

    raise NotImplementedError(
        "Candidate standard names for '%s' not implemented" % coord)


def get_coordinate_standard_name(obj: Any, coord: str) -> str:
    """Return standard name for coordinate.

    Parameters
    ----------
    obj : object
        Object to search for coordinate name in. Must have
        a 'dims' attribute.

    coord : str
        Coordinate to search for.

    Returns
    -------
    name : str
        Standard name of the coordinate if found.
    """
    candidates = _get_coordinate_candidate_names(coord)

    matches = [c for c in candidates if c in obj.dims]

    if len(matches) > 1:
        raise ValueError(
            "Found multiple possible matches for coordinate '%s' in "
            "object dimensions: %r" % tuple(matches))

    if len(matches) == 1:
        return matches[0]

    raise ValueError(
        "Unable to find coordinate '%s'" % coord)


def get_depth_name(obj: Any) -> str:
    """Return name of depth coordinate.

    Parameters
    ----------
    obj : object
        Object to search for coordinate name in.

    Returns
    -------
    name : str
        Standard name of depth coordinate, if found.
    """
    return get_coordinate_standard_name(obj, 'depth')


def get_lat_name(obj: Any) -> str:
    """Return name of latitude coordinate.

    Parameters
    ----------
    obj : object
        Object to search for coordinate name in.

    Returns
    -------
    name : str
        Standard name of latitude coordinate, if found.
    """
    return get_coordinate_standard_name(obj, 'latitude')


def get_lon_name(obj: Any) -> str:
    """Return name of longitude coordinate.

    Parameters
    ----------
    obj : object
        Object to search for coordinate name in.

    Returns
    -------
    name : str
        Standard name of longitude coordinate, if found.
    """
    return get_coordinate_standard_name(obj, 'longitude')


def get_time_name(obj: Any) -> str:
    """Return name of time coordinate.

    Parameters
    ----------
    obj : object
        Object to search for coordinate name in.

    Returns
    -------
    name : str
        Standard name of time coordinate, if found.
    """
    return get_coordinate_standard_name(obj, 'time')
