"""
Provides routines for calculating indices of the NAO.
"""


from __future__ import absolute_import, division

import numpy as np

from ..utils import (check_base_period, check_data_array,
                     detect_frequency, downsample_data,
                     get_coordinate_standard_name,
                     standardized_anomalies)



def hurrell_nao(mslp, frequency=None, standardize_by='month',
                interpolate=True, method='nearest',
                base_period=None, lat_name=None,
                lon_name=None, time_name=None):
    """Calculate Hurrell (station based) NAO index.

    The station based index used by Hurrell is defined as
    the difference between the standardized MSLP at
    Ponta Delgada, Azores, and Stykkisholmur/Reykjavik, Iceland.

    See Hurrell, J. W., "Decadal Trends in the North Atlantic
    Oscillation: Regional Temperatures and Precipitation",
    Science 269, 676 - 679 (1995), doi:10.1126/science.269.5224.676 .

    Parameters
    ----------
    mslp : xarray.DataArray
        Array containing MSLP values.

    frequency : str
        If given, sampling rate at which to calculate index.

    standardize_by : str
        Interval within which to standardise.

    interpolate : bool
        If True, interpolate given data to station
        locations. If False, use nearest gridpoint to each station.

    base_period : list
        If given, a two element list containing the
            earliest and latest dates to include when calculating the
            climatology used for standardization. If None, the
            climatology is formed from the full dataset.

    lat_name : str
        Name of the latitude coordinate.

    lon_name : str
        Name of the longitude coordinate.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    index : xarray.DataArray
        Array containing the values of the index.
    """

    # Ensure that the data provided is a data array
    mslp = check_data_array(mslp)

    # Get coordinate names
    if lat_name is None:
        lat_name = get_coordinate_standard_name(mslp, 'lat')

    if lon_name is None:
        lon_name = get_coordinate_standard_name(mslp, 'lon')

    if time_name is None:
        time_name = get_coordinate_standard_name(mslp, 'time')

    # Default Northern station (Stykkisholmur) for Hurrell NAO index.
    if np.any(mslp[lon_name] < 0):
        northern_station = {lat_name : 65.0, lon_name : -22.8}
    else:
        northern_station = {lat_name : 65.0, lon_name : 337.2}

    # Default Southern station (Ponta Delgada) for Hurrell NAO index.
    if np.any(mslp[lon_name] < 0):
        southern_station = {lat_name : 37.7, lon_name : -25.7}
    else:
        southern_station = {lat_name : 37.7, lon_name : 334.3}

    if interpolate:
        northern_station_data = mslp.interp(northern_station, method=method)
        southern_station_data = mslp.interp(southern_station, method=method)
    else:
        northern_station_data = mslp.sel(northern_station, method=method)
        southern_station_data = mslp.sel(southern_station, method=method)

    # If required, resample the data to a different time frequency.
    if frequency is None:
        frequency = 'monthly'

    if frequency not in ('daily', 'monthly'):
        raise ValueError("Unsupported frequency '%r'" % frequency)

    input_frequency = detect_frequency(mslp, time_name=time_name)

    if input_frequency == 'daily' and frequency == 'monthly':

        northern_station_data = downsample_data(
            northern_station_data, frequency='monthly', time_name=time_name)

        southern_station_data = downsample_data(
            southern_station_data, frequency='monthly', time_name=time_name)

    elif input_frequency == 'monthly' and frequency == 'daily':

        raise ValueError('Cannot calculate daily index from monthly data')

    # Calculate standardized anomalies of MSLP.
    stand_mslp_northern = standardized_anomalies(
        northern_station_data, base_period=base_period,
        standardize_by=standardize_by, time_name=time_name)
    stand_mslp_southern = standardized_anomalies(
        southern_station_data, base_period=base_period,
        standardize_by=standardize_by, time_name=time_name)

    return (stand_mslp_northern - stand_mslp_southern).rename('nao_index')
