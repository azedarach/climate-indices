"""Test default helper functions."""

# License: MIT

import numpy as np
import pytest
import xarray as xr

import climate_indices.utils as cu


def test_get_coordinate_standard_name():
    """Test getting coordinate standard name."""
    coord = 'depth'
    coord_name = 'depth'
    x = xr.DataArray(
        np.random.uniform(size=10),
        coords={coord_name: np.arange(10)},
        dims=[coord_name])

    assert cu.get_coordinate_standard_name(x, coord) == coord_name

    coord = 'latitude'
    coord_name = 'lat'
    x = xr.DataArray(
        np.random.uniform(size=(10, 20, 10)),
        coords={coord_name: np.arange(10), 'lon': np.arange(20),
                'depth': np.arange(10)},
        dims=[coord_name, 'lon', 'depth'])

    assert cu.get_coordinate_standard_name(x, coord) == coord_name

    coord = 'longitude'
    coord_name = 'g0_lon_3'
    x = xr.DataArray(
        np.random.uniform(size=(10, 20)),
        coords={coord_name: np.arange(10), 'lat': np.arange(20)},
        dims=[coord_name, 'lat'])

    assert cu.get_coordinate_standard_name(x, coord) == coord_name

    coord = 'level'
    coord_name = 'level'
    x = xr.DataArray(
        np.random.uniform(size=(10, 20, 10)),
        coords={coord_name: np.arange(10), 'lon': np.arange(20),
                'lat': np.arange(10)},
        dims=[coord_name, 'lon', 'lat'])

    assert cu.get_coordinate_standard_name(x, coord) == coord_name

    coord = 'time'
    coord_name = 'initial_time0_hours'
    x = xr.DataArray(
        np.random.uniform(size=(20, 10)),
        coords={coord_name: np.arange(20),
                'depth': np.arange(10)},
        dims=[coord_name, 'depth'])

    assert cu.get_coordinate_standard_name(x, coord) == coord_name

    with pytest.raises(NotImplementedError):
        cu.get_coordinate_standard_name(x, 'unknown_coordinate')


def test_get_depth_name():
    """Test getting depth standard name."""
    expected_valid_names = ['depth']

    for depth_name in expected_valid_names:

        x = xr.DataArray(
            np.random.uniform(size=(10, 20, 10)),
            coords={depth_name: np.arange(10), 'lat': np.arange(20),
                    'lon': np.arange(10)},
            dims=[depth_name, 'lat', 'lon'])

        assert cu.get_depth_name(x) == depth_name

    with pytest.raises(ValueError):
        x = xr.DataArray(
            np.random.uniform(size=(10, 10)),
            coords={'lon': np.arange(10), 'lat': np.arange(10)},
            dims=['lon', 'lat'])

        cu.get_depth_name(x)


def test_get_lat_name():
    """Test getting latitude standard name."""
    expected_valid_names = ['lat', 'latitude', 'g0_lat_1', 'g0_lat_2',
                            'lat_2', 'yt_ocean']

    for lat_name in expected_valid_names:

        x = xr.DataArray(
            np.random.uniform(size=(10, 10)),
            coords={lat_name: np.arange(10), 'lon': np.arange(10)},
            dims=[lat_name, 'lon'])

        assert cu.get_lat_name(x) == lat_name

    with pytest.raises(ValueError):
        x = xr.DataArray(
            np.random.uniform(size=(10, 10)),
            coords={'lon': np.arange(10), 'depth': np.arange(10)},
            dims=['lon', 'depth'])

        cu.get_lat_name(x)

    for lat_name in expected_valid_names:
        for other_lat_name in expected_valid_names:

            if lat_name == other_lat_name:
                continue

            with pytest.raises(ValueError):
                x = xr.DataArray(
                    np.random.uniform(size=(10, 15, 20)),
                    coords={lat_name: np.arange(10),
                            other_lat_name: np.arange(15),
                            'level': np.arange(20)},
                    dims=[lat_name, other_lat_name, 'level'])

                cu.get_lat_name(x)


def test_get_lon_name():
    """Test getting longitude standard name."""
    expected_valid_names = ['lon', 'longitude', 'g0_lon_2', 'g0_lon_3',
                            'lon_2', 'xt_ocean']

    for lon_name in expected_valid_names:

        x = xr.DataArray(
            np.random.uniform(size=(10, 10)),
            coords={'lat': np.arange(10), lon_name: np.arange(10)},
            dims=['lat', lon_name])

        assert cu.get_lon_name(x) == lon_name

    with pytest.raises(ValueError):
        x = xr.DataArray(
            np.random.uniform(size=(10, 10)),
            coords={'lat': np.arange(10), 'depth': np.arange(10)},
            dims=['lat', 'depth'])

        cu.get_lon_name(x)

    for lon_name in expected_valid_names:
        for other_lon_name in expected_valid_names:

            if lon_name == other_lon_name:
                continue

            with pytest.raises(ValueError):
                x = xr.DataArray(
                    np.random.uniform(size=(10, 15, 20)),
                    coords={lon_name: np.arange(10),
                            other_lon_name: np.arange(15),
                            'level': np.arange(20)},
                    dims=[lon_name, other_lon_name, 'level'])

                cu.get_lon_name(x)


def test_get_time_name():
    """Test getting time standard name."""
    expected_valid_names = ['time', 'initial_time0_hours']

    for time_name in expected_valid_names:

        x = xr.DataArray(
            np.random.uniform(size=(30, 10, 10)),
            coords={time_name: np.arange(30),
                    'lat': np.arange(10), 'lon': np.arange(10)},
            dims=[time_name, 'lat', 'lon'])

        assert cu.get_time_name(x) == time_name

    with pytest.raises(ValueError):
        x = xr.DataArray(
            np.random.uniform(size=(10, 10)),
            coords={'lat': np.arange(10), 'lon': np.arange(10)},
            dims=['lat', 'lon'])

        cu.get_time_name(x)

    for time_name in expected_valid_names:
        for other_time_name in expected_valid_names:

            if time_name == other_time_name:
                continue

            with pytest.raises(ValueError):
                x = xr.DataArray(
                    np.random.uniform(size=(10, 15, 20)),
                    coords={time_name: np.arange(10),
                            other_time_name: np.arange(15),
                            'level': np.arange(20)},
                    dims=[time_name, other_time_name, 'level'])

                cu.get_time_name(x)
