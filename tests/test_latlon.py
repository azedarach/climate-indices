"""Provides unit tests for latitude-longitude helper routines."""

# License: MIT

import dask
import dask.array as da
import hypothesis.extra.numpy as hn
import hypothesis.strategies as st
import numpy as np
import pytest
import xarray as xr
from hypothesis import given

import climate_indices.utils as cu


class CountingScheduler:
    """Simple dask scheduler counting the number of computes.

    See https://stackoverflow.com/questions/53289286/
    """

    def __init__(self, max_computes=0):
        """Initialize counting scheduler."""
        self.total_computes = 0
        self.max_computes = max_computes

    def __call__(self, dsk, keys, **kwargs):
        """Call counting scheduler."""
        self.total_computes += 1
        if self.total_computes > self.max_computes:
            raise RuntimeError(
                "Too many computes. Total: %d > max: %d."
                % (self.total_computes, self.max_computes)
            )
        return dask.get(dsk, keys, **kwargs)


@given(x=st.floats(min_value=-10000000, max_value=10000000),
       base=st.floats(min_value=-10000000, max_value=10000000),
       period=st.floats(min_value=-10000000, max_value=10000000))
def test_wrap_coordinate_scalar_input(x, base, period):
    """Test wrapping coordinates with scalar input."""
    if period == 0.0:
        with pytest.raises(ValueError):
            cu.wrap_coordinate(x, base=base, period=period)
    else:
        if period >= 0.0:
            lower_bound = base
            upper_bound = base + period
        else:
            lower_bound = base + period
            upper_bound = base

        wrapped = cu.wrap_coordinate(x, base=base, period=period)

        assert lower_bound <= wrapped <= upper_bound


@given(x=st.one_of(
        st.lists(st.floats(min_value=-10000000, max_value=10000000)),
        st.tuples(st.floats(min_value=-10000000, max_value=10000000))),
       base=st.floats(min_value=-10000000, max_value=10000000),
       period=st.floats(min_value=-10000000, max_value=10000000))
def test_wrap_coordinate_sequence_input(x, base, period):
    """Test wrapping coordinates with list input."""
    if period == 0.0:
        with pytest.raises(ValueError):
            cu.wrap_coordinate(x, base=base, period=period)
    else:
        if period >= 0.0:
            lower_bound = base
            upper_bound = base + period
        else:
            lower_bound = base + period
            upper_bound = base

        wrapped = cu.wrap_coordinate(x, base=base, period=period)

        assert all(lower_bound <= xi <= upper_bound for xi in wrapped)


@given(x=hn.arrays(
        hn.floating_dtypes(sizes=(32, 64)), hn.array_shapes(),
        elements=dict(min_value=-10000000, max_value=10000000)),
       base=st.floats(min_value=-10000000, max_value=10000000),
       period=st.floats(min_value=-10000000, max_value=10000000))
def test_wrap_coordinate_numpy_input(x, base, period):
    """Test wrapping coordinates with numpy input."""
    if period == 0.0:
        with pytest.raises(ValueError):
            cu.wrap_coordinate(x, base=base, period=period)
    else:
        if period >= 0.0:
            lower_bound = base
            upper_bound = base + period
        else:
            lower_bound = base + period
            upper_bound = base

        wrapped = cu.wrap_coordinate(x, base=base, period=period)

        assert np.all(np.logical_and(wrapped >= lower_bound,
                                     wrapped <= upper_bound))

        assert wrapped.dtype == x.dtype


@given(x=hn.arrays(
        hn.floating_dtypes(sizes=(32, 64)), hn.array_shapes(),
        elements=dict(min_value=-10000000, max_value=10000000)),
       base=st.floats(min_value=-10000000, max_value=10000000),
       period=st.floats(min_value=-10000000, max_value=10000000))
def test_wrap_coordinate_dask_input(x, base, period):
    """Test wrapping coordinates with dask input."""
    x = da.from_array(x)

    if period == 0.0:
        with pytest.raises(ValueError):
            cu.wrap_coordinate(x, base=base, period=period)
    else:
        if period >= 0.0:
            lower_bound = base
            upper_bound = base + period
        else:
            lower_bound = base + period
            upper_bound = base

        scheduler = CountingScheduler(max_computes=0)
        with dask.config.set(scheduler=scheduler):
            wrapped = cu.wrap_coordinate(x, base=base, period=period)

        assert np.all(np.logical_and(wrapped >= lower_bound,
                                     wrapped <= upper_bound)).compute()

        assert wrapped.dtype == x.dtype


def test_wrap_coordinate_values():
    """Test wrapped values correctly calculated."""
    x = np.array(
        [-210.0, -180.0, -150.0, -30.0, 0.0, 10.0, 90.0, 180.0, 360.0, 400.0])
    wrapped = cu.wrap_coordinate(x, base=0.0, period=360.0)
    expected = np.array(
        [150.0, 180.0, 210.0, 330.0, 0.0, 10.0, 90.0, 180.0, 0.0, 40.0])

    assert np.allclose(wrapped, expected)

    wrapped = cu.wrap_coordinate(x, base=-180.0, period=360.0)
    expected = np.array(
        [150.0, -180.0, -150.0, -30.0, 0.0, 10.0, 90.0, -180.0, 0.0, 40.0])

    assert np.allclose(wrapped, expected)


def test_select_latlon_box_latlon_dims():
    """Test selection of lat-lon box."""
    lat = np.array([-20.0, -10.0, 0.0, 10.0, 20.0])
    lon = np.array([0.0, 10.0, 20.0, 30.0, 40.0])

    x = xr.DataArray(np.zeros((5, 5)), coords={'lat': lat, 'lon': lon},
                     dims=['lat', 'lon'])

    with pytest.raises(ValueError):
        cu.select_latlon_box(
            x, lat_bounds=[-10, 10, 10], lon_bounds=[0.0, 20.0])

    with pytest.raises(ValueError):
        cu.select_latlon_box(
            x, lat_bounds=[-10, 10], lon_bounds=[0.0, 20.0, 0.0])

    box = cu.select_latlon_box(
        x, lat_bounds=[-10, 10], lon_bounds=[0.0, 20.0], drop=True)

    expected_lat = np.array([-10.0, 0.0, 10.0])
    expected_lon = np.array([0.0, 10.0, 20.0])

    assert np.allclose(box['lat'], expected_lat)
    assert np.allclose(box['lon'], expected_lon)

    lat = np.array([-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0])
    lon = np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0,
                    210.0, 240.0, 270.0, 300.0, 330.0, 359.0])

    x = xr.DataArray(np.zeros((7, 13)), coords={'lat': lat, 'lon': lon},
                     dims=['lat', 'lon'])

    box = cu.select_latlon_box(
        x, lat_bounds=[-50, 50], lon_bounds=[0.0, 100.0], drop=True)

    expected_lat = np.array([-30.0, 0.0, 30.0])
    expected_lon = np.array([0.0, 30.0, 60.0, 90.0])

    assert np.allclose(box['lat'], expected_lat)
    assert np.allclose(box['lon'], expected_lon)

    box = cu.select_latlon_box(
        x, lat_bounds=[0.0, 50.0], lon_bounds=[120.0, 200.0], drop=True)

    expected_lat = np.array([0.0, 30.0])
    expected_lon = np.array([120.0, 150.0, 180.0])

    assert np.allclose(box['lat'], expected_lat)
    assert np.allclose(box['lon'], expected_lon)

    box = cu.select_latlon_box(
        x, lat_bounds=[0.0, 50.0], lon_bounds=[300.0, 50.0], drop=True)

    expected_lat = np.array([0.0, 30.0])
    expected_lon = np.array([0.0, 30.0, 300.0, 330.0, 359.0])

    assert np.allclose(box['lat'], expected_lat)
    assert np.allclose(box['lon'], expected_lon)

    box = cu.select_latlon_box(
        x, lat_bounds=[-50.0, -10.0], lon_bounds=[-150.0, -5.0], drop=True)

    expected_lat = np.array([-30.0])
    expected_lon = np.array([210.0, 240.0, 270.0, 300.0, 330.0])

    assert np.allclose(box['lat'], expected_lat)
    assert np.allclose(box['lon'], expected_lon)

    box = cu.select_latlon_box(
        x, lat_bounds=[-50.0, -10.0], lon_bounds=[-150.0, 50.0], drop=True)

    expected_lat = np.array([-30.0])
    expected_lon = np.array(
        [0.0, 30.0, 210.0, 240.0, 270.0, 300.0, 330.0, 359.0])

    assert np.allclose(box['lat'], expected_lat)
    assert np.allclose(box['lon'], expected_lon)


def test_select_latlon_box_latlon_coords():
    """Test selection of lat-lon box."""
    x = np.arange(13)
    y = np.arange(7)

    lat = np.array([-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0])
    lon = np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0,
                    210.0, 240.0, 270.0, 300.0, 330.0, 359.0])

    lon, lat = np.meshgrid(lon, lat)

    assert lon.shape == (7, 13)
    assert lat.shape == (7, 13)

    data = xr.DataArray(
        np.random.uniform(size=(7, 13)),
        coords={'x': x, 'y': y}, dims=['y', 'x'])
    data = data.assign_coords({'geolat': (['y', 'x'], lat),
                               'geolon': (['y', 'x'], lon)})

    box = cu.select_latlon_box(
        data, lat_bounds=[-50.0, 50.0], lon_bounds=[40.0, 100.0], drop=True,
        lat_name='geolat', lon_name='geolon')

    expected_x = np.array([2, 3])
    expected_y = np.array([2, 3, 4])

    print(box['geolat'])

    assert np.allclose(box['x'], expected_x)
    assert np.allclose(box['y'], expected_y)
