"""Provides unit tests for EOFs."""

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
from climate_indices.utils.eofs import _fix_svd_phases


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


def test_fix_svd_phases_real_numpy_input():
    """Test fixing SVD phases for real matrices."""

    u = np.array([[-2.0, 0.0, 1.0],
                  [1.0, 1.0, 2.0],
                  [0.0, -2.0, 1.0],
                  [-0.5, -0.5, 0.0]])
    s = np.array([2.0, 1.0, 0.5])
    vh = np.array([[2.0, 3.0, 4.0],
                   [1.0, -1.0, 0.0],
                   [-1.0, -1.0, 0.5]])
    x = np.matmul((u * s[np.newaxis, :]), vh)

    u_fixed, vh_fixed = _fix_svd_phases(u, vh)
    x_fixed = np.matmul((u_fixed * s[np.newaxis, :]), vh_fixed)

    expected_u = np.array([[2.0, 0.0, 1.0],
                           [-1.0, -1.0, 2.0],
                           [0.0, 2.0, 1.0],
                           [0.5, 0.5, 0.0]])
    expected_vh = np.array([[-2.0, -3.0, -4.0],
                            [-1.0, 1.0, 0.0],
                            [-1.0, -1.0, 0.5]])

    assert np.allclose(u_fixed, expected_u)
    assert np.allclose(vh_fixed, expected_vh)
    assert np.allclose(x, x_fixed)

    u = np.array([[[0.5, 1.0],
                   [3.0, -2.0],
                   [2.0, -0.3]],
                  [[2.0, -2.0],
                   [-4.0, 0.5],
                   [2.0, -0.5]]])
    s = np.array([[3.0, 2.0], [0.5, 0.1]])
    vh = np.array([[[1.0, 2.0, -2.0],
                    [-0.5, -1.0, 3.0]],
                   [[-0.4, 3.0, -2.0],
                    [3.0, 2.0, -3.0]]])

    x = np.matmul((u * s[..., np.newaxis, :]), vh)

    assert x.shape == (2, 3, 3)

    u_fixed, vh_fixed = _fix_svd_phases(u, vh)
    x_fixed = np.matmul((u_fixed * s[..., np.newaxis, :]), vh_fixed)

    expected_u = np.array([[[0.5, -1.0],
                            [3.0, 2.0],
                            [2.0, 0.3]],
                           [[-2.0, 2.0],
                            [4.0, -0.5],
                            [-2.0, 0.5]]])
    expected_vh = np.array([[[1.0, 2.0, -2.0],
                             [0.5, 1.0, -3.0]],
                            [[0.4, -3.0, 2.0],
                             [-3.0, -2.0, 3.0]]])

    assert np.allclose(u_fixed, expected_u)
    assert np.allclose(vh_fixed, expected_vh)
    assert np.allclose(x, x_fixed)


def test_fix_svd_phases_complex_numpy_input():
    """Test fixing SVD phases for complex matrices."""

    u = np.array([[2.5 * np.exp(-1j * np.pi / 2.0),
                   np.sqrt(2) * np.exp(1j * np.pi / 2.0)],
                  [np.sqrt(3) * np.exp(1j * np.pi / 3.0), 5.0],
                  [0.0, np.sqrt(3) * np.exp(-1j * np.pi / 6.0)]])
    s = np.array([5.0, 3.4])
    vh = np.array([[2.0, 3.0 * np.exp(1j * np.pi / 2.0)],
                   [np.exp(2.j * np.pi / 3.0), -1.0]])
    x = np.matmul((u * s[np.newaxis, :]), vh)

    u_fixed, vh_fixed = _fix_svd_phases(u, vh)
    x_fixed = np.matmul((u_fixed * s[np.newaxis, :]), vh_fixed)

    expected_u = np.array([[2.5, np.sqrt(2) * np.exp(1j * np.pi / 2.0)],
                           [np.sqrt(3) * np.exp(5.j * np.pi / 6.0), 5.0],
                           [0.0, np.sqrt(3) * np.exp(-1j * np.pi / 6.0)]])
    expected_vh = np.array([[2.0 * np.exp(-1j * np.pi / 2.0), 3.0],
                            [np.exp(2.j * np.pi / 3.0), -1.0]])

    assert np.allclose(u_fixed, expected_u)
    assert np.allclose(vh_fixed, expected_vh)
    assert np.allclose(x, x_fixed)

    u = np.array([[[0.5 * np.exp(1j * np.pi / 2.0), 1.0],
                   [3.0 * np.exp(1j * np.pi / 4.0),
                    2.0 * np.exp(1j * np.pi / 2.0)],
                   [2.0 * np.exp(1j * np.pi / 8.0), -0.3]],
                  [[2.0 * np.exp(-1j * np.pi / 4.0),
                    2.0 * np.exp(1j * np.pi / 2.0)],
                   [-4.0, 0.5],
                   [2.0 * np.exp(1j * np.pi / 3.0), -0.5]]])
    s = np.array([[3.0, 2.0], [0.5, 0.1]])
    vh = np.array([[[1.0, 2.0, -2.0],
                    [0.5 * np.exp(1j * np.pi / 3.0), -1.0,
                     3.0 * np.exp(1j * np.pi / 3.0)]],
                   [[-0.4, 3.0 * np.exp(1j * 2.0 * np.pi / 3.0), -2.0],
                    [3.0 * np.exp(1j * 4.0 * np.pi / 3.0),
                     2.0, 3.0 * np.exp(-1j * np.pi / 5.0)]]])

    x = np.matmul((u * s[..., np.newaxis, :]), vh)

    assert x.shape == (2, 3, 3)

    u_fixed, vh_fixed = _fix_svd_phases(u, vh)
    x_fixed = np.matmul((u_fixed * s[..., np.newaxis, :]), vh_fixed)

    expected_u = np.array([[[0.5 * np.exp(1j * np.pi / 4.0),
                             1.0 * np.exp(-1j * np.pi / 2.0)],
                            [3.0, 2.0],
                            [2.0 * np.exp(-1j * np.pi / 8.0),
                             0.3 * np.exp(1j * np.pi / 2.0)]],
                           [[2.0 * np.exp(1j * 3.0 * np.pi / 4.0), 2.0],
                            [4.0, 0.5 * np.exp(-1j * np.pi / 2.0)],
                            [2.0 * np.exp(1j * 4.0 * np.pi / 3.0),
                             0.5  * np.exp(1j * np.pi / 2.0)]]])
    expected_vh = np.array([[[1.0 * np.exp(1j * np.pi / 4.0),
                              2.0 * np.exp(1j * np.pi / 4.0),
                              2.0 * np.exp(1j * 5.0 * np.pi / 4.0)],
                             [0.5 * np.exp(1j * 5.0 * np.pi / 6.0),
                              1.0 * np.exp(1j * 3.0 * np.pi / 2.0),
                              3.0 * np.exp(1j * 5.0 * np.pi / 6.0)]],
                            [[0.4, 3.0 * np.exp(1j * 5.0 * np.pi / 3.0), 2.0],
                             [3.0 * np.exp(1j * 11.0 * np.pi / 6.0),
                              2.0 * np.exp(1j * np.pi / 2.0),
                              3.0 * np.exp(1j * 3.0 * np.pi / 10.0)]]])

    assert np.allclose(u_fixed, expected_u)
    assert np.allclose(vh_fixed, expected_vh)
    assert np.allclose(x, x_fixed)


@given(x=hn.arrays(
        hn.floating_dtypes(sizes=(64,)),
        hn.array_shapes(max_dims=2),
        elements=dict(min_value=-1e10, max_value=1e10)))
def test_calc_truncated_svd_2d_numpy_input(x):
    """Test truncated SVD calculation with numpy input."""
    if x.ndim < 2:
        with pytest.raises(ValueError):
            u, s, vh = cu.calc_truncated_svd(x, k=1)
    else:
        max_rank = min(x.shape[-2:])
        for k in range(1, max_rank + 1):
            u, s, vh = cu.calc_truncated_svd(x, k, coerce_signs=False)

            assert u.shape == (x.shape[-2], k)
            assert s.shape == (k,)
            assert vh.shape == (k, x.shape[-1])

            u, s, vh = cu.calc_truncated_svd(x, k, coerce_signs=True)

            max_elem_rows = np.argmax(np.abs(u), axis=0)
            for j, i in enumerate(max_elem_rows):
                assert np.real(u[i, j]) >= 0.0

            assert u.shape == (x.shape[-2], k)
            assert s.shape == (k,)
            assert vh.shape == (k, x.shape[-1])

        x_svd = np.matmul((u * s[..., None, :]), vh)

        assert x.shape == x_svd.shape

        assert np.allclose(x, x_svd, atol=1e-7)


@given(x=hn.arrays(
        hn.floating_dtypes(sizes=(64,)),
        hn.array_shapes(min_dims=2),
        elements=dict(min_value=-1e10, max_value=1e10)))
def test_calc_truncated_svd_nd_numpy_input(x):
    """Test truncated SVD calculation with numpy input."""
    max_rank = min(x.shape[-2:])

    for k in range(1, max_rank + 1):

        u, s, vh = cu.calc_truncated_svd(x, k, coerce_signs=False)

        assert u.shape == x.shape[:-2] + (x.shape[-2], k)
        assert s.shape == x.shape[:-2] + (k,)
        assert vh.shape == x.shape[:-2] + (k, x.shape[-1])

        u, s, vh = cu.calc_truncated_svd(x, k, coerce_signs=True)

        assert u.shape == x.shape[:-2] + (x.shape[-2], k)
        assert s.shape == x.shape[:-2] + (k,)
        assert vh.shape == x.shape[:-2] + (k, x.shape[-1])

        for idx in np.ndindex(x.shape[:-2]):
            u_i = u[idx]
            max_elem_rows = np.argmax(np.abs(u_i), axis=0)
            for j, i in enumerate(max_elem_rows):
                assert np.real(u_i[i, j]) >= 0.0

    x_svd = np.matmul((u * s[..., None, :]), vh)

    assert x.shape == x_svd.shape

    assert np.allclose(x, x_svd)


@given(x=hn.arrays(
        hn.floating_dtypes(sizes=(64,)),
        hn.array_shapes(max_dims=2),
        elements=dict(min_value=-1e10, max_value=1e10)))
def test_calc_truncated_svd_2d_dask_input(x):
    """Test truncated SVD calculation with numpy input."""
    x = da.from_array(x)
    if x.ndim < 2:
        with pytest.raises(ValueError):
            u, s, vh = cu.calc_truncated_svd(x, k=1)
    else:
        max_rank = min(x.shape[-2:])

        for k in range(1, max_rank + 1):
            scheduler = CountingScheduler(max_computes=0)
            with dask.config.set(scheduler=scheduler):

                u, s, vh = cu.calc_truncated_svd(x, k, coerce_signs=False)

                assert u.shape == (x.shape[-2], k)
                assert s.shape == (k,)
                assert vh.shape == (k, x.shape[-1])

                u, s, vh = cu.calc_truncated_svd(x, k, coerce_signs=True)

                assert u.shape == (x.shape[-2], k)
                assert s.shape == (k,)
                assert vh.shape == (k, x.shape[-1])

        x_svd = da.matmul((u * s[..., None, :]), vh)

        assert x.shape == x_svd.shape

        assert np.allclose(x, x_svd, atol=1e-7).compute()


@given(x=hn.arrays(
        hn.floating_dtypes(sizes=(64,)),
        hn.array_shapes(min_dims=2),
        elements=dict(min_value=-1e10, max_value=1e10)))
def test_calc_truncated_svd_nd_dask_input(x):
    """Test truncated SVD calculation with dask input."""
    x = da.from_array(x)
    max_rank = min(x.shape[-2:])

    for k in range(1, max_rank + 1):

        scheduler = CountingScheduler(max_computes=0)
        with dask.config.set(scheduler=scheduler):

            u, s, vh = cu.calc_truncated_svd(x, k, coerce_signs=False)

            assert u.shape == x.shape[:-2] + (x.shape[-2], k)
            assert s.shape == x.shape[:-2] + (k,)
            assert vh.shape == x.shape[:-2] + (k, x.shape[-1])

            u, s, vh = cu.calc_truncated_svd(
                x, k, coerce_signs=False, compressed=False)

            assert u.shape == x.shape[:-2] + (x.shape[-2], k)
            assert s.shape == x.shape[:-2] + (k,)
            assert vh.shape == x.shape[:-2] + (k, x.shape[-1])

            u, s, vh = cu.calc_truncated_svd(x, k, coerce_signs=True)

            assert u.shape == x.shape[:-2] + (x.shape[-2], k)
            assert s.shape == x.shape[:-2] + (k,)
            assert vh.shape == x.shape[:-2] + (k, x.shape[-1])

    x_svd = da.matmul((u * s[..., None, :]), vh)

    assert x.shape == x_svd.shape

    assert np.allclose(x, x_svd).compute()
