"""Provides routines for EOF analysis."""

# License: MIT

import dask
import dask.array as da
import numpy as np
import scipy.linalg as sl
import scipy.sparse.linalg as spl


def _fix_svd_phases(u, vh):
    """Impose fixed phase convention on left- and right-singular vectors.

    Given a set of left- and right-singular vectors as the columns of u
    and rows of vh, respectively, imposes the phase convention that for
    each left-singular vector, the element with largest absolute value
    is real and positive.

    Parameters
    ----------
    u : array, shape (..., M, K)
        Unitary array containing the left-singular vectors as columns.

    vh : array, shape (..., K, N)
        Unitary array containing the right-singular vectors as rows.

    Returns
    -------
    u_fixed : array, shape (..., M, K)
        Unitary array containing the left-singular vectors as columns,
        conforming to the chosen phase convention.

    vh_fixed : array, shape (..., K, N)
        Unitary array containing the right-singular vectors as rows,
        conforming to the chosen phase convention.
    """

    n_cols = u.shape[-1]

    max_elem_rows = np.argmax(np.abs(u), axis=-2)
    max_elems = np.take_along_axis(u, max_elem_rows[..., None, :], axis=-2)

    if np.any(np.iscomplexobj(u)):
        phases = np.exp(-1j * np.angle(max_elems))
    else:
        phases = np.sign(max_elems)

    u *= phases
    vh *= np.conj(phases.swapaxes(-1,-2))

    return u, vh


def _calc_truncated_svd_2d(X, k, coerce_signs=True, **kwargs):
    """Calculate the truncated SVD of a matrix."""
    max_rank = min(X.shape)

    if k < max_rank:
        u, s, vh = spl.svds(X, k=k, **kwargs)

        # Note that svds returns the singular values with the
        # opposite (i.e., non-decreasing) ordering convention.
        u = u[:, ::-1]
        s = s[::-1]
        vh = vh[::-1]
    else:
        u, s, vh = sl.svd(X, full_matrices=False)

    if coerce_signs:
        # Impose a fixed phase convention on the singular vectors
        # to avoid phase ambiguity.
        u, vh = _fix_svd_phases(u, vh)

    return u, s, vh


def _calc_truncated_svd_default(X, k, coerce_signs=True, **kwargs):
    """Calculate the truncated SVD of an array."""

    if X.ndim < 2:
        raise ValueError(
            'Input array must be at least 2-dimensional '
            '(got X.ndim=%d)' % X.ndim)

    if X.ndim == 2:
        return _calc_truncated_svd_2d(
            X, k, coerce_signs=coerce_signs, **kwargs)

    m, n = X.shape[-2:]
    max_rank = min(m, n)
    K = min(k, max_rank)

    u = np.empty(X.shape[:-2] + (m, K))
    s = np.empty(X.shape[:-2] + (K,))
    vh = np.empty(X.shape[:-2] + (K, n))

    it = np.ndindex(X.shape[:-2])
    for idx in it:
        mat_index = idx + np.index_exp[:, :]
        vec_index = idx + np.index_exp[:]
        u[mat_index], s[vec_index], vh[mat_index] = \
            _calc_truncated_svd_2d(
                X[idx], K, coerce_signs=coerce_signs, **kwargs)

    return u, s, vh


def _calc_truncated_svd_dask(X, k, coerce_signs=True, compressed=True,
                             allow_rechunk=False, **kwargs):
    """Calculate the truncated SVD of a dask array."""

    if X.ndim < 2:
        raise ValueError(
            'Input array must be at least 2-dimensional '
            '(got X.ndim=%d)' % X.ndim)

    m, n = X.shape[-2:]
    max_rank = min(m, n)
    K = min(k, max_rank)

    if X.ndim == 2:
        if compressed:
            return da.linalg.svd_compressed(
                X, k, coerce_signs=coerce_signs, **kwargs)

        u, s, vh = da.linalg.svd(X, coerce_signs=coerce_signs)

        if K < max_rank:
            u = u[:, :K]
            s = s[:K]
            vh = vh[:K, :]

        return u, s, vh

    def _svd(a):
        return _calc_truncated_svd_default(
            a, k, coerce_signs=coerce_signs, **kwargs)

    gusvd = da.gufunc(
        _svd, signature='(m,n)->(m,k),(k),(k,n)',
        output_dtypes=(X.dtype, X.dtype, X.dtype),
        output_sizes={'k': K, 'm': m, 'n': n},
        allow_rechunk=allow_rechunk)

    return gusvd(X)


def calc_truncated_svd(X, k, coerce_signs=True, compressed=True, **kwargs):
    """Calculate the truncated SVD of an array.

    Given an array X with shape (..., M, N), the SVD of X is computed and the
    leading K = min(k, min(M, N)) singular values are retained.

    Parameters
    ----------
    X : array, shape (..., M, N)
        The array to calculate the SVD of.

    k : integer
        Number of singular values to retain in truncated decomposition.
        If k > min(M, N), then all singular values are retained.

    coerce_signs : bool, default: True
        If True, apply sign coercion to singular vectors to maintain
        deterministic results.

    compressed : bool, default: True
        If True and given dask array input, compute randomized SVD.

    kwargs : dict
        Additional keyword arguments to pass to the underlying SVD
        calculation.

    Returns
    -------
    u : array, shape (..., M, K)
        Unitary array containing the retained left-singular vectors of X
        as columns.

    s : array, shape (..., K)
        Array containing the leading K singular vectors of X.

    vh : array, shape (..., K, N)
        Unitary array containing the retained right-singular vectors of
        X as rows.
    """

    if  dask.is_dask_collection(X):
        return _calc_truncated_svd_dask(
            X, k, coerce_signs=coerce_signs,
            compressed=compressed, **kwargs)

    return _calc_truncated_svd_default(
        X, k, coerce_signs=coerce_signs, **kwargs)
