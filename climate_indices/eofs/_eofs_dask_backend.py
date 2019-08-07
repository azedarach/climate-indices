
import dask.array as da
import numpy as np

import climate_indices.eofs._eofs_default_backend as default_backend


def _check_array_shape(A, shape, whom):
    if A.shape != shape:
        raise ValueError(
            'Array with wrong shape passed to %s. '
            'Expected %s, but got %s' % (whom, shape, A.shape))


def _is_dask_array(X):
    return isinstance(X, da.Array)


def _to_2d_array(X, rowvar=True):
    if not _is_dask_array(X):
        return default_backend._to_2d_array(X, rowvar=rowvar)

    if X.ndim < 2:
        raise ValueError(
            'array with fewer than 2 dimensions passed to _to_2d_array')
    elif X.ndim == 2:
        return X

    if rowvar:
        n_samples = X.shape[-1]
        n_features = np.product(X.shape[:-1])

        return da.reshape(X, (n_features, n_samples))
    else:
        n_samples = X.shape[0]
        n_features = np.product(X.shape[1:])

        return da.reshape(X, (n_samples, n_features))


def _to_nd_array(X, target_shape, rowvar=True):
    if not _is_dask_array(X):
        return default_backend._to_nd_array(X, target_shape, rowvar=rowvar)

    if X.ndim != 2:
        raise ValueError(
            'array with number of dimensions other than '
            '2 passed to _to_nd_array')

    rowvar = int(bool(rowvar))
    n_samples = X.shape[rowvar]
    n_features = X.shape[1 - rowvar]

    n_target_features = np.product(target_shape)
    if n_target_features != n_features:
        raise ValueError(
            'number of elements of target shape does not match '
            'number of elements in given array '
            '(got target shape = %r, but X.shape[%d] = %d' %
            (target_shape, 1 - rowvar, n_features))

    if rowvar:
        return da.reshape(X, target_shape + (n_samples,))
    else:
        return da.reshape(X, (n_samples,) + target_shape)


def _center_data(X, axis=0, skipna=True, dtype=None):
    if skipna:
        X_bar = da.nanmean(X, axis=axis, dtype=dtype, keepdims=True)
    else:
        X_bar = da.mean(X, axis=axis, dtype=dtype, keepdims=True)

    return X - X_bar


def _has_fixed_missing_values(X, rowvar=True):
    nan_mask = da.isnan(X)
    rowvar = int(bool(rowvar))
    return (nan_mask.any(axis=rowvar) == nan_mask.all(axis=rowvar)).all()


def _check_fixed_missing_values(X, rowvar=True):
    if not _has_fixed_missing_values(X, rowvar=rowvar):
        raise ValueError(
            'variable has partial missing values')


def _get_valid_variables(X, rowvar=True):
    if rowvar:
        valid_vars = da.nonzero(da.logical_not(
            da.isnan(X[:, 0])))[0].compute()
        valid_data = X[valid_vars]
    else:
        valid_vars = da.nonzero(da.logical_not(
            da.isnan(X[0])))[0].compute()
        valid_data = X[:, valid_vars]

    return valid_data, valid_vars


def _calc_svd_dask(X, k, compressed=True,
                   n_power_iter=0, seed=None):
    if compressed:
        dsvd = da.linalg.svd_compressed(X, k=k, n_power_iter=n_power_iter,
                                        seed=seed)
        u, s, vt = (x.compute() for x in dsvd)
    else:
        dsvd = da.linalg.svd(X)
        u, s, vt = (x.compute() for x in dsvd)

        u = u[:, :min(k, len(s))]
        if k < len(s):
            s = s[:k]
        vt = vt[:min(k, len(s)), :]

    return u, s, vt


def _calc_eofs_dask_svd(X, n_components=None, rowvar=True, center=True,
                        bias=False, ddof=None, normalize_pcs=False,
                        compressed=False, n_power_iter=4, seed=None):
    """Calculate standard EOFs using SVD of data matrix.

    Given a data matrix X in which each row corresponds to a variable
    (i.e., each column is a separate observation), the rank-k SVD of
    X is first computed as

        X ~= U_k * Sigma_k * V_k^T ,

    where k = n_components is the number of EOFs to retain, and U_k,
    Sigma_k, V_k are the truncated SVD factors.

    In the case that the principal components are not normalized, the
    PCs are then given by

        PCs = Sigma_k * V_k^T ,

    and the corresponding EOFs are

        EOFs = U_k ,

    such that the reconstructed data matrix is X_rec = EOFs * PCs.

    Note that, with these conventions, the sample covariance matrix is
    given by::

        C = X * X^T / (n_samples - ddof)
          ~= U_k Sigma_k^2 U_k.T / (n_samples - ddof) .

    Hence, the variance explained by each mode i is given by
    S_{ii}^2 / (n_samples - ddof).

    Variables with partially missing values are not supported, that is,
    a variable must either have no missing values or must have all
    missing values. In the latter case, the variable is ignored in the
    analysis.

    Parameters
    ----------
    X : array-like, shape (n_variables, n_samples) or (n_samples, n_variables)
        Data array containing data to analyse. If rowvar=True (default),
        each row of X is taken to correspond to a separate variable.
        Otherwise, the columns of X are taken to correspond to separate
        variables.

    n_components : integer or None
        If an integer, the number of principal components to retain.
        If None, all variables are kept.

    rowvar : boolean, default: True
        If True, the rows of X are taken to correspond to variables.
        If False, the columns of X are taken to correspond to variables.

    center : boolean, default: True
        If True, center the data before performing the SVD by subtracting
        the mean of each variable.

    bias : boolean, default: False
        If False, normalize the covariance matrix by n_samples - 1.
        If True, normalize by n_samples. These values may be overridden
        by using the ddof argument.

    ddof : integer or None
        If an integer, normalize the covariance matrix by n_samples - ddof.
        Note that this overrides the value implied by the bias argument.

    normalize_pcs : boolean, default: False
        If True, the returned PCs are normalized to have unit variance.
        If False, the returned PCs are scaled by the singular values
        of X.

    Returns
    -------
    eofs : array-like, shape (n_variables, n_components) or (n_components, n_variables)
        Array containing the EOFs of the data.

    pcs : array-like, shape (n_components, n_samples) or (n_samples, n_components)

    ev : array, shape (n_components,)
        The variance explained by each of the EOF modes.

    evr : array, shape (n_components,)
        The fraction of the total variance explained by each EOF mode.
        The total fraction of the variance explained is given by
        the sum the entries.

    sv : array, shape (n_components,)
        The leading singular values of the given data.
    """
    _check_fixed_missing_values(X, rowvar=rowvar)

    if ddof is None:
        if bias:
            ddof = 0
        else:
            ddof = 1

    valid_data, valid_vars = _get_valid_variables(X, rowvar=rowvar)

    rowvar = int(bool(rowvar))

    if center:
        valid_data = _center_data(valid_data, axis=rowvar)

    n_features = X.shape[1 - rowvar]
    n_valid_features = valid_data.shape[1 - rowvar]

    n_samples = valid_data.shape[rowvar]
    fact = n_samples * 1. - ddof

    if n_components is None:
        n_components = n_valid_features

    u, s, vt = _calc_svd_dask(valid_data, k=n_components,
                              compressed=compressed,
                              n_power_iter=n_power_iter, seed=seed)

    variances = da.var(X, ddof=ddof, axis=rowvar)
    ev = s ** 2 / fact
    evr = ev / variances.sum()

    if rowvar:
        if normalize_pcs:
            scaled_u = da.matmul(u, da.diag(s))
            eof_rows = [None] * n_features
            row_pos = 0
            for i in range(n_features):
                if i in valid_vars:
                    eof_rows[i] = scaled_u[row_pos]
                    row_pos += 1
                else:
                    eof_rows[i] = da.full((1, n_components), np.NaN)
            eofs = da.vstack(eof_rows)
            pcs = np.sqrt(fact) * vt
        else:
            scaled_u = u
            eof_rows = [None] * n_features
            row_pos = 0
            for i in range(n_features):
                if i in valid_vars:
                    eof_rows[i] = scaled_u[row_pos]
                    row_pos += 1
                else:
                    eof_rows[i] = da.full((1, n_components), np.NaN)
            eofs = da.vstack(eof_rows)
            pcs = da.matmul(da.diag(s), vt)
    else:
        if normalize_pcs:
            scaled_vt = da.matmul(da.diag(s) / np.sqrt(fact), vt)
            eof_cols = [None] * n_features
            col_pos = 0
            for i in range(n_features):
                if i in valid_vars:
                    eof_cols[i] = np.reshape(scaled_vt[:, col_pos],
                                             (n_components, 1))
                    col_pos += 1
                else:
                    eof_cols[i] = da.full((n_components, 1), np.NaN)
            eofs = da.hstack(eof_cols)
            pcs = np.sqrt(fact) * u
        else:
            scaled_vt = vt
            eof_cols = [None] * n_features
            col_pos = 0
            for i in range(n_features):
                if i in valid_vars:
                    eof_cols[i] = np.reshape(scaled_vt[:, col_pos],
                                             (n_components, 1))
                    col_pos += 1
                else:
                    eof_cols[i] = da.full((n_components, 1), np.NaN)
            eofs = da.hstack(eof_cols)
            pcs = da.matmul(u, da.diag(s))

    return eofs, pcs, ev, evr, s


def _calc_eofs_dask(X, n_components=None, rowvar=True, method=None,
                    **kwargs):
    """Calculate EOFs of the given dataset.

    Parameters
    ----------
    X : array-like, shape (n_variables, n_samples) or (n_samples, n_variables)
        Data array containing data to analyse. If rowvar=True (default),
        each row of X is taken to correspond to a separate variable.
        Otherwise, the columns of X are taken to correspond to separate
        variables.

    n_components : integer or None
        If an integer, the number of principal components to retain.
        If None, all variables are kept.

    rowvar : boolean, default: True
        If True, the rows of X are taken to correspond to variables.
        If False, the columns of X are taken to correspond to variables.

    method : None | 'svd' | 'cov'
        The method used to compute the EOFs and PCs. If None, defaults to
        'svd' if the data has fixed location missing values, and 'cov'
        otherwise.

    center : boolean, default: True
        If True, center the data before performing the SVD by subtracting
        the mean of each variable.

    bias : boolean, default: False
        If False, normalize the covariance matrix by n_samples - 1.
        If True, normalize by n_samples. These values may be overridden
        by using the ddof argument.

    ddof : integer or None
        If an integer, normalize the covariance matrix by n_samples - ddof.
        Note that this overrides the value implied by the bias argument.

    normalize_pcs : boolean, default: False
        If True, the returned PCs are normalized to have unit variance.
        If False, the returned PCs are scaled by the singular values
        of X.

    Returns
    -------
    eofs : array-like, shape (n_variables, n_components) or (n_components, n_variables)
        Array containing the EOFs of the data.

    pcs : array-like, shape (n_components, n_samples) or (n_samples, n_components)

    ev : array, shape (n_components,)
        The variance explained by each of the EOF modes.

    evr : array, shape (n_components,)
        The fraction of the total variance explained by each EOF mode.
        The total fraction of the variance explained is given by
        the sum the entries.
    """
    if X.ndim < 2:
        raise ValueError(
            'Matrix with at least two dimensions expected '
            '(got X.ndim = %d)' % X.ndim)

    if rowvar:
        original_shape = X.shape[:-1]
    else:
        original_shape = X.shape[1:]

    data_2d = _to_2d_array(X, rowvar=rowvar)

    if method is None:
        if _has_fixed_missing_values(data_2d, rowvar=rowvar):
            method = 'svd'
        else:
            method = 'cov'

    if method == 'svd':
        eofs_2d, pcs, ev, evr, _ = _calc_eofs_dask_svd(
            data_2d, rowvar=rowvar, **kwargs)
    else:
        raise ValueError(
            "invalid method parameter '%r'" % method)

    eofs = _to_nd_array(eofs_2d, original_shape, rowvar=rowvar)

    return eofs, pcs, ev, evr
