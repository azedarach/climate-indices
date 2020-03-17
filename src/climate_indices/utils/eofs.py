"""
Provides routines for performing EOF analysis.
"""


import numbers
import warnings

import dask.array as da
import numpy as np
import scipy.linalg as sl
import xarray as xr

from scipy.sparse.linalg import svds

from .validation import (check_fixed_missing_values, get_valid_variables,
                         is_dask_array, is_data_array, is_dataset,
                         is_xarray_object)

INTEGER_TYPES = (numbers.Integral, np.integer)


def _get_other_dims(ds, dim):
    """Return all dimensions other than the given dimension."""

    all_dims = list(ds.dims)

    return [d for d in all_dims if d != dim]


def _check_either_all_xarray_or_plain_arrays(objects):
    """Check if all elements of list are xarray objects or plain arrays."""

    has_xarray_input = False
    has_plain_array_input = False

    for obj in objects:
        if is_xarray_object(obj):
            has_xarray_input = True
        else:
            has_plain_array_input = True

    if has_xarray_input and has_plain_array_input:
        raise NotImplementedError(
            'mixed xarray and plain array input not supported')

    return has_xarray_input


def _expand_and_weight_datasets(objects, sample_dim=0, weight=None):
    """Replace xarray Datasets in a list by separate DataArrays."""

    arrays = []
    weights = []

    if weight is None or is_data_array(weight):
        has_common_weight = True
    elif is_dataset(weight):
        raise ValueError(
            'weights must either be an xarray DataArray or list of DataArrays')
    else:
        if len(weight) != len(objects):
            raise ValueError(
                'number of weight arrays does not match number of input arrays')
        has_common_weight = False

    for idx, obj in enumerate(objects):
        if is_data_array(obj):
            arrays.append(obj)
            if has_common_weight:
                weights.append(weight)
            else:
                weights.append(weight[idx])
        else:
            for v in obj.data_vars:
                arrays.append(obj[v])
                if has_common_weight:
                    weights.append(weight)
                else:
                    weights.append(weight[idx])

    initial_dims = [list(a.dims) for a in arrays]
    arrays = [w.fillna(0) * arrays[i] if w is not None else arrays[i]
              for i, w in enumerate(weights)]

    if not isinstance(sample_dim, INTEGER_TYPES):
        n_arrays = len(arrays)
        for i in range(n_arrays):
            if list(arrays[i].dims) != initial_dims[i]:
                arrays[i] = arrays[i].transpose(*initial_dims[i])

            if arrays[i].get_axis_num(sample_dim) != 0:
                other_dims = _get_other_dims(arrays[i], sample_dim)
                arrays[i] = arrays[i].transpose(*([sample_dim] + other_dims))
    else:
        if sample_dim != 0:
            raise ValueError('sampling dimension must be first axis')

    return arrays


def _get_data_values(X):
    """Get numerical values from object."""

    if is_data_array(X):
        return X.values

    return X


def _fix_svd_phases(u, vh):
    """Impose fixed phase convention on left- and right-singular vectors.

    Given a set of left- and right-singular vectors as the columns of u
    and rows of vh, respectively, imposes the phase convention that for
    each left-singular vector, the element with largest absolute value
    is real and positive.

    Parameters
    ----------
    u : array, shape (M, K)
        Unitary array containing the left-singular vectors as columns.

    vh : array, shape (K, N)
        Unitary array containing the right-singular vectors as rows.

    Returns
    -------
    u_fixed : array, shape (M, K)
        Unitary array containing the left-singular vectors as columns,
        conforming to the chosen phase convention.

    vh_fixed : array, shape (K, N)
        Unitary array containing the right-singular vectors as rows,
        conforming to the chosen phase convention.
    """

    n_cols = u.shape[1]
    max_elem_rows = np.argmax(np.abs(u), axis=0)

    if np.any(np.iscomplexobj(u)):
        phases = np.exp(-1j * np.angle(u[max_elem_rows, range(n_cols)]))
    else:
        phases = np.sign(u[max_elem_rows, range(n_cols)])

    u *= phases
    vh *= phases[:, np.newaxis]

    return u, vh


def _calc_truncated_svd(X, k):
    """Calculate the truncated SVD of a 2D array.

    Given an array X with shape (M, N), the SVD of X is computed and the
    leading K = min(k, min(M, N)) singular values are retained.

    The singular values are returned as a 1D array in non-increasing
    order, and the singular vectors are defined such that the array X
    is decomposed as ```u @ np.diag(s) @ vh```.

    Parameters
    ----------
    X : array, shape (M, N)
        The matrix to calculate the SVD of.

    k : integer
        Number of singular values to retain in truncated decomposition.
        If k > min(M, N), then all singular values are retained.

    Returns
    -------
    u : array, shape (M, K)
        Unitary array containing the retained left-singular vectors of X
        as columns.

    s : array, shape (K)
        Array containing the leading K singular vectors of X.

    vh : array, shape (K, N)
        Unitary array containing the retained right-singular vectors of
        X as rows.
    """

    max_modes = min(X.shape)

    if is_dask_array(X):
        dsvd = da.linalg.svd(X)

        u, s, vh = (x.compute() for x in dsvd)

        u = u[:, :min(k, len(s))]
        if k < len(s):
            s = s[:k]
        vh = vh[:min(k, len(s)), :]
    elif k < max_modes:
        u, s, vh = svds(X, k=k)

        # Note that svds returns the singular values with the
        # opposite (i.e., non-decreasing) ordering convention.
        u = u[:, ::-1]
        s = s[::-1]
        vh = vh[::-1]
    else:
        u, s, vh = sl.svd(X, full_matrices=False)

        u = u[:, :min(k, len(s))]
        if k < len(s):
            s = s[:k]
        vh = vh[:min(k, len(s)), :]

    # Impose a fixed phase convention on the singular vectors
    # to avoid phase ambiguity.
    u, vh = _fix_svd_phases(u, vh)

    return u, s, vh


def _apply_weights_to_arrays(arrays, weights=None):
    """Apply weights to the corresponding input array.

    Given a list of arrays and an equal length list of
    weights, apply the weights to the corresponding array
    element-wise.

    Each given array is assumed to have shape (N, ...) where N
    is the number of samples. The corresponding element in the
    list of weights must either be None, in which case no weighting
    is applied, or must have shape compatible with the remaining
    dimensions of the input array.

    Parameters
    ----------
    arrays : list of arrays
        List of arrays to apply weights to.

    weights : None or list of arrays
        List of weights to apply. If None, no weighting is applied
        and the original arrays are returned. If not None, each
        element of the list should be either None or broadcastable
        to the shape of the corresponding list of input arrays.

    Returns
    -------
    weighted_arrays : list of arrays
        List of arrays with element-wise weights applied.
    """

    if weights is None:
        return arrays

    if weights is not None:
        if not isinstance(weights, tuple) and not isinstance(weights, list):
            weights = list(weights)

    if len(arrays) != len(weights):
        raise ValueError(
            'number of weight arrays does not match number of input arrays')

    has_dask_arrays = np.any([is_dask_array(a) for a in arrays])

    if has_dask_arrays:
        weights_to_apply = [da.broadcast_arrays(
            arrays[i][0:1], w)[1][0]
                            if w is not None
                            else da.ones_like(arrays[i][0:1])
                            for i, w in enumerate(weights)
                            ]
    else:
        weights_to_apply = [np.broadcast_arrays(
            arrays[i][0:1], w)[1][0]
                            if w is not None
                            else np.ones(arrays[i][0:1].shape)
                            for i, w in enumerate(weights)]

    return [a * weights_to_apply[i] for i, a in enumerate(arrays)]


def _eofs_impl(*arrays, n_modes=None,
               bias=False, ddof=None, skipna=True):
    """Perform standard empirical orthogonal function analysis.

    Given one or more arrays, combines the contents of the arrays
    to form a single dataset and performs a standard empirical orthogonal
    function (EOF) analysis on the combined dataset.

    Each array is assumed to have shape (N, ...), where N is the number
    of observations or samples, and hence all input arrays must have the
    same number of elements in the first dimension. The remaining dimensions
    for each array may differ.

    Parameters
    ----------
    *arrays : arrays of shape (N, ...)
        Arrays for perform EOF analysis on. All of the input arrays must
        have the same length first dimension.

    n_modes : None or integer, optional
        If an integer, the number of EOF modes to compute. If None, the
        maximum possible number of modes is computed.

    bias : boolean, default: False
        If False, normalize the covariance matrix by N - 1.
        If True, normalize by N. These values may be overridden
        by using the ddof argument.

    ddof : integer or None
        If an integer, normalize the covariance matrix by N - ddof.
        Note that this overrides the value implied by the bias argument.

    skipna : boolean, optional
       If True, ignore NaNs for the purpose of computing the total variance.

    Returns
    -------
    eofs : list of arrays of shape (n_modes, ...)
        List of arrays containing the components of the EOFs associated
        with each of the input arrays.

    pcs : array, shape (N, n_modes)
        Array containing the corresponding principal components.

    lambdas : array, shape (n_modes,)
        Array containing the eigenvalues of the covariance matrix.

    explained_var : array, shape (n_modes,)
        Array containing the fraction of the total variance associated with
        each mode.
    """

    n_input_arrays = len(arrays)

    # All inputs must have the same number of samples, i.e.,
    # same size for the first dimension.
    n_samples = None
    has_dask_array = False
    for a in arrays:
        if n_samples is None:
            n_samples = a.shape[0]
        else:
            if a.shape[0] != n_samples:
                raise ValueError(
                    'numbers of samples do not agree for all arrays')

        check_fixed_missing_values(a, axis=0)
        has_dask_array = has_dask_array or is_dask_array(a)

    # Retain original shapes for reshaping after computing EOFs.
    original_shapes = [a.shape[1:] for a in arrays]
    original_sizes = [np.product(shp) for shp in original_shapes]

    # Reshape to 2D to perform SVD on.
    flat_arrays = [a.reshape((n_samples, original_sizes[i]))
                   for i, a in enumerate(arrays)]

    # Remove fixed missing values and store indices of corresponding
    # features for later replacement.
    valid_features = []
    valid_sizes = []
    for i, a in enumerate(flat_arrays):
        flat_arrays[i], valid_vars = get_valid_variables(a)
        valid_features.append(valid_vars)
        valid_sizes.append(flat_arrays[i].shape[1])

    if has_dask_array:
        concatenate_arrays = da.concatenate
    else:
        concatenate_arrays = np.concatenate

    combined_dataset = concatenate_arrays(flat_arrays, axis=1)
    if has_dask_array:
        # Initial workaround to avoid error if the concatenated
        # array is not tall-and-skinny as required by dask.linalg.svd.
        combined_dataset = combined_dataset.rechunk('auto')

    if n_modes is None:
        n_modes = min(combined_dataset.shape)

    u, s, vh = _calc_truncated_svd(combined_dataset, k=n_modes)

    if ddof is None:
        if bias:
            ddof = 0
        else:
            ddof = 1

    fact = n_samples * 1. - ddof

    lambdas = s ** 2 / fact

    if skipna:
        calc_variance = np.nanvar if not has_dask_array else da.nanvar
    else:
        calc_variance = np.var if not has_dask_array else da.var

    variances = calc_variance(combined_dataset, ddof=ddof, axis=0)
    explained_var = lambdas / variances.sum()

    pcs = np.dot(u, np.diag(s)) if not has_dask_array else da.dot(u, da.diag(s))

    eofs = []
    pos = 0
    for i in range(n_input_arrays):
        input_shape = original_shapes[i]
        input_size = original_sizes[i]
        valid_cols = valid_features[i]
        var_size = valid_sizes[i]

        if has_dask_array:
            eof_2d_cols = [None] * input_size
            for j in range(input_size):
                if j in valid_cols:
                    eof_2d_cols[j] = vh[:, pos].reshape((n_modes, 1))
                    pos += 1
                else:
                    eof_2d_cols[j] = da.full((n_modes, 1), np.NaN)

            eof_2d_components = da.hstack(eof_2d_cols)
        else:
            eof_2d_components = np.full((n_modes, input_size), np.NaN)
            eof_2d_components[:, valid_features[i]] = vh[:, pos:pos + var_size]
            pos += var_size

        eofs.append(eof_2d_components.reshape((n_modes,) + input_shape))

    return eofs, pcs, lambdas, explained_var


def _bsv_orthomax_rotation(A, T0=None, gamma=1.0, tol=1e-6, max_iter=500):
    """Returns optimal rotation found by BSV algorithm.

    Given an initial column-wise orthonormal matrix A of dimension
    p x k, p >= k, the orthomax family of criteria seek an
    orthogonal matrix T that maximizes the objective function

        Q(Lambda) = (1 / 4) Tr[(L ^ 2)^T (L^2 - gamma * \bar{L^2})]

    where L = A * T, L^2 denotes the element-wise square
    of L, and \bar{L^2} denotes the result of replacing each
    element of L^2 by the mean of its corresponding column. The
    parameter gamma defines the particular objective function to be
    maximized; for gamma = 1, the procedure corresponds to
    VARIMAX rotation.

    Parameters
    ----------
    A : array-like, shape (n_features, n_components)
        The matrix to be rotated.

    T0 : None or array-like, shape (n_components, n_components)
        If given, an initial guess for the rotation matrix T.

    gamma : float, default: 1.0
        Objective function parameter.

    tol : float, default: 1e-6
        Tolerance of the stopping condition.

    max_iter : integer, default: 500
        Maximum number of iterations before stopping.

    Returns
    -------
    T : array-like, shape (n_components, n_components)
        Approximation to the optimal rotation.

    n_iter : integer
        The actual number of iterations.

    References
    ----------
    R. I. Jennrich, "A simple general procedure for orthogonal rotation",
    Psychometrika 66, 2 (2001), 289-306
    """

    n_features, n_components = A.shape

    if n_components > n_features:
        raise ValueError(
            'Number of rows in input array must be greater than '
            'or equal to number of columns, got A.shape = %r' %
            A.shape)

    if T0 is None:
        if is_dask_array(A):
            T = da.eye(n_components, dtype=A.dtype)
        else:
            T = np.eye(n_components, dtype=A.dtype)
    else:
        if T0.shape != (n_components, n_components):
            raise ValueError(
                'Array with wrong shape passed to %s. '
                'Expected %s, but got %s' %
                ('_bsv_orthomax_rotation',
                 (n_components, n_components), A.shape))
        T = T0.copy()

    if is_dask_array(A):
        to_diag = da.diag
        calc_svd = da.linalg.svd
    else:
        to_diag = np.diag
        calc_svd = np.linalg.svd

    delta = 0
    for n_iter in range(max_iter):
        delta_old = delta

        Li = A.dot(T)

        grad = Li ** 3 - gamma * Li.dot(to_diag((Li ** 2).mean(axis=0)))
        G = (A.T).dot(grad)

        u, s, vt = calc_svd(G)

        T = u.dot(vt)

        delta = s.sum()
        if delta < delta_old * (1 + tol):
            break

    if n_iter == max_iter and tol > 0:
        warnings.warn('Maximum number of iterations %d reached.' % max_iter,
                      UserWarning)

    return T, n_iter


def _reorder_eofs(eofs, pcs, lambdas, explained_variance):
    """Sorts modes in descending order of explained variance."""

    n_components = np.size(explained_variance)
    sort_order = np.flip(np.argsort(explained_variance))

    perm_inv = np.zeros((n_components, n_components))
    for i in range(n_components):
        perm_inv[i, sort_order[i]] = 1

    return (eofs[sort_order], pcs[:, sort_order],
            lambdas[sort_order], explained_variance[sort_order])


def _varimax_reofs_impl(*arrays, n_modes=None, n_rotated_modes=None,
                        scale_eofs=True, kaiser_normalise=True,
                        T0=None, tol=1e-6, max_iter=500,
                        bias=False, ddof=None, skipna=True):
    """Perform VARIMAX rotated empirical orthogonal function analysis.

    Given one or more arrays, combines the contents of the arrays
    to form a single dataset and first performs a standard empirical orthogonal
    function (EOF) analysis on the combined dataset. The obtained EOFs
    are then rotated using the VARIMAX rotation criterion.

    Each array is assumed to have shape (N, ...), where N is the number
    of observations or samples, and hence all input arrays must have the
    same number of elements in the first dimension. The remaining dimensions
    for each array may differ.

    Parameters
    ----------
    *arrays : arrays of shape (N, ...)
        Arrays for perform EOF analysis on. All of the input arrays must
        have the same length first dimension.

    n_modes : None or integer, optional
        If an integer, the number of EOF modes to compute. If None, the
        maximum possible number of modes is computed.

    n_rotated_modes : None or integer, optional
        If an integer, the number of modes to retain when performing
        rotation. If None, all modes are included in the rotation.

    scale_eofs : boolean, optional
        If True, scale the EOFs by multiplying by the square root
        of the corresponding eigenvalue of the covariance matrix.
        In this case, the rotated PCs are uncorrelated.

    kaiser_normalise : boolean, optional
        Use Kaiser normalisation for EOFs.

    T0 : None or array-like, shape (n_rotated_modes, n_rotated_modes)
        If given, an initial guess for the rotation matrix to be
        used.

    tol : float, optional
        Stopping tolerance for calculating rotation matrix.

    max_iter : integer, optional
        The maximum number of iterations to allow in calculating
        the rotation matrix.

    bias : boolean, default: False
        If False, normalize the covariance matrix by N - 1.
        If True, normalize by N. These values may be overridden
        by using the ddof argument.

    ddof : integer or None
        If an integer, normalize the covariance matrix by N - ddof.
        Note that this overrides the value implied by the bias argument.

    skipna : boolean, optional
       If True, ignore NaNs for the purpose of computing the total variance.

    Returns
    -------
    reofs : list of arrays of shape (n_rotated_modes, ...)
        List of arrays containing the components of the EOFs associated
        with each of the input arrays.

    pcs : array, shape (N, n_rotated_modes)
        Array containing the corresponding principal components.

    lambdas : array, shape (n_rotated_modes,)
        Array containing the eigenvalues of the covariance matrix.

    explained_var : array, shape (n_rotated_modes,)
        Array containing the fraction of the total variance associated with
        each mode.

    rotation : array, shape (n_rotated_modes, n_rotated_modes)
        Array containing the orthogonal rotation matrix used to
        obtain the rotated EOFs.
    """

    varimax_gamma = 1.0

    n_input_arrays = len(arrays)

    eofs, pcs, lambdas, explained_var = _eofs_impl(
        *arrays, n_modes=n_modes, bias=bias, ddof=ddof, skipna=skipna)

    has_dask_array = False
    for a in eofs:
        has_dask_array = has_dask_array or is_dask_array(a)

    n_modes = pcs.shape[1]

    if n_rotated_modes is None:
        n_rotated_modes = n_modes
    elif n_rotated_modes > n_modes:
        raise ValueError('Number of rotated modes must be less than or equal '
                         'to the number of unrotated modes')

    total_variance = lambdas[0] / explained_var[0]

    # Select subset of modes for rotation.
    reofs = [a[:n_rotated_modes] for a in eofs]
    pcs = pcs[:, :n_rotated_modes]
    lambdas = lambdas[:n_rotated_modes]
    explained_var = explained_var[:n_rotated_modes]

    # Retain original shapes for reshaping after computing rotated EOFs.
    original_shapes = [a.shape[1:] for a in reofs]
    original_sizes = [np.product(shp) for shp in original_shapes]

    # Reshape to 2D to perform rotation on.
    flat_arrays = [a.reshape((n_rotated_modes, original_sizes[i]))
                   for i, a in enumerate(reofs)]

    # Remove fixed missing values and store indices of corresponding
    # features for later replacement.
    valid_features = []
    valid_sizes = []
    for i, a in enumerate(flat_arrays):
        flat_arrays[i], valid_vars = get_valid_variables(a)
        valid_features.append(valid_vars)
        valid_sizes.append(flat_arrays[i].shape[1])

    if has_dask_array:
        concatenate_arrays = da.concatenate
    else:
        concatenate_arrays = np.concatenate

    combined_eofs = concatenate_arrays(flat_arrays, axis=1)
    if has_dask_array:
        # Initial workaround to avoid error if the concatenated
        # array is not tall-and-skinny as required by dask.linalg.svd.
        combined_eofs = combined_eofs.rechunk('auto')

    if scale_eofs:
        combined_eofs *= np.sqrt(lambdas[:, np.newaxis])
        pcs /= np.sqrt(lambdas[np.newaxis, :])

    if kaiser_normalise:
        normalisation = np.sqrt((combined_eofs ** 2).sum(axis=0))
        normalisation[normalisation == 0] = 1
        combined_eofs /= normalisation

    # Perform rotation on combined EOFs.
    rotation, _ = _bsv_orthomax_rotation(
        combined_eofs.T, T0=T0, gamma=varimax_gamma, tol=tol, max_iter=max_iter)

    pcs = pcs.dot(rotation)
    rotated_combined_eofs = (rotation.T).dot(combined_eofs)

    if kaiser_normalise:
        rotated_combined_eofs *= normalisation

    # Recompute amount of variance explained by each of the individual
    # rotated modes. Note that this only strictly meaningful if
    # the EOFs are scaled so that the rotated modes remain uncorrelated.
    lambdas = (rotated_combined_eofs ** 2).sum(axis=1)
    explained_var = lambdas / total_variance

    # Ensure returned EOFs and PCs are still ordered according
    # to the amount of variance explained.
    rotated_combined_eofs, pcs, lambdas, explained_var = _reorder_eofs(
        rotated_combined_eofs, pcs, lambdas, explained_var)

    reofs = []
    pos = 0
    for i in range(n_input_arrays):
        input_shape = original_shapes[i]
        input_size = original_sizes[i]
        valid_cols = valid_features[i]
        var_size = valid_sizes[i]

        if has_dask_array:
            reof_2d_cols = [None] * input_size
            for j in range(input_size):
                if j in valid_cols:
                    reof_2d_cols[j] = rotated_combined_eofs[:, pos].reshape(
                        (n_rotated_modes, 1))
                    pos += 1
                else:
                    reof_2d_cols[j] = da.full((n_rotated_modes, 1), np.NaN)

            reof_2d_components = da.hstack(reof_2d_cols)
        else:
            reof_2d_components = np.full((n_rotated_modes, input_size), np.NaN)
            reof_2d_components[:, valid_features[i]] = rotated_combined_eofs[
                :, pos:pos + var_size]
            pos += var_size

        reofs.append(reof_2d_components.reshape(
            (n_rotated_modes,) + input_shape))

    return reofs, pcs, lambdas, explained_var, rotation


def eofs(*objects, sample_dim=None, weight=None, n_modes=None):
    """Perform standard empirical orthogonal function analysis.

    Given one or more objects, perform a standard empirical orthogonal
    function (EOF) analysis on the dataset formed from the combined objects.

    Parameters
    ----------
    *objects : arrays
        Objects containing data for perform EOF analysis on.

    sample_dim : str or integer, optional
        Axis corresponding to the sampling dimension.

    weight : arrays, optional
        If given, weights to apply to the data. If multiple objects are
        given, the number of elements of weight must be the same as the
        given number of objects. For each object used in the analysis,
        the given weight, if not None, must be broadcastable onto the
        corresponding data object.

    n_modes : None or integer, optional
        The number of EOFs to retain. If None, then the maximum
        possible number of EOFs will be retained.

    Returns
    -------
    eofs : xarray Dataset
        Dataset containing the following variables:

        - 'EOFs' : array containing the empirical orthogonal functions

        - 'PCs' : array containing the associated principal components

        - 'lambdas' : array containing the eigenvalues of the covariance
          matrix of the input data

        - 'explained_var' : array containing the fraction of the total
          variance explained by each mode
    """

    # Currently, mixed xarray and plain array inputs are not supported.
    is_xarray_input = _check_either_all_xarray_or_plain_arrays(objects)

    if is_xarray_input:
        def _get_time_name(obj):
            if 'time' in obj.dims:
                return 'time'
            return None

        if sample_dim is None:
            for obj in objects:
                if is_xarray_object(obj):
                    sample_dim = _get_time_name(obj)
                    if sample_dim is not None:
                        break

        if sample_dim is None:
            raise RuntimeError(
                'Unable to automatically determine sampling dimension')

        arrays = _expand_and_weight_datasets(
            objects, sample_dim=sample_dim, weight=weight)
    else:
        if sample_dim is None:
            sample_dim = 0

        if not isinstance(sample_dim, INTEGER_TYPES):
            raise ValueError(
                'sample dimension must be an integer for plain arrays')

        if sample_dim != 0:
            raise ValueError('sampling dimension must be first axis')

        arrays = _apply_weights_to_arrays(objects, weights=weight)

    # Convert to plain arrays for computation.
    plain_arrays = [_get_data_values(a) for a in arrays]

    eofs_arr, pcs_arr, lambdas_arr, expl_var_arr = _eofs_impl(
        *plain_arrays, n_modes=n_modes)

    n_arrays = len(arrays)
    n_modes = pcs_arr.shape[1]

    result = []
    if is_xarray_input:
        for idx in range(n_arrays):
            input_dims = _get_other_dims(arrays[idx], sample_dim)

            data_vars = {'EOFs': (['mode'] + input_dims, eofs_arr[idx]),
                         'PCs': ([sample_dim, 'mode'], pcs_arr),
                         'lambdas': (['mode'], lambdas_arr),
                         'explained_var': (['mode'], expl_var_arr)
                         }

            coords = dict(arrays[idx].coords.items())
            coords['mode'] = np.arange(n_modes)

            result.append(
                xr.Dataset(data_vars, coords=coords))
    else:
        for idx in range(n_arrays):
            input_shape = arrays[idx].shape[1:]
            input_dims = ['axis_{:d}'.format(i + 1)
                          for i in range(len(input_shape))]

            data_vars = {'EOFs': (['mode'] + input_dims, eofs_arr[idx]),
                         'PCs': ([sample_dim, 'mode'], pcs_arr),
                         'lambdas': (['mode'], lambdas_arr),
                         'explained_var': (['mode'], expl_var_arr)
                         }

            coords = {d : np.arange(input_shape[i])
                      for i, d in enumerate(input_dims)}
            coords['mode'] = np.arange(n_modes)

            result.append(
                xr.Dataset(data_vars, coords=coords))

    if len(result) == 1:
        return result[0]

    return result


def reofs(*objects, sample_dim=None, weight=None, n_modes=None,
          n_rotated_modes=None, scale_eofs=True, kaiser_normalise=True,
          T0=None, tol=1e-6, max_iter=500):
    """Perform rotated empirical orthogonal function analysis.

    Given one or more objects, perform a VARIMAX rotated empirical orthogonal
    function (EOF) analysis on the dataset formed from the combined objects.

    Parameters
    ----------
    *objects : arrays
        Objects containing data for perform EOF analysis on.

    sample_dim : str or integer, optional
        Axis corresponding to the sampling dimension.

    weight : arrays, optional
        If given, weights to apply to the data. If multiple objects are
        given, the number of elements of weight  must be the same as the
        given number of objects. For each object used in the analysis,
        the given weight, if not None, must be broadcastable onto the
        corresponding data object.

    n_modes : None or integer, optional
        The number of EOFs to retain. If None, then the maximum
        possible number of EOFs will be retained.

    n_rotated_modes : None or integer, optional
        If an integer, the number of modes to retain when performing
        rotation. If None, all modes are included in the rotation.

    scale_eofs : boolean, optional
        If True, scale the EOFs by multiplying by the square root
        of the corresponding eigenvalue of the covariance matrix.
        In this case, the rotated PCs are uncorrelated.

    kaiser_normalise : boolean, optional
        Use Kaiser normalisation for EOFs.

    T0 : None or array-like, shape (n_rotated_modes, n_rotated_modes)
        If given, an initial guess for the rotation matrix to be
        used.

    tol : float, optional
        Stopping tolerance for calculating rotation matrix.

    max_iter : integer, optional
        The maximum number of iterations to allow in calculating
        the rotation matrix.

    Returns
    -------
    eofs : xarray Dataset
        Dataset containing the following variables:

        - 'EOFs' : array containing the rotated empirical orthogonal functions

        - 'PCs' : array containing the associated principal components

        - 'lambdas' : array containing the eigenvalues of the covariance
          matrix of the input data

        - 'explained_var' : array containing the fraction of the total
          variance explained by each mode
    """

    # Currently, mixed xarray and plain array inputs are not supported.
    is_xarray_input = _check_either_all_xarray_or_plain_arrays(objects)

    if is_xarray_input:
        def _get_time_name(obj):
            if 'time' in obj.dims:
                return 'time'
            return None

        if sample_dim is None:
            for obj in objects:
                if is_xarray_object(obj):
                    sample_dim = _get_time_name(obj)
                    if sample_dim is not None:
                        break

        if sample_dim is None:
            raise RuntimeError(
                'Unable to automatically determine sampling dimension')

        arrays = _expand_and_weight_datasets(
            objects, sample_dim=sample_dim, weight=weight)
    else:
        if sample_dim is None:
            sample_dim = 0

        if not isinstance(sample_dim, INTEGER_TYPES):
            raise ValueError(
                'sample dimension must be an integer for plain arrays')

        if sample_dim != 0:
            raise ValueError('sampling dimension must be first axis')

        arrays = _apply_weights_to_arrays(objects, weights=weight)

    # Convert to plain arrays for computation.
    plain_arrays = [_get_data_values(a) for a in arrays]

    eofs_arr, pcs_arr, lambdas_arr, expl_var_arr, _ = _varimax_reofs_impl(
        *plain_arrays, n_modes=n_modes, n_rotated_modes=n_rotated_modes,
        kaiser_normalise=kaiser_normalise, scale_eofs=scale_eofs,
        T0=T0, tol=tol, max_iter=max_iter)

    n_arrays = len(arrays)
    n_modes = pcs_arr.shape[1]

    result = []
    if is_xarray_input:
        for idx in range(n_arrays):
            input_dims = _get_other_dims(arrays[idx], sample_dim)

            data_vars = {'EOFs': (['mode'] + input_dims, eofs_arr[idx]),
                         'PCs': ([sample_dim, 'mode'], pcs_arr),
                         'lambdas': (['mode'], lambdas_arr),
                         'explained_var': (['mode'], expl_var_arr)
                         }

            coords = dict(arrays[idx].coords.items())
            coords['mode'] = np.arange(n_modes)

            result.append(
                xr.Dataset(data_vars, coords=coords))
    else:
        for idx in range(n_arrays):
            input_shape = arrays[idx].shape[1:]
            input_dims = ['axis_{:d}'.format(i + 1)
                          for i in range(len(input_shape))]

            data_vars = {'EOFs': (['mode'] + input_dims, eofs_arr[idx]),
                         'PCs': ([sample_dim, 'mode'], pcs_arr),
                         'lambdas': (['mode'], lambdas_arr),
                         'explained_var': (['mode'], expl_var_arr)
                         }

            coords = {d : np.arange(input_shape[i])
                      for i, d in enumerate(input_dims)}
            coords['mode'] = np.arange(n_modes)

            result.append(
                xr.Dataset(data_vars, coords=coords))

    if len(result) == 1:
        return result[0]

    return result
