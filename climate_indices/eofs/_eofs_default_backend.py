import numpy as np
import warnings


VARIMAX_GAMMA = 1.0


def _check_array_shape(A, shape, whom):
    if np.shape(A) != shape:
        raise ValueError(
            'Array with wrong shape passed to %s. '
            'Expected %s, but got %s' % (whom, shape, np.shape(A)))


def _to_2d_array(X, rowvar=True):
    if X.ndim < 2:
        raise ValueError(
            'array with fewer than 2 dimensions passed to _to_2d_array')
    elif X.ndim == 2:
        return X

    if rowvar:
        n_samples = X.shape[-1]
        n_features = np.product(X.shape[:-1])

        return np.reshape(X, (n_features, n_samples))
    else:
        n_samples = X.shape[0]
        n_features = np.product(X.shape[1:])

        return np.reshape(X, (n_samples, n_features))


def _to_nd_array(X, target_shape, rowvar=True):
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
        return np.reshape(X, target_shape + (n_samples,))
    else:
        return np.reshape(X, (n_samples,) + target_shape)


def _center_data(X, axis=0, skipna=True, dtype=None):
    if skipna:
        X_bar = np.nanmean(X, axis=axis, dtype=dtype, keepdims=True)
    else:
        X_bar = np.mean(X, axis=axis, dtype=dtype, keepdims=True)

    return X - X_bar


def _has_fixed_missing_values(X, rowvar=True):
    nan_mask = np.isnan(X)
    rowvar = int(bool(rowvar))
    return (nan_mask.any(axis=rowvar) == nan_mask.all(axis=rowvar)).all()


def _check_fixed_missing_values(X, rowvar=True):
    if not _has_fixed_missing_values(X, rowvar=rowvar):
        raise ValueError(
            'variable has partial missing values')


def _get_valid_variables(X, rowvar=True):
    if rowvar:
        valid_vars = np.where(np.logical_not(np.isnan(X[:, 0])))[0]
        valid_data = X[valid_vars]
    else:
        valid_vars = np.where(np.logical_not(np.isnan(X[0])))[0]
        valid_data = X[:, valid_vars]

    return valid_data, valid_vars


def _calc_svd_default(X, k):
    u, s, vt = np.linalg.svd(X, full_matrices=False)

    u = u[:, :min(k, len(s))]
    if k < len(s):
        s = s[:k]
    vt = vt[:min(k, len(s)), :]

    return u, s, vt


def _calc_eofs_default_svd(X, n_components=None, rowvar=True, center=True,
                           bias=False, ddof=None, normalize_pcs=False):
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
        n_components = min(n_samples, n_valid_features)

    u, s, vt = _calc_svd_default(valid_data, k=n_components)

    variances = np.var(X, ddof=ddof, axis=rowvar)
    ev = s ** 2 / fact
    evr = ev / variances.sum()

    if rowvar:
        eofs = np.full((n_features, n_components), np.NaN)
        if normalize_pcs:
            eofs[valid_vars] = np.dot(u, np.diag(s)) / np.sqrt(fact)
            pcs = np.sqrt(fact) * vt
        else:
            eofs[valid_vars] = u
            pcs = np.dot(np.diag(s), vt)
    else:
        eofs = np.full((n_components, n_features), np.NaN)
        if normalize_pcs:
            eofs[:, valid_vars] = np.dot(np.diag(s) / np.sqrt(fact), vt)
            pcs = np.sqrt(fact) * u
        else:
            eofs[:, valid_vars] = vt
            pcs = np.dot(u, np.diag(s))

    return eofs, pcs, ev, evr, s


def _calc_eofs_default_cov(X, n_components=None, rowvar=True, center=True,
                           bias=False, ddof=None, normalize_pcs=False):
    """Calculate standard EOFs using eigendecomposition of covariance matrix.

    Given a data matrix X in which each row corresponds to a variable
    (i.e., each column is a separate observation), the covariance matrix
    is formed according to::

        C = X * X^T / (n_samples - ddof)

    where n_samples is the number of observations (columns) and ddof
    is the bias correction determined from the bias and ddof parameters.
    The eigenvalues and eigenvectors of the symmetric covariance matrix
    are computed, and, in the case that the PCs are not normalized to
    unit variance, the EOFs are then given by the eigenvectors of C.

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
    """
    if ddof is None:
        if bias:
            ddof = 0
        else:
            ddof = 1

    X_mask = np.ma.getmaskarray(X)
    missing_mask = np.isnan(X)
    mask = np.logical_or(X_mask, missing_mask)

    masked_data = np.ma.array(X, ndmin=2, copy=True, mask=mask, dtype=X.dtype)
    cov_mat = np.ma.cov(masked_data, rowvar=rowvar, bias=bias, ddof=ddof)

    cov_mask = np.ma.getmaskarray(cov_mat)
    non_missing_vars = np.where(~np.all(cov_mask, axis=0))[0]

    n_features = X.shape[1 - rowvar]
    n_valid_features = non_missing_vars.size

    if n_components is None:
        n_components = n_valid_features

    valid_cov = np.empty((n_valid_features, n_valid_features))
    valid_cov = cov_mat[non_missing_vars][:, non_missing_vars]

    w, v = np.linalg.eigh(valid_cov)

    w = np.flipud(w)[:n_components]
    v = np.fliplr(v)[:, :n_components]

    variances = np.var(X, ddof=ddof, axis=int(bool(rowvar)))
    ev = w[:n_components]
    evr = ev / variances.sum()

    if rowvar:
        eofs = np.full((n_features, n_components), np.NaN)
        if normalize_pcs:
            eofs[non_missing_vars] = np.dot(v, np.diag(1 / np.sqrt(w)))
            pcs = np.dot(eofs[non_missing_vars].T, X[non_missing_vars])
        else:
            eofs[non_missing_vars] = v
            pcs = np.dot(v.T, X[non_missing_vars])
    else:
        eofs = np.full((n_components, n_features), np.NaN)
        if normalize_pcs:
            eofs[:, non_missing_vars] = np.dot(np.diag(1 / np.sqrt(w)), v.T)
            pcs = np.dot(X[:, non_missing_vars], eofs[:, non_missing_vars].T)
        else:
            eofs[:, non_missing_vars] = v.T
            pcs = np.dot(X[:, non_missing_vars], eofs[:, non_missing_vars].T)

    return eofs, pcs, ev, evr


def _calc_eofs_default(X, n_components=None, rowvar=True, method=None,
                       **kwargs):
    """Calculate EOFs of the given dataset.

    Parameters
    ----------
    X : array-like,
        Data array containing data to analyse. If rowvar=True (default),
        the last dimension of X is taken to correspond to distinct
        observations. Otherwise, the first dimension is taken to label
        the separate observations.

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

    if 'center' in kwargs:
        center = kwargs['center']
    else:
        center = True

    if 'bias' in kwargs:
        bias = kwargs['bias']
    else:
        bias = False

    if 'ddof' in kwargs:
        ddof = kwargs['ddof']
    else:
        ddof = None

    if 'normalize_pcs' in kwargs:
        normalize_pcs = kwargs['normalize_pcs']
    else:
        normalize_pcs = False

    if method == 'svd':
        eofs_2d, pcs, ev, evr, _ = _calc_eofs_default_svd(
            data_2d, n_components=n_components, rowvar=rowvar,
            center=center, bias=bias, ddof=ddof,
            normalize_pcs=normalize_pcs)
    elif method == 'cov':
        eofs_2d, pcs, ev, evr = _calc_eofs_default_cov(
            data_2d, n_components=n_components, rowvar=rowvar,
            center=center, bias=bias, ddof=ddof,
            normalize_pcs=normalize_pcs)
    else:
        raise ValueError(
            "invalid method parameter '%r'" % method)

    eofs = _to_nd_array(eofs_2d, original_shape, rowvar=rowvar)

    return eofs, pcs, ev, evr


def _bsv_orthomax_rotation(A, T0=None, gamma=1.0, tol=1e-6, max_iter=500):
    """Return optimal rotation found by BSV algorithm.

    Given an initial set row-wise orthonormal matrix A of dimension
    k x p, p >= k, the orthomax family of criteria seek an
    orthogonal matrix T that maximizes the objective function

        Q(Lambda) = (1 / 4) Tr[(L ^ 2)^T (L^2 - gamma * \bar{L^2})]

    where L = A^T * T, L^2 denotes the element-wise square
    of L, and \bar{L^2} denotes the result of replacing each
    element of L^2 by the mean of its corresponding column. The
    parameter gamma defines the particular objective function to be
    maximized; for gamma = 1, the procedure corresponds to the
    standard VARIMAX rotation.

    Parameters
    ----------
    A : array-like, shape (n_components, n_features)
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
    n_components, n_features = A.shape

    if T0 is None:
        T = np.eye(n_components, dtype=A.dtype)
    else:
        _check_array_shape(
            T0, (n_components, n_components), '_bsv_orthomax_rotation')
        T = T0.copy()

    delta = 0
    for n_iter in range(max_iter):
        delta_old = delta

        Li = np.dot(A.T, T)

        grad = Li ** 3 - gamma * np.dot(Li, np.diag(np.mean(Li ** 2, axis=0)))
        G = np.dot(A, grad)

        u, s, vt = np.linalg.svd(G)

        T = np.dot(u, vt)

        delta = np.sum(s)
        if delta < delta_old * (1 + tol):
            break

    if n_iter == max_iter and tol > 0:
        warnings.warn('Maximum number of iterations %d reached.' % max_iter,
                      warnings.UserWarning)

    return T, n_iter


def _unit_normalize_eofs(pcs, eofs, rowvar=True):
    if rowvar:
        norms = np.linalg.norm(eofs, axis=0)
        return pcs * norms[:, np.newaxis], eofs / norms[np.newaxis, :]
    else:
        norms = np.linalg.norm(eofs, axis=1)
        return pcs * norms[np.newaxis, :], eofs / norms[:, np.newaxis]


def _fix_sign_convention(pcs, eofs, rowvar=True):
    axis = 1 - int(bool(rowvar))
    abs_min = np.abs(np.min(eofs, axis=axis))
    abs_max = np.abs(np.max(eofs, axis=axis))

    flip_signs = abs_min > abs_max

    if rowvar:
        eofs[:, flip_signs] *= -1
        pcs[flip_signs] *= -1
    else:
        eofs[flip_signs] *= -1
        pcs[:, flip_signs] *= -1

    return pcs, eofs


def _reorder_eofs(pcs, eofs, explained_variance, rowvar=True):
    n_components = np.size(explained_variance)
    sort_order = np.flip(np.argsort(explained_variance))

    perm_inv = np.zeros((n_components, n_components))
    for i in range(n_components):
        perm_inv[i, sort_order[i]] = 1

    if rowvar:
        return (pcs[sort_order], eofs[:, sort_order],
                explained_variance[sort_order])
    else:
        return (pcs[:, sort_order], eofs[sort_order],
                explained_variance[sort_order])


def _varimax_rotation_2d(eofs_2d, pcs_2d, ev=None,
                         evr=None,
                         rowvar=False,
                         scale_eofs=True, kaiser_normalize=True,
                         unit_normalize=False, reorder_eofs=True,
                         tol=1e-6, max_iter=500):
    if rowvar:
        raise NotImplementedError('rowvar=True not implemented')

    if rowvar:
        n_features, n_components = eofs_2d.shape
    else:
        n_components, n_features = eofs_2d.shape

    if ev is not None:
        _check_array_shape(ev, (n_components,),
                           '_varimax_rotation_2d')
    if evr is not None:
        _check_array_shape(evr, (n_components,),
                           '_varimax_rotation_2d')

    pcs = pcs_2d.copy()
    eofs = eofs_2d.copy()

    if scale_eofs and ev is None:
        warnings.warn('scale_eofs=True but explained variance not given',
                      UserWarning)
    elif scale_eofs and ev is not None:
        eofs *= np.sqrt(ev[:, np.newaxis])

    if kaiser_normalize:
        normalization = np.sqrt((eofs ** 2).sum(axis=0))
        normalization[normalization == 0] = 1
        eofs /= normalization

    if rowvar:
        rotT, n_iter = _bsv_orthomax_rotation(
            eofs.T, gamma=VARIMAX_GAMMA, tol=tol, max_iter=max_iter)
    else:
        rot, n_iter = _bsv_orthomax_rotation(
            eofs, gamma=VARIMAX_GAMMA, tol=tol, max_iter=max_iter)

    rpcs = np.dot(pcs, rot)
    reofs = np.dot(rot.T, eofs)

    if kaiser_normalize:
        reofs *= normalization

    rpcs, reofs = _fix_sign_convention(rpcs, reofs)

    rev = np.sum(reofs ** 2, axis=1)

    if reorder_eofs:
        rpcs, reofs, rev = _reorder_eofs(rpcs, reofs, rev)

    if unit_normalize:
        rpcs, reofs = _unit_normalize_eofs(rpcs, reofs)

    if ev is not None and evr is not None:
        total_variance = ev[0] / evr[0]
        revr = rev / total_variance
    else:
        revr = None

    return reofs, rpcs, rev, revr, n_iter


def _varimax_rotation_default(eofs, pcs, ev=None,
                              evr=None, scale_eofs=True,
                              kaiser_normalize=True, unit_normalize=False,
                              reorder_eofs=True,
                              tol=1e-6, max_iter=500, rowvar=False):

    if rowvar:
        raise NotImplementedError('rowvar=True not implemented')

    if eofs.ndim < 2:
        raise ValueError(
            'Matrix with at least two dimensions expected '
            '(got eofs.ndim = %d)' % eofs.ndim)

    original_shape = eofs.shape[1:]
    eofs_2d = _to_2d_array(eofs, rowvar=rowvar)

    reofs, rpcs, rev, revr, n_iter = _varimax_rotation_2d(
        eofs_2d, pcs, ev=ev, evr=evr,
        scale_eofs=scale_eofs, kaiser_normalize=kaiser_normalize,
        unit_normalize=unit_normalize, reorder_eofs=reorder_eofs,
        tol=tol, max_iter=max_iter)

    reofs = _to_nd_array(reofs, original_shape, rowvar=rowvar)

    return reofs, rpcs, rev, revr, n_iter
