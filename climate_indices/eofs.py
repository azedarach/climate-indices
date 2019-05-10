import numpy as np
import warnings

from .utils._validation import _check_array_shape

from sklearn.decomposition import PCA


EOF_RESULTS_KEYS = [
    'pcs', 'eofs', 'explained_variance', 'explained_variance_ratio',
    'singular_values']

VARIMAX_GAMMA = 1.0


def _to_2d_array(X):
    if X.ndim < 2:
        raise ValueError(
            'array with fewer than 2 dimension passed to _to_2d_array')
    elif X.ndim == 2:
        return X

    n_samples = X.shape[0]
    n_features = np.product(X.shape[1:])

    return np.reshape(X, (n_samples, n_features))


def _to_nd_array(X, target_shape):
    if X.ndim != 2:
        raise ValueError(
            'array with number of dimensions other than '
            '2 passed to _to_nd_array')

    n_samples, n_features = X.shape
    n_target_features = np.product(target_shape)

    if n_target_features != n_features:
        raise ValueError(
            'number of elements of target shape does not match '
            'number of elements in given array '
            '(got target_shape = %r, but X.shape[1] = %d' %
            (target_shape, X.shape[1]))

    return np.reshape(X, (n_samples,) + target_shape)


def _calc_eofs_2d(X, n_eofs=None, random_state=None):
    pca = PCA(n_components=n_eofs, random_state=random_state, copy=True)

    pcs = pca.fit_transform(X)
    eofs_2d = pca.components_
    ev = pca.explained_variance_
    evr = pca.explained_variance_ratio_
    sv = pca.singular_values_

    return pcs, eofs_2d, ev, evr, sv


def calc_eofs(X, n_eofs=None, weights=None, random_state=None):
    """Calculate standard EOFs of given data.

    Parameters
    ----------
    X : array-like, shape (n_samples, ...)
        Data array with at least two dimensions, of which the left-most
        must correspond to different data points.

    n_eofs : None or integer
        The number of EOFs to retain. If None, then the maximum
        possible number of EOFs will be retained.

    weights : None or array-like, shape X.shape[1:]
        If given, an array of weights to be applied to the data.

    random_state : integer, RandomState, or None
        If an integer, random_state is the seed used by the
        random number generator. If a RandomState instance,
        random_state is the random number generator. If
        None, the random number generator is the
        RandomState instance used by `np.random`.

    Returns
    -------
    results : dict
        A dictionary containing the following keys:

        - 'pcs' : array-like, shape (n_samples, n_eofs)
            An array containing the principal component time-series for
            each EOF.

        - 'eofs' : array-like, shape (n_eofs,) + X.shape[1:]
            An array containing the EOFs of the data.

        - 'explained_variance' : array-like, shape (n_eofs)
            An array containing the variance explained by each EOF mode.

        - 'explained_variance_ratio' : array-like, shape (n_eofs)
            An array containing the fraction of the total variance
            explained by each EOF mode.

        - 'singular_values' : array-like, shape (n_eofs)
            An array containing the singular values of the data array
            X associated with each EOF mode.
    """
    if X.ndim < 2:
        raise ValueError(
            'Matrix with at least two dimensions expected '
            '(got X.ndim = %d)' % X.ndim)

    original_shape = X.shape[1:]

    if weights is not None:
        _check_array_shape(weights, original_shape, 'calc_eofs')
        data = weights * X
    else:
        data = X

    data_2d = _to_2d_array(data)

    pcs, eofs_2d, ev, evr, sv = _calc_eofs_2d(
        data_2d, n_eofs=n_eofs, random_state=random_state)

    eofs = _to_nd_array(eofs_2d, original_shape)

    results = {'pcs': pcs, 'eofs': eofs, 'explained_variance': ev,
               'explained_variance_ratio': evr,
               'singular_values': sv}

    return results


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


def _unit_normalize_eofs(pcs, eofs):
    norms = np.linalg.norm(eofs, axis=1)
    return pcs * norms[np.newaxis, :], eofs / norms[:, np.newaxis]


def _fix_sign_convention(pcs, eofs, singular_values):
    abs_min = np.abs(np.min(eofs, axis=1))
    abs_max = np.abs(np.max(eofs, axis=1))

    flip_signs = abs_min > abs_max

    eofs[flip_signs] *= -1
    pcs[:, flip_signs] *= -1

    sv = singular_values.copy()
    sv[:, flip_signs] *= -1
    sv[flip_signs, :] *= -1

    return pcs, eofs, sv


def _reorder_eofs(pcs, eofs, explained_variance, singular_values):
    n_eofs = np.size(explained_variance)
    sort_order = np.flip(np.argsort(explained_variance))

    perm_inv = np.zeros((n_eofs, n_eofs))
    for i in range(n_eofs):
        perm_inv[i, sort_order[i]] = 1

    sv = np.dot(perm_inv, np.dot(singular_values, perm_inv.T))

    return (pcs[:, sort_order], eofs[sort_order],
            explained_variance[sort_order],
            sv)


def _varimax_rotation_2d(pcs_2d, eofs_2d, explained_variance,
                         explained_variance_ratio,
                         singular_values,
                         scale_eofs=True, kaiser_normalize=True,
                         unit_normalize=False, reorder_eofs=True,
                         tol=1e-6, max_iter=500):
    n_eofs, n_features = eofs_2d.shape

    _check_array_shape(explained_variance, (n_eofs,), '_varimax_rotation_2d')
    _check_array_shape(explained_variance_ratio, (n_eofs,),
                       '_varimax_rotation_2d')
    _check_array_shape(singular_values, (n_eofs,), '_varimax_rotation_2d')

    pcs = pcs_2d.copy()
    eofs = eofs_2d.copy()

    if scale_eofs:
        eofs *= np.sqrt(explained_variance[:, np.newaxis])

    if kaiser_normalize:
        normalization = np.sqrt((eofs ** 2).sum(axis=0))
        normalization[normalization == 0] = 1
        eofs /= normalization

    rot, n_iter = _bsv_orthomax_rotation(
        eofs, gamma=VARIMAX_GAMMA, tol=tol, max_iter=max_iter)

    rpcs = np.dot(pcs, rot)
    reofs = np.dot(rot.T, eofs)
    rsv = np.dot(rot.T, np.dot(np.diag(singular_values), rot))

    if kaiser_normalize:
        reofs *= normalization

    # @todo since singular values already absorbed into PC scaling,
    # may be incorrect to also apply sign changes to singular values
    # as here
    rpcs, reofs, rsv = _fix_sign_convention(rpcs, reofs, rsv)

    rev = np.sum(reofs ** 2, axis=1)

    if reorder_eofs:
        rpcs, reofs, rev, rsv = _reorder_eofs(rpcs, reofs, rev, rsv)

    if unit_normalize:
        rpcs, reofs = _unit_normalize_eofs(rpcs, reofs)

    total_variance = explained_variance[0] / explained_variance_ratio[0]
    revr = rev / total_variance

    results = {'pcs': rpcs, 'eofs': reofs,
               'explained_variance': rev,
               'explained_variance_ratio': revr,
               'singular_values': rsv,
               'n_iter': n_iter}

    return results


def varimax_rotation(eof_results, scale_eofs=True,
                     kaiser_normalize=True, unit_normalize=False,
                     reorder_eofs=True,
                     tol=1e-6, max_iter=500):
    """Perform VARIMAX rotation on the given PCs and EOFs.

    Parameters
    ----------
    eof_results : dict
        A dictionary with the same keys as that produced
        as output from `calc_eofs`.

    scale_eofs : boolean, default : True
        If True, scale each EOF by the square root
        of the variance associated with that mode.

    kaiser_normalize : boolean, default : True
        If True, normalize the elements of the rotated
        EOFs by their communalities in the VARIMAX
        objective function.

    unit_normalize: boolean, default : False
        If True, scale the calculated rotated EOFs to
        have unit norm before returning them.

    reorder_eofs : boolean, default : True
        If True, sort the rotated EOFs in descending
        order of the variance explained by each rotated
        mode.

    tol : float, default: 1e-4
        Tolerance of the stopping condition.

    max_iter : integer, default: 200
        Maximum number of iterations before stopping.

    Returns
    -------
    results : dict
        A dictionary containing the following keys:

        - 'pcs' : array-like, shape (n_samples, n_eofs)
            An array containing the rotated principal component
            time-series for each rotated EOF.

        - 'eofs' : array-like, shape (n_eofs,) + X.shape[1:]
            An array containing the rotated EOFs of the data.

        - 'explained_variance' : array-like, shape (n_eofs)
            An array containing the variance explained by each
            rotated EOF mode.

        - 'explained_variance_ratio' : array-like, shape (n_eofs)
            An array containing the fraction of the total variance
            explained by each rotated EOF mode.

        - 'singular_values' : array-like, shape (n_eofs)
            An array containing the rotated singular values of
            the original data.

        - 'n_iter' : integer
            The actual number of iterations required to perform
            the VARIMAX rotation.

    References
    ----------
    """
    for key in EOF_RESULTS_KEYS:
        if key not in eof_results:
            raise ValueError(
                "missing key '%s' in EOF results provided "
                "to varimax_rotation" % key)

    pcs = eof_results['pcs']
    eofs = eof_results['eofs']
    ev = eof_results['explained_variance']
    evr = eof_results['explained_variance_ratio']
    sv = eof_results['singular_values']

    if eofs.ndim < 2:
        raise ValueError(
            'Matrix with at least two dimensions expected '
            '(got eofs.ndim = %d)' % eofs.ndim)

    original_shape = eofs.shape[1:]
    eofs_2d = _to_2d_array(eofs)

    results = _varimax_rotation_2d(
        pcs, eofs_2d, ev, evr, sv,
        scale_eofs=scale_eofs, kaiser_normalize=kaiser_normalize,
        unit_normalize=unit_normalize, reorder_eofs=reorder_eofs,
        tol=tol, max_iter=max_iter)

    results['eofs'] = _to_nd_array(results['eofs'], original_shape)

    return results


__all__ = ['calc_eofs', 'varimax_rotation']
