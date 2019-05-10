import numpy as np
import unittest

from sklearn.utils import check_random_state

from climate_indices.eofs import calc_eofs, varimax_rotation


def _random_anomalies(size, low=-10.0, high=10.0, random_state=None):
    rng = check_random_state(random_state)

    X = rng.uniform(low=low, high=high, size=size)
    anom = X - X.mean(axis=0)

    return anom


class TestStandardEOFs(unittest.TestCase):

    def test_standard_eof_reconstruction_2d(self):
        random_seed = 0
        rng = np.random.RandomState(seed=random_seed)

        n_samples = 100
        n_features = 7

        X = _random_anomalies((n_samples, n_features),
                              random_state=rng)

        pcs, eofs = calc_eofs(X, random_state=rng)[:2]

        X_rec = np.dot(pcs, eofs)

        self.assertTrue(np.allclose(X_rec, X))

    def test_standard_eof_reconstruction_3d(self):
        random_seed = 0
        rng = np.random.RandomState(seed=random_seed)

        n_samples = 200
        n_x_features = 3
        n_y_features = 8
        n_features = n_x_features * n_y_features

        X = _random_anomalies((n_samples, n_x_features, n_y_features),
                              random_state=rng)

        pcs, eofs = calc_eofs(X, n_eofs=n_features,
                              random_state=rng)[:2]

        X_rec = np.tensordot(pcs, eofs, axes=(-1, 0))

        self.assertTrue(np.allclose(X_rec, X))


class TestVarimaxRotation(unittest.TestCase):

    def test_convergence_2d(self):
        random_seed = 0
        rng = np.random.RandomState(seed=random_seed)

        n_samples = 30
        n_features = 4
        max_iter = 100

        X = _random_anomalies((n_samples, n_features),
                              random_state=rng)

        pcs, eofs = calc_eofs(X, random_state=rng)[:2]

        rpcs, reofs, n_iter = varimax_rotation(pcs, eofs, max_iter=max_iter)

        self.assertTrue(n_iter < max_iter)

    def test_rotated_eof_reconstruction_2d(self):
        random_seed = 0
        rng = np.random.RandomState(seed=random_seed)

        n_samples = 150
        n_features = 9
        max_iter = 100

        X = _random_anomalies((n_samples, n_features),
                              random_state=rng)

        pcs, eofs = calc_eofs(X, random_state=rng)[:2]

        rpcs, reofs, _ = varimax_rotation(pcs, eofs, max_iter=max_iter)

        X_rec = np.dot(rpcs, reofs)

        self.assertTrue(np.allclose(X_rec, X))

    def test_rotated_eof_reconstruction_3d(self):
        random_seed = 0
        rng = np.random.RandomState(seed=random_seed)

        n_samples = 90
        n_x_features = 9
        n_y_features = 4
        n_features = n_x_features * n_y_features

        X = _random_anomalies((n_samples, n_x_features, n_y_features),
                              random_state=rng)

        pcs, eofs = calc_eofs(X, n_eofs=n_features, random_state=rng)[:2]

        rpcs, reofs, _ = varimax_rotation(pcs, eofs)

        X_rec = np.tensordot(rpcs, reofs, axes=(-1, 0))

        self.assertTrue(np.allclose(X_rec, X))
