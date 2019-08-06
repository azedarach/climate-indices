import numpy as np
import unittest

import climate_indices.eofs._eofs_default_backend as eofs_default

from sklearn.utils import check_random_state


def _random_anomalies(size, low=-10.0, high=10.0, rowvar=True,
                      random_state=None):
    rng = check_random_state(random_state)

    X = rng.uniform(low=low, high=high, size=size)

    if X.ndim > 1:
        anom = X - np.mean(X, axis=int(bool(rowvar)), keepdims=True)
    else:
        anom = X - np.mean(X, axis=0, keepdims=True)

    return anom


class TestDefaultBackendHelpers(unittest.TestCase):

    def test_2d_to_2d_array_rowvars(self):
        random_seed = 0
        rng = np.random.RandomState(seed=random_seed)

        n_samples = 100
        n_features = 8

        X = _random_anomalies((n_features, n_samples),
                              random_state=rng, rowvar=True)

        X_2d = eofs_default._to_2d_array(X, rowvar=True)

        self.assertTrue(np.all(X == X_2d))

    def test_2d_to_2d_array_colvars(self):
        random_seed = 0
        rng = np.random.RandomState(seed=random_seed)

        n_samples = 50
        n_features = 10

        X = _random_anomalies((n_samples, n_features),
                              random_state=rng, rowvar=False)

        X_2d = eofs_default._to_2d_array(X, rowvar=False)

        self.assertTrue(np.all(X == X_2d))

    def test_1d_to_2d_array_rowvar_fails(self):
        random_seed = 0
        rng = np.random.RandomState(seed=random_seed)

        n_samples = 100

        X = _random_anomalies(n_samples, random_state=rng,
                              rowvar=True)

        with self.assertRaises(ValueError):
            eofs_default._to_2d_array(X, rowvar=True)

        X = _random_anomalies((n_samples,), random_state=rng,
                              rowvar=True)

        with self.assertRaises(ValueError):
            eofs_default._to_2d_array(X, rowvar=True)

    def test_1d_to_2d_array_colvar_fails(self):
        random_seed = 0
        rng = np.random.RandomState(seed=random_seed)

        n_samples = 500

        X = _random_anomalies(n_samples, random_state=rng,
                              rowvar=False)

        with self.assertRaises(ValueError):
            eofs_default._to_2d_array(X, rowvar=False)

        X = _random_anomalies((n_samples,), random_state=rng,
                              rowvar=False)

        with self.assertRaises(ValueError):
            eofs_default._to_2d_array(X, rowvar=False)

    def test_3d_to_2d_array_rowvar(self):
        n_samples = 3
        observation_shape = (2, 3)

        X = np.empty(observation_shape + (n_samples,))
        X[:, :, 0] = np.array([[1, 2, 3], [4, 5, 6]])
        X[:, :, 1] = np.array([[7, 8, 9], [10, 11, 12]])
        X[:, :, 2] = np.array([[13, 14, 15], [16, 17, 18]])

        X_2d = eofs_default._to_2d_array(X, rowvar=True)

        expected_shape = (np.product(observation_shape), n_samples)
        self.assertTrue(X_2d.shape == expected_shape)

        X_2d_expected = np.array(
            [[1, 7, 13],
             [2, 8, 14],
             [3, 9, 15],
             [4, 10, 16],
             [5, 11, 17],
             [6, 12, 18]])

        self.assertTrue(np.all(X_2d == X_2d_expected))

    def test_3d_to_2d_array_colvar(self):
        n_samples = 4
        observation_shape = (2, 3)

        X = np.empty((n_samples,) + observation_shape)
        X[0] = np.array([[1, 2, 3], [4, 5, 6]])
        X[1] = np.array([[7, 8, 9], [10, 11, 12]])
        X[2] = np.array([[13, 14, 15], [16, 17, 18]])
        X[3] = np.array([[19, 20, 21], [22, 23, 24]])

        X_2d = eofs_default._to_2d_array(X, rowvar=False)

        expected_shape = (n_samples, np.product(observation_shape))
        self.assertTrue(X_2d.shape == expected_shape)

        X_2d_expected = np.array(
            [[1, 2, 3, 4, 5, 6],
             [7, 8, 9, 10, 11, 12],
             [13, 14, 15, 16, 17, 18],
             [19, 20, 21, 22, 23, 24]])

        self.assertTrue(np.all(X_2d == X_2d_expected))

    def test_1d_to_nd_array_fails_rowvar(self):
        random_seed = 0
        rng = np.random.RandomState(seed=random_seed)

        n_samples = 10
        target_shape = (2, 5)

        X = _random_anomalies(n_samples, random_state=rng,
                              rowvar=True)

        with self.assertRaises(ValueError):
            eofs_default._to_nd_array(X, target_shape, rowvar=True)

    def test_1d_to_nd_array_fails_colvar(self):
        random_seed = 0
        rng = np.random.RandomState(seed=random_seed)

        n_samples = 50
        target_shape = (10, 5)

        X = _random_anomalies(n_samples, random_state=rng,
                              rowvar=False)

        with self.assertRaises(ValueError):
            eofs_default._to_nd_array(X, target_shape, rowvar=False)

    def test_2d_to_2d_array_rowvar(self):
        random_seed = 0
        rng = np.random.RandomState(seed=random_seed)

        n_samples = 10
        n_features = 2
        target_shape = (n_features,)

        X = _random_anomalies((n_features, n_samples), random_state=rng,
                              rowvar=True)

        X_nd = eofs_default._to_nd_array(X, target_shape, rowvar=True)

        self.assertTrue(np.all(X_nd == X))

    def test_2d_to_2d_array_colvar(self):
        random_seed = 0
        rng = np.random.RandomState(seed=random_seed)

        n_samples = 40
        n_features = 2
        target_shape = (n_features,)

        X = _random_anomalies((n_samples, n_features), random_state=rng,
                              rowvar=False)

        X_nd = eofs_default._to_nd_array(X, target_shape, rowvar=False)

        self.assertTrue(np.all(X_nd == X))

    def test_2d_to_4d_rowvar(self):
        n_samples = 3
        target_shape = (1, 2, 2)

        X = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9],
                      [10, 11, 12]])

        X_nd = eofs_default._to_nd_array(X, target_shape, rowvar=True)

        expected_shape = target_shape + (n_samples,)
        self.assertTrue(X_nd.shape == expected_shape)

        X_nd_expected = np.empty(expected_shape)
        X_nd_expected[:, :, :, 0] = np.array([[[1, 4], [7, 10]]])
        X_nd_expected[:, :, :, 1] = np.array([[[2, 5], [8, 11]]])
        X_nd_expected[:, :, :, 2] = np.array([[[3, 6], [9, 12]]])

        self.assertTrue(np.all(X_nd_expected == X_nd))

    def test_2d_to_3d_colvar(self):
        n_samples = 4
        target_shape = (3, 2)

        X = np.array([[1, 2, 3, 4, 5, 6],
                      [7, 8, 9, 10, 11, 12],
                      [13, 14, 15, 16, 17, 18],
                      [19, 20, 21, 22, 23, 24]])

        X_nd = eofs_default._to_nd_array(X, target_shape, rowvar=False)

        expected_shape = (n_samples,) + target_shape
        self.assertTrue(X_nd.shape == expected_shape)

        X_nd_expected = np.empty(expected_shape)
        X_nd_expected[0] = np.array([[1, 2], [3, 4], [5, 6]])
        X_nd_expected[1] = np.array([[7, 8], [9, 10], [11, 12]])
        X_nd_expected[2] = np.array([[13, 14], [15, 16], [17, 18]])
        X_nd_expected[3] = np.array([[19, 20], [21, 22], [23, 24]])

        self.assertTrue(np.all(X_nd_expected == X_nd))

    def test_center_2d_data_no_missing_rowvar(self):
        random_seed = 0
        rng = np.random.RandomState(seed=random_seed)

        n_samples = 10
        n_features = 2

        X = rng.uniform(size=(n_features, n_samples))

        initial_mean = X.mean(axis=1)
        self.assertTrue(initial_mean.shape == (2,))
        self.assertTrue(np.all(initial_mean != 0))

        X_anom = eofs_default._center_data(X, axis=1,
                                           skipna=False)
        final_mean = X_anom.mean(axis=1)
        self.assertTrue(np.allclose(final_mean, 0))

    def test_center_2d_data_no_missing_colvar(self):
        random_seed = 0
        rng = np.random.RandomState(seed=random_seed)

        n_samples = 34
        n_features = 4

        X = rng.uniform(size=(n_samples, n_features))

        initial_mean = X.mean(axis=0)
        self.assertTrue(initial_mean.shape == (4,))
        self.assertTrue(np.all(initial_mean != 0))

        X_anom = eofs_default._center_data(X, axis=0,
                                           skipna=False)

        final_mean = X_anom.mean(axis=0)
        self.assertTrue(np.allclose(final_mean, 0))

    def test_center_2d_data_missing_rowvar(self):
        X = np.array([[1, 2, np.NaN, 6],
                      [3, 4, 5, 6],
                      [np.NaN, np.NaN, np.NaN, 1]])

        X_anom = eofs_default._center_data(X, axis=1,
                                           skipna=True)

        expected_shape = X.shape
        self.assertTrue(X_anom.shape == expected_shape)

        X_anom_expected = np.array([[-2, -1, np.NaN, 3],
                                    [-1.5, -0.5, 0.5, 1.5],
                                    [np.NaN, np.NaN, np.NaN, 0]])

        self.assertTrue(np.all(np.isnan(X_anom) == np.isnan(X_anom_expected)))

        nonmiss_mask = ~np.isnan(X_anom_expected)
        self.assertTrue(np.all(X_anom_expected[nonmiss_mask] ==
                               X_anom[nonmiss_mask]))

    def test_center_2d_data_missing_colvar(self):
        X = np.array([[3, np.NaN], [np.NaN, 4]])

        X_anom = eofs_default._center_data(X, axis=0, skipna=True)

        expected_shape = X.shape
        self.assertTrue(X_anom.shape == expected_shape)

        X_anom_expected = np.array([[0, np.NaN], [np.NaN, 0]])

        self.assertTrue(np.all(np.isnan(X_anom) == np.isnan(X_anom_expected)))

        nonmiss_mask = ~np.isnan(X_anom_expected)
        self.assertTrue(np.all(X_anom_expected[nonmiss_mask] ==
                               X_anom[nonmiss_mask]))

    def test_fixed_missing_values_check_rowvar(self):
        random_seed = 0
        rng = np.random.RandomState(seed=random_seed)

        n_samples = 100
        n_features = 5

        X = _random_anomalies((n_features, n_samples), random_state=rng,
                              rowvar=True)

        self.assertTrue(eofs_default._has_fixed_missing_values(X, rowvar=True))

        X[1, :] = np.NaN

        self.assertTrue(eofs_default._has_fixed_missing_values(X, rowvar=True))

        X[1, :] = rng.uniform(size=(n_samples,))
        X[:, 4] = np.NaN

        self.assertFalse(eofs_default._has_fixed_missing_values(
            X, rowvar=True))

    def test_fixed_missing_values_check_colvar(self):
        random_seed = 0
        rng = np.random.RandomState(seed=random_seed)

        n_samples = 40
        n_features = 10

        X = _random_anomalies((n_samples, n_features), random_state=rng,
                              rowvar=False)

        self.assertTrue(eofs_default._has_fixed_missing_values(
            X, rowvar=False))

        X[1, :] = np.NaN

        self.assertFalse(eofs_default._has_fixed_missing_values(
            X, rowvar=False))

        X[1, :] = rng.uniform(size=(n_features,))

        X[:, 2] = np.NaN

        self.assertTrue(eofs_default._has_fixed_missing_values(
            X, rowvar=False))

    def test_get_valid_variables_rowvar(self):
        random_seed = 0
        rng = np.random.RandomState(seed=random_seed)

        n_samples = 50
        n_features = 7

        X = _random_anomalies((n_features, n_samples), random_state=rng,
                              rowvar=True)

        X_valid, valid_mask = eofs_default._get_valid_variables(
            X, rowvar=True)

        self.assertTrue(np.all(X == X_valid))
        self.assertTrue(np.all(valid_mask == np.arange(n_features)))

        X[5, :] = np.NaN

        X_valid, valid_mask = eofs_default._get_valid_variables(
            X, rowvar=True)

        X_expected = np.empty((n_features - 1, n_samples))
        X_expected[0] = X[0]
        X_expected[1] = X[1]
        X_expected[2] = X[2]
        X_expected[3] = X[3]
        X_expected[4] = X[4]
        X_expected[5] = X[6]

        self.assertTrue(np.all(X_expected == X_valid))
        self.assertTrue(np.all(valid_mask == np.array([0, 1, 2, 3, 4, 6])))

    def test_get_valid_variables_colvar(self):
        random_seed = 0
        rng = np.random.RandomState(seed=random_seed)

        n_samples = 40
        n_features = 4

        X = _random_anomalies((n_samples, n_features), random_state=rng,
                              rowvar=False)

        X_valid, valid_mask = eofs_default._get_valid_variables(
            X, rowvar=False)

        self.assertTrue(np.all(X == X_valid))
        self.assertTrue(np.all(valid_mask == np.arange(n_features)))

        X[:, 0] = np.NaN
        X[:, 3] = np.NaN

        X_valid, valid_mask = eofs_default._get_valid_variables(
            X, rowvar=False)

        X_expected = np.empty((n_samples, n_features - 2))
        X_expected[:, 0] = X[:, 1]
        X_expected[:, 1] = X[:, 2]

        self.assertTrue(np.all(X_expected == X_valid))
        self.assertTrue(np.all(valid_mask == np.array([1, 2])))


class TestStandardEOFsDefaultBackend(unittest.TestCase):

    def test_standard_eof_reconstruction_2d_svd_rowvar(self):
        random_seed = 0
        rng = np.random.RandomState(seed=random_seed)

        n_samples = 100
        n_features = 7

        X = _random_anomalies((n_features, n_samples),
                              random_state=rng, rowvar=True)

        eofs, pcs, ev, evr, _ = eofs_default._calc_eofs_default_svd(
            X, rowvar=True)

        self.assertTrue(eofs.shape == (n_features, n_features))
        self.assertTrue(pcs.shape == (n_features, n_samples))

        X_rec = np.dot(eofs, pcs)

        self.assertTrue(np.allclose(X_rec, X))

    def test_standard_eof_reconstruction_2d_svd_colvar(self):
        random_seed = 0
        rng = np.random.RandomState(seed=random_seed)

        n_samples = 50
        n_features = 5

        X = _random_anomalies((n_samples, n_features), random_state=rng,
                              rowvar=False)

        eofs, pcs, ev, evr, _ = eofs_default._calc_eofs_default_svd(
            X, rowvar=False)

        self.assertTrue(eofs.shape == (n_features, n_features))
        self.assertTrue(pcs.shape == (n_samples, n_features))

        X_rec = np.dot(pcs, eofs)

        self.assertTrue(np.allclose(X_rec, X))

    def test_standard_eof_reconstruction_2d_cov_rowvar(self):
        random_seed = 0
        rng = np.random.RandomState(seed=random_seed)

        n_samples = 60
        n_features = 12

        X = _random_anomalies((n_features, n_samples), random_state=rng,
                              rowvar=True)

        eofs, pcs, ev, evr = eofs_default._calc_eofs_default_cov(
            X, rowvar=True)

        self.assertTrue(eofs.shape == (n_features, n_features))
        self.assertTrue(pcs.shape == (n_features, n_samples))

        X_rec = np.dot(eofs, pcs)

        self.assertTrue(np.allclose(X_rec, X))

    def test_standard_eof_reconstruction_2d_cov_colvar(self):
        random_seed = 0
        rng = np.random.RandomState(seed=random_seed)

        n_samples = 50
        n_features = 5

        X = _random_anomalies((n_samples, n_features), random_state=rng,
                              rowvar=False)

        eofs, pcs, ev, evr = eofs_default._calc_eofs_default_cov(
            X, rowvar=False)

        self.assertTrue(eofs.shape == (n_features, n_features))
        self.assertTrue(pcs.shape == (n_samples, n_features))

        X_rec = np.dot(pcs, eofs)

        self.assertTrue(np.allclose(X_rec, X))

    def test_standard_eof_reconstruction_2d_svd_missing_rowvar(self):
        random_seed = 0
        rng = np.random.RandomState(seed=random_seed)

        n_samples = 68
        n_features = 14

        X = _random_anomalies((n_features, n_samples), random_state=rng,
                              rowvar=True)

        X[1, 4] = np.NaN

        with self.assertRaises(ValueError):
            eofs_default._calc_eofs_default_svd(X, rowvar=True)

        X = _random_anomalies((n_features, n_samples), random_state=rng,
                              rowvar=True)

        X[5, :] = np.NaN

        eofs, pcs, ev, evr, _ = eofs_default._calc_eofs_default_svd(
            X, rowvar=True)

        self.assertTrue(eofs.shape == (n_features, n_features - 1))
        self.assertTrue(pcs.shape == (n_features - 1, n_samples))

        self.assertTrue(np.all(np.isnan(eofs[5])))

        X_rec = np.dot(eofs, pcs)

        nonmiss_mask = ~np.isnan(X_rec)
        self.assertTrue(np.all(np.isnan(X) == np.isnan(X_rec)))
        self.assertTrue(np.allclose(X_rec[nonmiss_mask], X[nonmiss_mask]))

    def test_standard_eof_reconstruction_2d_svd_missing_colvar(self):
        random_seed = 0
        rng = np.random.RandomState(seed=random_seed)

        n_samples = 110
        n_features = 6

        X = _random_anomalies((n_samples, n_features), random_state=rng,
                              rowvar=False)

        X[60, :] = np.NaN

        with self.assertRaises(ValueError):
            eofs_default._calc_eofs_default_svd(X, rowvar=False)

        X = _random_anomalies((n_samples, n_features), random_state=rng,
                              rowvar=False)

        X[:, 0] = np.NaN
        X[:, 4] = np.NaN

        eofs, pcs, ev, evr, _ = eofs_default._calc_eofs_default_svd(
            X, rowvar=False)

        self.assertTrue(eofs.shape == (n_features - 2, n_features))
        self.assertTrue(pcs.shape == (n_samples, n_features - 2))

        self.assertTrue(np.all(np.isnan(eofs[:, 0])))
        self.assertTrue(np.all(np.isnan(eofs[:, 4])))

        X_rec = np.dot(pcs, eofs)

        nonmiss_mask = ~np.isnan(X_rec)
        self.assertTrue(np.all(np.isnan(X) == np.isnan(X_rec)))
        self.assertTrue(np.allclose(X_rec[nonmiss_mask], X[nonmiss_mask]))

    def test_standard_eof_reconstruction_2d_cov_missing_rowvar(self):
        random_seed = 0
        rng = np.random.RandomState(seed=random_seed)

        n_samples = 50
        n_features = 23

        X = _random_anomalies((n_features, n_samples), random_state=rng,
                              rowvar=True)

        X[2, 34] = np.NaN
        X[15, 45] = np.NaN

        eofs, pcs, ev, evr = eofs_default._calc_eofs_default_cov(
            X, rowvar=True)

        self.assertTrue(eofs.shape == (n_features, n_features))
        self.assertTrue(pcs.shape == (n_features, n_samples))
        self.assertTrue(np.all(np.isnan(pcs[:, 34])))
        self.assertTrue(np.all(np.isnan(pcs[:, 45])))

        X_rec = np.dot(eofs, pcs)

        nonmiss_mask = ~np.isnan(X_rec)
        self.assertTrue(np.allclose(X_rec[nonmiss_mask], X[nonmiss_mask]))

        X = _random_anomalies((n_features, n_samples), random_state=rng,
                              rowvar=True)

        X[10, :] = np.NaN

        eofs, pcs, ev, evr = eofs_default._calc_eofs_default_cov(
            X, rowvar=True)

        self.assertTrue(eofs.shape == (n_features, n_features - 1))
        self.assertTrue(pcs.shape == (n_features - 1, n_samples))

        self.assertTrue(np.all(np.isnan(eofs[10])))

        X_rec = np.dot(eofs, pcs)

        nonmiss_mask = ~np.isnan(X_rec)
        self.assertTrue(np.all(np.isnan(X) == np.isnan(X_rec)))
        self.assertTrue(np.allclose(X_rec[nonmiss_mask], X[nonmiss_mask]))

        X = _random_anomalies((n_features, n_samples), random_state=rng,
                              rowvar=True)

        X[19, :] = np.NaN
        X[20, :] = np.NaN
        X[3, 47] = np.NaN
        X[9, 0] = np.NaN

        eofs, pcs, ev, evr = eofs_default._calc_eofs_default_cov(
            X, rowvar=True)

        self.assertTrue(eofs.shape == (n_features, n_features - 2))
        self.assertTrue(pcs.shape == (n_features - 2, n_samples))
        self.assertTrue(np.all(np.isnan(pcs[:, 47])))
        self.assertTrue(np.all(np.isnan(pcs[:, 0])))
        self.assertTrue(np.all(np.isnan(eofs[19])))
        self.assertTrue(np.all(np.isnan(eofs[20])))

        X_rec = np.dot(eofs, pcs)

        nonmiss_mask = ~np.isnan(X_rec)
        self.assertTrue(np.allclose(X_rec[nonmiss_mask], X[nonmiss_mask]))

    def test_standard_eof_reconstruction_2d_cov_missing_colvar(self):
        random_seed = 0
        rng = np.random.RandomState(seed=random_seed)

        n_samples = 200
        n_features = 54

        X = _random_anomalies((n_samples, n_features), random_state=rng,
                              rowvar=False)

        X[34, 10] = np.NaN
        X[35, 10] = np.NaN
        X[36, 10] = np.NaN
        X[2, 32] = np.NaN
        X[150, 32] = np.NaN

        eofs, pcs, ev, evr = eofs_default._calc_eofs_default_cov(
            X, rowvar=False)

        self.assertTrue(eofs.shape == (n_features, n_features))
        self.assertTrue(pcs.shape == (n_samples, n_features))
        self.assertTrue(np.all(np.isnan(pcs[34])))
        self.assertTrue(np.all(np.isnan(pcs[35])))
        self.assertTrue(np.all(np.isnan(pcs[36])))
        self.assertTrue(np.all(np.isnan(pcs[2])))
        self.assertTrue(np.all(np.isnan(pcs[150])))

        X_rec = np.dot(pcs, eofs)

        nonmiss_mask = ~np.isnan(X_rec)
        self.assertTrue(np.allclose(X_rec[nonmiss_mask], X[nonmiss_mask]))

        X = _random_anomalies((n_samples, n_features), random_state=rng,
                              rowvar=False)

        X[:, 10] = np.NaN

        eofs, pcs, ev, evr = eofs_default._calc_eofs_default_cov(
            X, rowvar=False)

        self.assertTrue(eofs.shape == (n_features - 1, n_features))
        self.assertTrue(pcs.shape == (n_samples, n_features - 1))

        self.assertTrue(np.all(np.isnan(eofs[:, 10])))

        X_rec = np.dot(pcs, eofs)

        nonmiss_mask = ~np.isnan(X_rec)
        self.assertTrue(np.all(np.isnan(X) == np.isnan(X_rec)))
        self.assertTrue(np.allclose(X_rec[nonmiss_mask], X[nonmiss_mask]))

        X = _random_anomalies((n_samples, n_features), random_state=rng,
                              rowvar=False)

        X[:, 19] = np.NaN
        X[:, 30] = np.NaN

        X[47, 3] = np.NaN
        X[0, 9] = np.NaN

        eofs, pcs, ev, evr = eofs_default._calc_eofs_default_cov(
            X, rowvar=False)

        self.assertTrue(eofs.shape == (n_features - 2, n_features))
        self.assertTrue(pcs.shape == (n_samples, n_features - 2))
        self.assertTrue(np.all(np.isnan(pcs[47])))
        self.assertTrue(np.all(np.isnan(pcs[0])))
        self.assertTrue(np.all(np.isnan(eofs[:, 19])))
        self.assertTrue(np.all(np.isnan(eofs[:, 30])))

        X_rec = np.dot(pcs, eofs)

        nonmiss_mask = ~np.isnan(X_rec)
        self.assertTrue(np.allclose(X_rec[nonmiss_mask], X[nonmiss_mask]))

    def test_standard_eof_reconstruction_2d_svd_cov_no_missing_agree_rowvar(self):
        random_seed = 0
        rng = np.random.RandomState(seed=random_seed)

        n_samples = 500
        n_features = 10

        X = _random_anomalies((n_features, n_samples), random_state=rng,
                              rowvar=True)

        eofs_svd, pcs_svd, ev_svd, evr_svd, _ = \
            eofs_default._calc_eofs_default_svd(X, rowvar=True)
        eofs_cov, pcs_cov, ev_cov, evr_cov = \
            eofs_default._calc_eofs_default_cov(X, rowvar=True)

        self.assertTrue(np.allclose(np.abs(eofs_svd), np.abs(eofs_cov)))
        self.assertTrue(np.allclose(np.abs(pcs_svd), np.abs(pcs_cov)))

        eofs_svd, pcs_svd, ev_svd, evr_svd, _ = \
            eofs_default._calc_eofs_default_svd(
                X, n_components=10, rowvar=True)
        eofs_cov, pcs_cov, ev_cov, evr_cov = \
            eofs_default._calc_eofs_default_cov(
                X, n_components=10, rowvar=True)

        self.assertTrue(np.allclose(np.abs(eofs_svd), np.abs(eofs_cov)))
        self.assertTrue(np.allclose(np.abs(pcs_svd), np.abs(pcs_cov)))

    def test_standard_eof_reconstruction_2d_svd_cov_no_missing_agree_colvar(self):
        random_seed = 0
        rng = np.random.RandomState(seed=random_seed)

        n_samples = 742
        n_features = 67

        X = _random_anomalies((n_samples, n_features), random_state=rng,
                              rowvar=False)

        eofs_svd, pcs_svd, ev_svd, evr_svd, _ = \
            eofs_default._calc_eofs_default_svd(X, rowvar=False)
        eofs_cov, pcs_cov, ev_cov, evr_cov = \
            eofs_default._calc_eofs_default_cov(X, rowvar=False)

        self.assertTrue(np.allclose(np.abs(eofs_svd), np.abs(eofs_cov)))
        self.assertTrue(np.allclose(np.abs(pcs_svd), np.abs(pcs_cov)))

        eofs_svd, pcs_svd, ev_svd, evr_svd, _ = \
            eofs_default._calc_eofs_default_svd(
                X, n_components=20, rowvar=False)
        eofs_cov, pcs_cov, ev_cov, evr_cov = \
            eofs_default._calc_eofs_default_cov(
                X, n_components=20, rowvar=False)

        self.assertTrue(np.allclose(np.abs(eofs_svd), np.abs(eofs_cov)))
        self.assertTrue(np.allclose(np.abs(pcs_svd), np.abs(pcs_cov)))

# class TestStandardEOFsDefaultBackend(unittest.TestCase):

#     def test_standard_eof_reconstruction_2d_rowvar(self):

# class TestStandardEOFs(unittest.TestCase):

#     def test_standard_eof_reconstruction_2d(self):
#         random_seed = 0
#         rng = np.random.RandomState(seed=random_seed)

#         n_samples = 100
#         n_features = 7

#         X = _random_anomalies((n_samples, n_features),
#                               random_state=rng)

#         pcs, eofs = calc_eofs(X, random_state=rng)[:2]

#         X_rec = np.dot(pcs, eofs)

#         self.assertTrue(np.allclose(X_rec, X))

#     def test_standard_eof_reconstruction_3d(self):
#         random_seed = 0
#         rng = np.random.RandomState(seed=random_seed)

#         n_samples = 200
#         n_x_features = 3
#         n_y_features = 8
#         n_features = n_x_features * n_y_features

#         X = _random_anomalies((n_samples, n_x_features, n_y_features),
#                               random_state=rng)

#         pcs, eofs = calc_eofs(X, n_eofs=n_features,
#                               random_state=rng)[:2]

#         X_rec = np.tensordot(pcs, eofs, axes=(-1, 0))

#         self.assertTrue(np.allclose(X_rec, X))


# class TestVarimaxRotation(unittest.TestCase):

#     def test_convergence_2d(self):
#         random_seed = 0
#         rng = np.random.RandomState(seed=random_seed)

#         n_samples = 30
#         n_features = 4
#         max_iter = 100

#         X = _random_anomalies((n_samples, n_features),
#                               random_state=rng)

#         pcs, eofs = calc_eofs(X, random_state=rng)[:2]

#         rpcs, reofs, n_iter = varimax_rotation(pcs, eofs, max_iter=max_iter)

#         self.assertTrue(n_iter < max_iter)

#     def test_rotated_eof_reconstruction_2d(self):
#         random_seed = 0
#         rng = np.random.RandomState(seed=random_seed)

#         n_samples = 150
#         n_features = 9
#         max_iter = 100

#         X = _random_anomalies((n_samples, n_features),
#                               random_state=rng)

#         pcs, eofs = calc_eofs(X, random_state=rng)[:2]

#         rpcs, reofs, _ = varimax_rotation(pcs, eofs, max_iter=max_iter)

#         X_rec = np.dot(rpcs, reofs)

#         self.assertTrue(np.allclose(X_rec, X))

#     def test_rotated_eof_reconstruction_3d(self):
#         random_seed = 0
#         rng = np.random.RandomState(seed=random_seed)

#         n_samples = 90
#         n_x_features = 9
#         n_y_features = 4
#         n_features = n_x_features * n_y_features

#         X = _random_anomalies((n_samples, n_x_features, n_y_features),
#                               random_state=rng)

#         pcs, eofs = calc_eofs(X, n_eofs=n_features, random_state=rng)[:2]

#         rpcs, reofs, _ = varimax_rotation(pcs, eofs)

#         X_rec = np.tensordot(rpcs, reofs, axes=(-1, 0))

#         self.assertTrue(np.allclose(X_rec, X))
