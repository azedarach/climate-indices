"""
Provides routines for testing EOF routines.
"""


import unittest

import dask
import dask.array as da
import numpy as np

from climate_indices.utils import eofs, reofs


class TestEOFsWithNumPyArrays(unittest.TestCase):
    """Provides unit tests for EOF routines using NumPy arrays."""

    def test_numpy_random_data_correctly_reconstructed(self):
        """Test random NumPy array is correctly factorized."""

        n_samples = 15
        n_features = 6

        random_data = np.random.normal(size=(n_samples, n_features))

        eofs_ds = eofs(random_data)

        self.assertTrue(eofs_ds['PCs'].shape == (n_samples, n_features))
        self.assertTrue(eofs_ds['EOFs'].shape == (n_features, n_features))
        self.assertTrue(
            np.allclose(np.dot(eofs_ds['PCs'], eofs_ds['EOFs']), random_data))

        reofs_ds = reofs(random_data)

        self.assertTrue(reofs_ds['PCs'].shape == (n_samples, n_features))
        self.assertTrue(reofs_ds['EOFs'].shape == (n_features, n_features))
        self.assertTrue(
            np.allclose(np.dot(reofs_ds['PCs'], reofs_ds['EOFs']), random_data))

        self.assertTrue(
            np.abs(eofs_ds['lambdas'].sum() - reofs_ds['lambdas'].sum()) < 1e-6)

    def test_numpy_multidimensional_random_data_correctly_reconstructed(self):
        """Test NumPy array with more than one spatial dimension is correctly factorized."""

        n_samples = 15
        n_features = (3, 4, 1)

        random_data = np.random.normal(size=((n_samples,) + n_features))

        eofs_ds = eofs(random_data)

        self.assertTrue(
            eofs_ds['PCs'].shape == (n_samples, np.product(n_features)))
        self.assertTrue(
            eofs_ds['EOFs'].shape == (np.product(n_features),) + n_features)
        self.assertTrue(
            np.allclose(eofs_ds['PCs'].dot(eofs_ds['EOFs']).values,
                        random_data))

        reofs_ds = reofs(random_data)

        self.assertTrue(
            reofs_ds['PCs'].shape == (n_samples, np.product(n_features)))
        self.assertTrue(
            reofs_ds['EOFs'].shape == (np.product(n_features),) + n_features)
        self.assertTrue(
            np.allclose(reofs_ds['PCs'].dot(reofs_ds['EOFs']).values,
                        random_data))

        self.assertTrue(
            np.abs(eofs_ds['lambdas'].sum() - reofs_ds['lambdas'].sum()) < 1e-6)

    def test_numpy_list_of_random_data_correctly_reconstructed(self):
        """Test multiple NumPy arrays are correctly factorized."""

        n_samples = 20
        n_features = [3, 3, 2]

        random_data = [np.random.normal(size=(n_samples, n))
                       for n in n_features]

        eofs_ds = eofs(*random_data)

        self.assertTrue(
            eofs_ds[0]['PCs'].shape == (n_samples, np.sum(n_features)))
        self.assertTrue(len(eofs_ds) == len(n_features))

        for i, eof in enumerate(eofs_ds):
            self.assertTrue(np.allclose(eof['PCs'].dot(eof['EOFs']).values,
                                        random_data[i]))

        reofs_ds = reofs(*random_data)

        self.assertTrue(
            reofs_ds[0]['PCs'].shape == (n_samples, np.sum(n_features)))
        self.assertTrue(len(reofs_ds) == len(n_features))

        for i, reof in enumerate(reofs_ds):
            self.assertTrue(
                np.allclose(reof['PCs'].dot(reof['EOFs']).values,
                            random_data[i]))

        self.assertTrue(
            np.abs(eofs_ds[0]['lambdas'].sum() - reofs_ds[0]['lambdas'].sum()) < 1e-6)


class TestEOFsWithDaskArrays(unittest.TestCase):
    """Provides unit tests for EOF routines using Dask arrays."""

    def test_dask_random_data_correctly_reconstructed(self):
        """Test random Dask array is correctly factorized."""

        with dask.config.set(scheduler='synchronous'):
            n_samples = 10
            n_features = 6

            random_data = da.from_array(
                np.random.normal(size=(n_samples, n_features)))

            eofs_ds = eofs(random_data)

            self.assertTrue(eofs_ds['PCs'].shape == (n_samples, n_features))
            self.assertTrue(eofs_ds['EOFs'].shape == (n_features, n_features))
            self.assertTrue(
                da.allclose(eofs_ds['PCs'].dot(eofs_ds['EOFs']).values,
                            random_data))

            reofs_ds = reofs(random_data)

            self.assertTrue(reofs_ds['PCs'].shape == (n_samples, n_features))
            self.assertTrue(reofs_ds['EOFs'].shape == (n_features, n_features))
            self.assertTrue(
                da.allclose(reofs_ds['PCs'].dot(reofs_ds['EOFs']).values,
                            random_data))

    def test_dask_multidimensional_random_data_correctly_reconstructed(self):
        """Test Dask array with more than one spatial dimension is correctly factorized."""

        with dask.config.set(scheduler='synchronous'):
            n_samples = 25
            n_features = (3, 2, 1, 2)

            random_data = da.from_array(
                np.random.normal(size=((n_samples,) + n_features)))

            eofs_ds = eofs(random_data)

            self.assertTrue(
                eofs_ds['PCs'].shape == (n_samples, np.product(n_features)))
            self.assertTrue(
                eofs_ds['EOFs'].shape == (np.product(n_features),) + n_features)
            self.assertTrue(
                da.allclose(eofs_ds['PCs'].dot(eofs_ds['EOFs']),
                            random_data))

            reofs_ds = reofs(random_data)

            self.assertTrue(
                reofs_ds['PCs'].shape == (n_samples, np.product(n_features)))
            self.assertTrue(
                reofs_ds['EOFs'].shape == (np.product(n_features),) + n_features)
            self.assertTrue(
                da.allclose(reofs_ds['PCs'].dot(reofs_ds['EOFs']),
                            random_data))

    def test_list_of_random_data_correctly_reconstructed(self):
        """Test multiple Dask arrays are correctly factorized."""

        with dask.config.set(scheduler='synchronous'):
            n_samples = 15
            n_features = [4, 1, 3, 2, 2]

            random_data = [da.from_array(np.random.normal(size=(n_samples, n)))
                           for n in n_features]

            eofs_ds = eofs(*random_data)

            self.assertTrue(
                eofs_ds[0]['PCs'].shape == (n_samples, np.sum(n_features)))
            self.assertTrue(len(eofs_ds) == len(n_features))

            for i, eof in enumerate(eofs_ds):
                self.assertTrue(da.allclose(eof['PCs'].dot(eof['EOFs']).values,
                                            random_data[i]))

            reofs_ds = reofs(*random_data)

            self.assertTrue(
                reofs_ds[0]['PCs'].shape == (n_samples, np.sum(n_features)))
            self.assertTrue(len(reofs_ds) == len(n_features))

            for i, reof in enumerate(reofs_ds):
                self.assertTrue(
                    da.allclose(reof['PCs'].dot(reof['EOFs']).values,
                                random_data[i]))
