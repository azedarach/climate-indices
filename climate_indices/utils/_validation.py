import numpy as np


def _check_array_shape(A, shape, whom):
    if np.shape(A) != shape:
        raise ValueError(
            'Array with wrong shape passed to %s. '
            'Expected %s, but got %s' % (whom, shape, np.shape(A)))


def _check_matching_lengths(time, data, whom):
    n_times = time.shape[0]
    n_samples = data.shape[0]
    if n_times != n_samples:
        raise ValueError(
            'Mismatch in number of records passed to %s. '
            'Number of times is %d, but number of records is %d.' %
            (whom, n_times, n_samples))
