from ._eofs_dask_backend import _calc_eofs_dask
from ._eofs_default_backend import (_calc_eofs_default,
                                    _calc_heofs_default,
                                    _varimax_rotation_default)

try:
    import dask.array
    has_dask = True
except ImportError:
    has_dask = False


def calc_eofs(X, backend=None, **kwargs):
    if backend is None:
        backend = 'default'

    if backend == 'default':
        return _calc_eofs_default(X, **kwargs)
    elif backend == 'dask':
        if not has_dask:
            raise ValueError(
                "backend 'dask' cannot be used as "
                "dask could not be imported")
        return _calc_eofs_dask(X, **kwargs)
    else:
        raise ValueError(
            "invalid backend parameter '%r'" % backend)


def calc_heofs(X, backend=None, **kwargs):
    if backend is None:
        backend = 'default'

    if backend == 'default':
        return _calc_heofs_default(X, **kwargs)
    else:
        raise ValueError(
            "invalid backend parameter '%r'" % backend)


def varimax_rotation(eofs, pcs, backend=None, **kwargs):
    if backend is None:
        backend = 'default'

    if backend == 'default':
        return _varimax_rotation_default(eofs, pcs, **kwargs)
    else:
        raise ValueError(
            "invalid backend parameter '%r'" % backend)
