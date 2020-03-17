"""Set-up routines for helper package providing climate indices."""

from setuptools import setup, find_packages


setup(
    name='climate_indices',
    version='0.0.1',
    author='Dylan.Harries',
    author_email='Dylan.Harries@csiro.au',
    description='Helper utilities for calculating climate indices',
    long_description='',
    install_requires=['dask', 'geopandas', 'numpy', 'pytest', 'regionmask', 'scipy', 'xarray'],
    setup_requires=['pytest-runner', 'pytest-pylint'],
    tests_require=['pytest', 'pytest-cov', 'pylint'],
    packages=find_packages('src'),
    package_dir={'':'src'},
    test_suite='tests',
    zip_safe=False
)
