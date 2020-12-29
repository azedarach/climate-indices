"""Test setting package version."""

# License: MIT


import climate_indices as ci


def test_version_attr():
    """Test module has __version__ attribute."""
    assert hasattr(ci, '__version__')
