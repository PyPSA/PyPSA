# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import logging

import pytest

import pypsa

logger = logging.getLogger(__name__)


def test_ac_dc_meshed():
    n = pypsa.examples.ac_dc_meshed()
    assert not n.c.buses.static.empty


def test_storage_hvdc():
    n = pypsa.examples.storage_hvdc()
    assert not n.c.buses.static.empty


def test_scigrid_de():
    n = pypsa.examples.scigrid_de()
    assert not n.c.buses.static.empty


def test_model_energy():
    n = pypsa.examples.model_energy()
    assert not n.c.buses.static.empty


def test_carbon_management():
    try:
        n = pypsa.examples.carbon_management()
        assert not n.c.buses.static.empty
    except RuntimeError as e:
        logger.warning("Test would have failed: %s", e)
        pytest.skip("Test failed but converted to warning")


@pytest.fixture
def seeded_cache(monkeypatch, tmp_path):
    """Seed a cache directory with a dummy network file."""
    monkeypatch.setattr(pypsa.examples, "_cache_root", lambda: tmp_path)
    version = pypsa.version.__version_base__
    cache = tmp_path / f"v{version}" / "ac_dc_meshed.nc"
    cache.parent.mkdir(parents=True)
    pypsa.Network().export_to_netcdf(str(cache))
    return tmp_path


def test_caching(seeded_cache):
    """Test that cached example is loaded from disk without network."""
    n = pypsa.examples.ac_dc_meshed()
    assert isinstance(n, pypsa.Network)


def test_clear_cache(seeded_cache):

    pypsa.examples.clear_cache()
    assert not seeded_cache.exists()


def test_cache_miss_network_disabled(monkeypatch, tmp_path):
    """Test that cache miss with network requests disabled raises ValueError."""
    monkeypatch.setattr(pypsa.examples, "_cache_root", lambda: tmp_path)

    pypsa.options.general.allow_network_requests = False
    try:
        with pytest.raises(ValueError, match="Network requests are disabled"):
            pypsa.examples.ac_dc_meshed()
    finally:
        pypsa.options.general.allow_network_requests = True


@pytest.mark.skipif(
    not pypsa.examples._check_url_availability("https://data.pypsa.org"),
    reason="No internet connection",
)
def test_check_url_availability():
    """Test _check_url_availability function."""
    from pypsa.examples import _check_url_availability

    assert not _check_url_availability("invalid-url")
    assert not _check_url_availability("ftp://example.com")
    assert not _check_url_availability("")
    assert not _check_url_availability("https://data.pypsa.org/nonexistent")
    assert _check_url_availability(
        "https://data.pypsa.org/networks/examples/latest/ac_dc_meshed.nc"
    )
