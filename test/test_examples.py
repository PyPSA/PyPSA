import pypsa


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
    n = pypsa.examples.carbon_management()
    assert not n.c.buses.static.empty


def test_check_url_availability():
    """Test _check_url_availability function."""
    from pypsa.examples import _check_url_availability

    # Test invalid URL formats
    assert not _check_url_availability("invalid-url")
    assert not _check_url_availability("ftp://example.com")
    assert not _check_url_availability("")
    assert not _check_url_availability("https://google.com/invalid-url")

    # Test valid URL format (should return True for valid URLs)
    assert _check_url_availability("https://google.com")
    assert _check_url_availability("https://google.com/search?q=pypsa")
