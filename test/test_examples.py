import pypsa


def test_ac_dc_meshed():
    n = pypsa.examples.ac_dc_meshed()
    assert not n.buses.empty


def test_storage_hvdc():
    n = pypsa.examples.storage_hvdc()
    assert not n.buses.empty


def test_scigrid_de():
    n = pypsa.examples.scigrid_de()
    assert not n.buses.empty


def test_model_energy():
    n = pypsa.examples.model_energy()
    assert not n.buses.empty


def test_carbon_management():
    n = pypsa.examples.carbon_management()
    assert not n.buses.empty


def test_check_url_availability():
    """Test _check_url_availability function."""
    from pypsa.examples import _check_url_availability

    # Test invalid URL formats
    assert not _check_url_availability("invalid-url")
    assert not _check_url_availability("ftp://example.com")
    assert not _check_url_availability("")

    # Test valid URL format (should return True for valid URLs)
    assert _check_url_availability("https://httpbin.org/status/200")

    # Test 404 response (should return False)
    assert not _check_url_availability("https://httpbin.org/status/404")
