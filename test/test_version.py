import pytest

from pypsa.version import check_pypsa_version


def test_version_check(caplog):
    check_pypsa_version("0.20.0")
    assert caplog.text == ""

    check_pypsa_version("0.0")
    assert "The correct version of PyPSA could not be resolved" in caplog.text


def test_deprecation():
    with pytest.warns(DeprecationWarning):
        from pypsa import release_version  # noqa
