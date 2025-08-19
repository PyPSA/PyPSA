import pytest

from pypsa.version import check_pypsa_version, parse_version_tuple


def test_version_check(caplog):
    caplog.clear()
    check_pypsa_version("0.20.0")
    assert caplog.text == ""

    caplog.clear()
    check_pypsa_version("0.0")
    assert "The correct version of PyPSA could not be resolved" in caplog.text


@pytest.mark.parametrize(
    ("version_str", "expected"),
    [
        ("1.0.0", (1, 0, 0)),
        ("1.0.0rc1", (1, 0, 0, "rc1")),
        ("2.1a2", (2, 1, "a2")),
        ("0.34.0", (0, 34, 0)),
        ("0.34.0b1", (0, 34, 0, "b1")),
    ],
)
def test_parse_version_tuple(version_str, expected):
    assert parse_version_tuple(version_str) == expected
