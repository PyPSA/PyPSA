# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import pytest

import pypsa
from pypsa.version import check_pypsa_version


def test_version_check(caplog):
    caplog.clear()
    check_pypsa_version("0.20.0")
    assert caplog.text == ""

    caplog.clear()
    check_pypsa_version("0.0")
    assert "The correct version of PyPSA could not be resolved" in caplog.text


def test_version_deprecations():
    """Test that deprecated version attributes raise DeprecationWarning."""
    # Test deprecated renamed attributes
    with pytest.raises(
        DeprecationWarning,
        match="pypsa.__version_semver__ is deprecated. Use pypsa.__version_base__ instead.",
    ):
        _ = pypsa.__version_semver__

    with pytest.raises(
        DeprecationWarning,
        match="pypsa.__version_short__ is deprecated. Use pypsa.__version_major_minor__ instead.",
    ):
        _ = pypsa.__version_short__

    # Test removed tuple attributes
    with pytest.raises(
        DeprecationWarning, match="pypsa.__version_semver_tuple__ has been removed"
    ):
        _ = pypsa.__version_semver_tuple__

    with pytest.raises(
        DeprecationWarning, match="pypsa.__version_short_tuple__ has been removed"
    ):
        _ = pypsa.__version_short_tuple__

    # Test that new attributes work
    assert isinstance(pypsa.__version_base__, str)
    assert isinstance(pypsa.__version_major_minor__, str)
