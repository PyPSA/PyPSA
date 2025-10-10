# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import builtins
import importlib.util
from unittest.mock import patch

import pytest

from pypsa.plot.maps.common import _is_cartopy_available


@pytest.fixture
def hide_pytables(monkeypatch):
    import_orig = builtins.__import__

    def mocked_import(name, *args, **kwargs):
        if name == "tables":
            raise ImportError()
        return import_orig(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mocked_import)


@pytest.mark.usefixtures("hide_pytables")
def test_message(ac_dc_network, hide_pytables):
    n = ac_dc_network

    with pytest.raises(
        ImportError,
        match=r"Missing optional dependencies to use HDF5 files\.",
    ):
        n.export_to_hdf5("test.h5")


class TestDynamicDependencyChecking:
    """Test dynamic dependency checking (fixes for GitHub issue #1341)."""

    def test_cartopy_availability_is_dynamic(self):
        """Test that cartopy availability is checked dynamically."""
        actual_availability = importlib.util.find_spec("cartopy") is not None
        assert _is_cartopy_available() == actual_availability

    def test_cartopy_checking_not_cached(self):
        """Test that cartopy checking responds to changes."""
        with patch(
            "pypsa.plot.maps.common.importlib.util.find_spec", return_value=None
        ):
            assert _is_cartopy_available() is False

        with patch(
            "pypsa.plot.maps.common.importlib.util.find_spec", return_value="mock"
        ):
            assert _is_cartopy_available() is True

    def test_cartopy_graceful_fallback(self):
        """Test that cartopy falls back gracefully when not available."""
        import pypsa

        n = pypsa.Network()
        n.add("Bus", "bus1", x=0, y=0)
        n.add(
            "Bus", "bus2", x=1, y=1
        )  # TODO: Actual issue lies in x_min, x_max, y_min, y_max calculation when only one bus is present. Fix this properly. Reproduce by removing one bus.

        with patch("pypsa.plot.maps.common._is_cartopy_available", return_value=False):
            n.plot(geomap=True)  # Should work despite geomap=True
