# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for the optional SMS++ accessor."""

import pypsa


def test_network_has_smspp_accessor():
    n = pypsa.Network()
    assert hasattr(n, "smspp")


def test_smspp_missing_optional_deps_raises(monkeypatch):
    import pytest

    n = pypsa.Network()

    # Force the dependency check to "not find" optional modules
    monkeypatch.setattr("pypsa.optimization.smspp.find_spec", lambda name: None)

    with pytest.raises(ImportError, match=r"pypsa\[smspp\]"):
        n.smspp(config="dummy.yaml")
