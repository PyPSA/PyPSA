# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT
"""Tests for the `suffix` argument of `Network.add` and `Network.remove`.

Covers the cases agreed in issue #1650:

  * `name` scalar + `suffix` scalar  -> single component (unchanged)
  * `name` list   + `suffix` scalar  -> broadcast suffix (unchanged)
  * `name` scalar + `suffix` list    -> broadcast name (was already accidentally
    working, now first-class supported and tested)
  * `name` list   + `suffix` list    -> raises ValueError. Previous behaviour
    silently paired the two one-by-one, producing fewer components than the
    user expected with no warning.

Also regression-tests the TypeError that used to fire whenever a list `suffix`
was combined with a `pd.Series` or `pd.DataFrame` kwarg in `n.add`.
"""

import numpy as np
import pandas as pd
import pytest

import pypsa


@pytest.fixture
def simple_network():
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2025-01-01", periods=3, freq="h"))
    n.add("Bus", "bus")
    return n


def test_add_list_name_scalar_suffix(simple_network):
    n = simple_network
    n.add("Generator", ["a", "b"], suffix=" wind", bus="bus")
    assert list(n.c.generators.static.index) == ["a wind", "b wind"]


def test_add_scalar_name_list_suffix(simple_network):
    n = simple_network
    n.add("Generator", "g", suffix=[" wind", " solar"], bus="bus")
    assert list(n.c.generators.static.index) == ["g wind", "g solar"]


def test_add_list_name_list_suffix_raises(simple_network):
    n = simple_network
    with pytest.raises(ValueError, match="Cannot pass list to both"):
        n.add(
            "Generator",
            ["a", "b"],
            suffix=[" wind", " solar"],
            bus="bus",
        )


def test_add_scalar_name_list_suffix_with_series_kwarg(simple_network):
    """Regression for the TypeError from str.endswith(list) (issue #1650)."""
    n = simple_network
    p_nom = pd.Series([100.0, 200.0], index=["g wind", "g solar"])
    n.add(
        "Generator",
        "g",
        suffix=[" wind", " solar"],
        bus="bus",
        p_nom=p_nom,
    )
    assert list(n.c.generators.static.index) == ["g wind", "g solar"]
    assert n.c.generators.static.loc["g wind", "p_nom"] == 100.0
    assert n.c.generators.static.loc["g solar", "p_nom"] == 200.0


def test_add_scalar_name_list_suffix_with_dataframe_kwarg(simple_network):
    """Regression for the TypeError on DataFrame kwargs (issue #1650)."""
    n = simple_network
    p_max_pu = pd.DataFrame(
        np.array([[0.1, 0.5], [0.2, 0.6], [0.3, 0.7]]),
        index=n.snapshots,
        columns=["g wind", "g solar"],
    )
    n.add(
        "Generator",
        "g",
        suffix=[" wind", " solar"],
        bus="bus",
        p_max_pu=p_max_pu,
    )
    assert list(n.c.generators.static.index) == ["g wind", "g solar"]
    pd.testing.assert_series_equal(
        n.c.generators.dynamic.p_max_pu["g wind"],
        p_max_pu["g wind"],
        check_names=False,
    )


def test_remove_list_name_scalar_suffix(simple_network):
    n = simple_network
    n.add("Generator", ["a wind", "b wind"], bus="bus")
    n.remove("Generator", ["a", "b"], suffix=" wind")
    assert list(n.c.generators.static.index) == []


def test_remove_scalar_name_list_suffix(simple_network):
    n = simple_network
    n.add("Generator", ["g wind", "g solar"], bus="bus")
    n.remove("Generator", "g", suffix=[" wind", " solar"])
    assert list(n.c.generators.static.index) == []


def test_remove_list_name_list_suffix_raises(simple_network):
    """Previously zipped silently, removing only as many components as the
    shorter list. Now raises (issue #1650)."""
    n = simple_network
    for b in ("DE0 0", "DE0 1"):
        for r in (" wind", " solar"):
            n.add("Generator", f"{b}{r}", bus="bus")

    with pytest.raises(ValueError, match="Cannot pass list to both"):
        n.remove(
            "Generator",
            ["DE0 0", "DE0 1"],
            suffix=[" wind", " solar"],
        )
    # All four still present after the failed call.
    assert set(n.c.generators.static.index) == {
        "DE0 0 wind",
        "DE0 0 solar",
        "DE0 1 wind",
        "DE0 1 solar",
    }
