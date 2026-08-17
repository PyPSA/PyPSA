# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import warnings

import pandas as pd
import pytest

from pypsa import Components, Network
from pypsa.components.legacy import Component
from pypsa.components.types import get as get_component_type


def test_components_non_implemented():
    """Test that the components module raises an ImportError if imported directly."""
    ct = get_component_type("Generator")
    with pytest.raises(NotImplementedError):
        Components(ctype=ct, names=["Generator"])
    n = Network()
    with pytest.raises(NotImplementedError):
        Components(ctype=ct, n=n)


@pytest.fixture
def legacy_component():
    n = Network()
    # Create a sample component object
    data = {"active": [True, False, True], "other_attr": [1, 2, 3]}
    static = pd.DataFrame(data, index=["asset1", "asset2", "asset3"])
    dynamic = {"time_series": pd.DataFrame({"value": [0.1, 0.2, 0.3]})}

    component = Component(
        name="Generator",
        n=n,
        static=static,
        dynamic=dynamic,
    )
    return component


def test_component_initialization(legacy_component):
    component = legacy_component
    assert component.name == "Generator"
    assert component.list_name == "generators"
    assert component.static.shape == (3, 2)
    assert "time_series" in component.dynamic


def test_active_assets(legacy_component):
    component = legacy_component
    active_assets = component.static.query("active").index
    assert len(active_assets) == 2
    assert "asset1" in active_assets
    assert "asset2" not in active_assets
    assert "asset3" in active_assets


def test_components_iteration_equivalence():
    """Test that self.components and self.iterate_components yield the same results."""
    n = Network()
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=10)
    n.add("Line", "line", bus0="bus", bus1="bus", x=0.1, r=0.01)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old_components = sorted([c.name for c in n.iterate_components()])

    new_components = sorted([c.name for c in n.components])

    assert old_components == new_components
    assert all(not c.empty for c in n.components)


def test_active_in_investment_period(legacy_component):
    component = legacy_component
    active_assets = component.get_active_assets()
    assert active_assets.sum() == 2
    assert active_assets["asset1"]
    assert not active_assets["asset2"]
    assert active_assets["asset3"]


def test_imports():
    with pytest.raises(ImportError):
        from pypsa.components import Network  # noqa: F401
    with pytest.raises(ImportError):
        from pypsa.components import SubNetwork  # noqa: F401


def test_modulars_property():
    """Test the modulars property returns correct indices."""
    n = Network()
    n.add("Bus", "bus")

    # Add modular generator
    n.add("Generator", "gen_mod", bus="bus", p_nom=100, p_nom_mod=10)

    # Add non-modular generator (explicit zero)
    n.add("Generator", "gen_nonmod", bus="bus", p_nom=100, p_nom_mod=0)

    # Add generator without mod attribute (defaults to 0)
    n.add("Generator", "gen_default", bus="bus", p_nom=100)

    modulars = n.c.generators.modulars

    # Check modular generator is in modulars
    assert "gen_mod" in modulars

    assert "gen_nonmod" not in modulars
    assert "gen_default" not in modulars


def test_modulars_with_extendables_and_committables():
    """Test modulars property works correctly with extendables and committables."""
    n = Network()
    n.add("Bus", "bus")

    # Modular + extendable
    n.add(
        "Generator",
        "gen_mod_ext",
        bus="bus",
        p_nom=100,
        p_nom_mod=10,
        p_nom_extendable=True,
        capital_cost=10,
    )

    # Modular + committable
    n.add(
        "Generator", "gen_mod_com", bus="bus", p_nom=100, p_nom_mod=10, committable=True
    )

    # Modular + extendable + committable
    n.add(
        "Generator",
        "gen_mod_ext_com",
        bus="bus",
        p_nom=100,
        p_nom_mod=10,
        p_nom_extendable=True,
        committable=True,
        capital_cost=10,
    )

    # Non-modular + extendable
    n.add(
        "Generator",
        "gen_ext",
        bus="bus",
        p_nom=100,
        p_nom_extendable=True,
        capital_cost=10,
    )

    # Non-modular + committable
    n.add("Generator", "gen_com", bus="bus", p_nom=100, committable=True)

    modulars = n.c.generators.modulars
    extendables = n.c.generators.extendables
    committables = n.c.generators.committables

    # Test modulars
    assert set(modulars) == {"gen_mod_ext", "gen_mod_com", "gen_mod_ext_com"}

    # Test intersections
    mod_ext = modulars.intersection(extendables)
    assert set(mod_ext) == {"gen_mod_ext", "gen_mod_ext_com"}

    mod_com = modulars.intersection(committables)
    assert set(mod_com) == {"gen_mod_com", "gen_mod_ext_com"}

    mod_ext_com = modulars.intersection(extendables).intersection(committables)
    assert set(mod_ext_com) == {"gen_mod_ext_com"}


def test_modulars_different_components():
    """Test modulars property works for different component types."""
    n = Network()
    n.add("Bus", "bus1")
    n.add("Bus", "bus2")

    # Test with Lines (s_nom_mod)
    n.add("Line", "line_mod", bus0="bus1", bus1="bus2", x=0.1, s_nom_mod=100)
    n.add("Line", "line_nonmod", bus0="bus1", bus1="bus2", x=0.1, s_nom_mod=0)

    line_modulars = n.c.lines.modulars

    assert "line_mod" in line_modulars
    assert "line_nonmod" not in line_modulars

    # Test with Links (p_nom_mod)
    n.add("Link", "link_mod", bus0="bus1", bus1="bus2", p_nom_mod=50)
    n.add("Link", "link_nonmod", bus0="bus1", bus1="bus2", p_nom_mod=0)

    link_modulars = n.c.links.modulars

    assert "link_mod" in link_modulars
    assert "link_nonmod" not in link_modulars

    # Test with Stores (e_nom_mod)
    n.add("Store", "store_mod", bus="bus1", e_nom_mod=1000)
    n.add("Store", "store_nonmod", bus="bus1", e_nom_mod=0)

    store_modulars = n.c.stores.modulars

    assert "store_mod" in store_modulars
