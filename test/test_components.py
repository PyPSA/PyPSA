# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import warnings

import pandas as pd
import pytest

import pypsa
from pypsa import Components, Network
from pypsa.components.legacy import Component
from pypsa.components.types import get as get_component_type


def test_components_init():
    """Test that components require a network."""
    ct = get_component_type("Generator")
    n = Network()
    c = Components(ctype=ct, n=n)
    assert c.n is n
    with pytest.raises(TypeError):
        Components(ctype=ct)
    with pytest.raises(TypeError):
        Component(name="Generator")


def test_n_save_deprecated():
    n = Network()
    with pytest.warns(DeprecationWarning, match="Use `c.n` instead"):
        assert n.c.generators.n_save is n


def test_attached_deprecated():
    n = Network()
    with pytest.warns(DeprecationWarning, match="always attached to a network"):
        assert n.c.generators.attached is True


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
    n.add(
        "Generator", "gen_mod", bus="bus", p_nom=100, p_nom_mod=10, build_mode="modular"
    )

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
        build_mode="modular",
        p_nom_extendable=True,
        capital_cost=10,
    )

    # Modular + committable
    n.add(
        "Generator",
        "gen_mod_com",
        bus="bus",
        p_nom=100,
        p_nom_mod=10,
        build_mode="modular",
        committable=True,
    )

    # Modular + extendable + committable
    n.add(
        "Generator",
        "gen_mod_ext_com",
        bus="bus",
        p_nom=100,
        p_nom_mod=10,
        build_mode="modular",
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
    n.add(
        "Line",
        "line_mod",
        bus0="bus1",
        bus1="bus2",
        x=0.1,
        s_nom_mod=100,
        build_mode="modular",
    )
    n.add("Line", "line_nonmod", bus0="bus1", bus1="bus2", x=0.1, s_nom_mod=0)

    line_modulars = n.c.lines.modulars

    assert "line_mod" in line_modulars
    assert "line_nonmod" not in line_modulars

    # Test with Links (p_nom_mod)
    n.add(
        "Link", "link_mod", bus0="bus1", bus1="bus2", p_nom_mod=50, build_mode="modular"
    )
    n.add("Link", "link_nonmod", bus0="bus1", bus1="bus2", p_nom_mod=0)

    link_modulars = n.c.links.modulars

    assert "link_mod" in link_modulars
    assert "link_nonmod" not in link_modulars

    # Test with Stores (e_nom_mod)
    n.add("Store", "store_mod", bus="bus1", e_nom_mod=1000, build_mode="modular")
    n.add("Store", "store_nonmod", bus="bus1", e_nom_mod=0)

    store_modulars = n.c.stores.modulars

    assert "store_mod" in store_modulars


@pytest.mark.parametrize(
    ("build_mode", "p_nom_mod", "semi_continuous", "modular"),
    [
        ("continuous", 0, False, False),
        ("semi_continuous", 0, True, False),
        ("modular", 50, False, True),
    ],
)
def test_build_mode_selects_subset(build_mode, p_nom_mod, semi_continuous, modular):
    """`build_mode` selects semi-continuous and fixed-size modular elements."""
    n = Network()
    n.add("Bus", "bus")
    n.add("Generator", "gen", bus="bus", build_mode=build_mode, p_nom_mod=p_nom_mod)

    c = n.c.generators
    assert ("gen" in c._semi_continuous) == semi_continuous
    assert ("gen" in c.modulars) == modular


@pytest.mark.parametrize(
    ("component", "kwargs"),
    [
        ("Line", {"bus0": "bus1", "bus1": "bus2", "x": 0.1}),
        ("Transformer", {"bus0": "bus1", "bus1": "bus2", "x": 0.1}),
        ("Link", {"bus0": "bus1", "bus1": "bus2"}),
        ("Store", {"bus": "bus1"}),
        ("StorageUnit", {"bus": "bus1"}),
    ],
)
def test_semi_continuous_different_components(component, kwargs):
    """`build_mode` works for all components with a nominal capacity."""
    n = Network()
    n.add("Bus", ["bus1", "bus2"])
    n.add(component, "on", build_mode="semi_continuous", **kwargs)
    n.add(component, "off", **kwargs)

    assert list(n.c[component]._semi_continuous) == ["on"]


def test_semi_continuous_without_attribute():
    """Components without the `build_mode` attribute return an empty index."""
    n = Network()
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=100)

    assert n.c.loads._semi_continuous.empty
    assert n.c.buses._semi_continuous.empty


def test_semi_continuous_with_scenarios():
    """The build decision cannot vary across scenarios."""
    n = Network(snapshots=range(2))
    n.set_scenarios({"low": 0.5, "high": 0.5})
    n.add("Bus", "bus")
    n.add(
        "Generator", "gen_modular", bus="bus", p_nom=100, build_mode="semi_continuous"
    )
    n.add("Generator", "gen_default", bus="bus", p_nom=100)

    semi_continuous = n.c.generators._semi_continuous

    assert not isinstance(semi_continuous, pd.MultiIndex)
    assert list(semi_continuous) == ["gen_modular"]


def test_build_cost_property():
    """`build_cost` is passed through when given directly."""
    n = Network(snapshots=range(2))
    n.add("Bus", "bus")
    n.add(
        "Generator",
        "gen",
        bus="bus",
        p_nom=100,
        build_mode="semi_continuous",
        build_cost=1234.0,
    )

    assert n.c.generators.periodized_build_cost.sel(name="gen").item() == 1234.0


def test_build_cost_property_overnight():
    """`build_cost_overnight` is annuitised and takes precedence over `build_cost`."""
    from pypsa.costs import annuity

    n = Network(snapshots=range(2))
    n.add("Bus", "bus")
    n.add(
        "Generator",
        "gen",
        bus="bus",
        p_nom=100,
        build_mode="semi_continuous",
        build_cost=1.0,
        build_cost_overnight=1000.0,
        discount_rate=0.07,
        lifetime=25,
    )

    expected = 1000.0 * annuity(0.07, 25) * n.c.generators.nyears
    periodized = n.c.generators.periodized_build_cost.sel(name="gen").item()
    assert periodized == pytest.approx(expected)


@pytest.mark.parametrize(
    ("component", "kwargs"),
    [
        ("Bus", {}),
        ("Line", {"bus0": "a", "bus1": "b", "x": 1}),
        ("Transformer", {"bus0": "a", "bus1": "b", "x": 0.1, "s_nom": 1}),
        ("Link", {"bus0": "a", "bus1": "b"}),
        ("Generator", {"bus": "a"}),
        ("StorageUnit", {"bus": "a"}),
    ],
)
@pytest.mark.filterwarnings("ignore::pypsa.PyPSA2Warning")
def test_add_fills_omitted_static_attributes_with_defaults(component, kwargs):
    with pypsa.option_context("future.string_dtype", None):
        n = Network()
        n.add("Bus", ["a", "b"])
        n.add(component, "x", **kwargs)
        empty = Network().c[component].static
    static = n.c[component].static.drop(columns=list(kwargs))
    defaults = n.components[component]["defaults"].default[static.columns]
    empty = empty[static.columns]
    assert static.dtypes.equals(empty.dtypes)
    assert static.loc["x"].astype(object).equals(defaults.astype(object))
