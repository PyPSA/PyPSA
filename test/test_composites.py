# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import linopy
import numpy as np
import pandas as pd
import pytest

import pypsa

pytest.importorskip("math_spec")

BATTERY = "examples/composites/battery.yaml"
PEAKER = {
    "name": "peaker",
    "parameters": {"bus": None, "availability": 1.0, "p_nom": 100.0},
    "components": {
        "Generator": {"gen": {"bus": "$bus", "p_nom": "$p_nom", "marginal_cost": 10}}
    },
    "math": {
        "variables": {"Generator_p": {"foreach": ["snapshot", "name"]}},
        "constraints": {
            "cap": {
                "foreach": ["snapshot", "peaker"],
                "expression": "sum(Generator_p, by=gen) <= 0.5 * availability * p_nom",
            }
        },
    },
}


@pytest.fixture
def v1_semantics():
    previous = linopy.options["semantics"]
    linopy.options["semantics"] = "v1"
    yield
    linopy.options["semantics"] = previous


@pytest.fixture
def n():
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=12, freq="h"))
    hours = np.arange(12)
    n.add("Bus", "elec")
    n.add("Load", "load", bus="elec", p_set=100 + 50 * np.sin(hours / 12 * 2 * np.pi))
    n.add(
        "Generator",
        "wind",
        bus="elec",
        p_nom=300,
        marginal_cost=1,
        p_max_pu=np.clip(np.cos(hours / 12 * 2 * np.pi), 0, None),
    )
    n.add("Generator", "gas", bus="elec", p_nom=200, marginal_cost=80)
    return n


def test_add_creates_members(n):
    battery = n.composites.register(BATTERY)
    battery.add("bat1", bus="elec")
    assert battery.instances.tolist() == ["bat1"]
    assert set(battery.members["component"]) == {
        "bat1-dc",
        "bat1-charger",
        "bat1-discharger",
        "bat1-store",
    }
    assert n.c.links.static.loc["bat1-charger", "bus1"] == "bat1-dc"
    assert (n.c.links.static["composite"] == "bat1").all()
    battery.remove("bat1")
    assert battery.instances.empty
    assert n.c.links.static.empty


@pytest.mark.parametrize("params", [{}, {"bus": "elec", "unknown": 1}])
def test_add_rejects_bad_parameters(n, params):
    battery = n.composites.register(BATTERY)
    with pytest.raises(ValueError):
        battery.add("bat1", **params)


def test_definition_rejects_unknown_reference():
    with pytest.raises(ValueError, match="unknown reference"):
        pypsa.composites.CompositeDefinition.from_dict(
            {"name": "x", "components": {"Link": {"a": {"bus0": "$nope"}}}}
        )


def test_add_rejects_duplicate_instance(n):
    battery = n.composites.register(BATTERY)
    battery.add("bat1", bus="elec")
    with pytest.raises(ValueError, match="already holds"):
        battery.add("bat1", bus="elec")


def test_add_overwrites_instance(n):
    n.composites.register(BATTERY)
    n.add("battery", "bat1", bus="elec", capital_cost=1)
    n.add("battery", "bat1", bus="elec", capital_cost=7, overwrite=True)
    assert n.c.links.static.at["bat1-charger", "capital_cost"] == 7
    assert n.composites.battery.instances.tolist() == ["bat1"]


def test_network_add_and_remove_dispatch(n):
    n.composites.register(BATTERY)
    names = n.add("battery", "bat", suffix=["1", "2"], bus="elec", return_names=True)
    assert names.tolist() == ["bat1", "bat2"]
    assert n.composites.battery.instances.tolist() == ["bat1", "bat2"]
    n.remove("battery", ["bat1", "bat2"])
    assert n.composites.battery.instances.empty
    assert n.c.stores.static.empty


def test_add_many_with_per_instance_and_time_varying_parameters(n):
    battery = n.composites.register(BATTERY)
    n.add("Bus", "elec2")
    cost = pd.DataFrame(
        np.arange(24.0).reshape(12, 2), index=n.snapshots, columns=["a", "b"]
    )
    battery.add(
        ["a", "b"], bus=["elec", "elec2"], capital_cost=[10, 20], marginal_cost=cost
    )
    links = n.c.links.static
    assert links.loc[["a-charger", "b-charger"], "bus0"].tolist() == ["elec", "elec2"]
    assert links.loc[["a-charger", "b-charger"], "capital_cost"].tolist() == [10, 20]
    assert links.at["b-charger", "bus1"] == "b-dc"
    assert (
        n.c.links.dynamic.marginal_cost["b-discharger"].tolist() == cost["b"].tolist()
    )
    assert links.loc[
        ["a-charger", "b-charger"], "composite_param_capital_cost"
    ].tolist() == [10, 20]
    stored = n.c.links.dynamic.composite_param_marginal_cost
    assert stored.columns.tolist() == ["a-charger", "b-charger"]
    assert stored["b-charger"].tolist() == cost["b"].tolist()


def test_remove_rejects_unknown_instance(n):
    battery = n.composites.register(BATTERY)
    with pytest.raises(ValueError, match="no instances"):
        battery.remove("ghost")


def test_register_rejects_duplicate_name(n):
    n.composites.register(BATTERY)
    with pytest.raises(ValueError, match="already registered"):
        n.composites.register(BATTERY)


@pytest.mark.parametrize("name", ["Store", "links"])
def test_register_rejects_component_class_name(name):
    with pytest.raises(ValueError, match="clashes"):
        pypsa.Network().composites.register(
            {"name": name, "components": {"Link": {"a": {}}}}
        )


def test_register_accepts_dict_and_text():
    import yaml

    definition = {
        "name": "pipe",
        "parameters": {"src": None, "dst": None},
        "components": {"Link": {"a": {"bus0": "$src", "bus1": "$dst"}}},
    }
    assert pypsa.Network().composites.register(definition).name == "pipe"
    text = yaml.safe_dump(definition)
    assert pypsa.Network().composites.register(text).name == "pipe"


@pytest.mark.parametrize(
    ("variables", "match"),
    [
        ({"Link_p": {"foreach": ["name"], "bounds": 1}}, "may only"),
        ({"Link_p": None}, "must be a mapping"),
        ({"Link_p": "name"}, "must be a mapping"),
    ],
)
def test_definition_rejects_bad_variable_decl(variables, match):
    with pytest.raises(ValueError, match=match):
        pypsa.composites.CompositeDefinition.from_dict(
            {
                "name": "x",
                "components": {"Link": {"a": {}}},
                "math": {"variables": variables},
            }
        )


def test_definition_rejects_several_bound_classes():
    with pytest.raises(ValueError, match="single class"):
        pypsa.composites.CompositeDefinition.from_dict(
            {
                "name": "x",
                "components": {"Link": {"a": {}}},
                "math": {
                    "variables": {
                        "Link_p": {"foreach": ["snapshot", "name"]},
                        "Store_e": {"foreach": ["snapshot", "name"]},
                    }
                },
            }
        )


@pytest.mark.usefixtures("v1_semantics")
def test_optimize_with_math_layer(n):
    battery = n.composites.register(BATTERY)
    battery.add("bat1", bus="elec", capital_cost=10, efficiency=0.9)
    battery.add("bat2", bus="elec", capital_cost=20)
    status, _ = n.optimize()
    assert status == "ok"
    links = n.c.links.static
    for inst in ("bat1", "bat2"):
        assert links.at[f"{inst}-charger", "p_nom_opt"] == pytest.approx(
            links.at[f"{inst}-discharger", "p_nom_opt"]
        )
    p_nom = battery.expressions["p_nom"]
    assert p_nom.index.tolist() == ["bat1", "bat2"]
    assert p_nom["bat1"] == pytest.approx(links.at["bat1-discharger", "p_nom_opt"])
    p = battery.expressions["p"]
    flows = n.c.links.dynamic.p
    expected = flows["bat1-discharger"] - flows["bat1-charger"]
    pd.testing.assert_series_equal(
        p["bat1"], expected, check_names=False, check_freq=False
    )
    capacity = n.statistics.optimal_capacity(groupby="composite")
    assert capacity.loc["Link"].index.tolist() == ["bat1", "bat2"]
    assert "Generator" not in capacity.index.get_level_values(0)


def test_optimize_requires_v1_semantics(n):
    previous = linopy.options["semantics"]
    linopy.options["semantics"] = "legacy"
    try:
        battery = n.composites.register(BATTERY)
        battery.add("bat1", bus="elec")
        with pytest.raises(ValueError, match="v1 semantics"):
            n.optimize.create_model()
    finally:
        linopy.options["semantics"] = previous


def test_register_python_definition(n):
    class Coupled(pypsa.composites.CompositeDefinition):
        name: str = "coupled"
        parameters: dict = {"bus": None, "p_nom": 10.0}
        components: dict = {
            "Link": {"a": {"bus0": "$bus", "bus1": "$bus", "p_nom": "$p_nom"}}
        }

    coupled = n.composites.register(Coupled())
    coupled.add("c1", bus="elec")
    assert n.c.links.static.loc["c1-a", "p_nom"] == 10.0
    with pytest.raises(ValueError, match="extra"):
        Coupled(unknown=1)


@pytest.fixture
def availability(n):
    series = pd.Series(1.0, index=n.snapshots, name="availability")
    series.iloc[4:7] = 0.0
    return series


def _assert_capped_dispatch(n, instance, availability, cap=50):
    p = n.c.generators.dynamic.p[f"{instance}-gen"]
    assert (p[availability == 0] == 0).all()
    assert p[n.snapshots[[3, 7]]].tolist() == pytest.approx([cap, cap])


@pytest.mark.usefixtures("v1_semantics")
def test_time_varying_parameter_reaches_math(n, availability):
    peaker = n.composites.register(PEAKER)
    peaker.add("pk", bus="elec", availability=availability)
    status, _ = n.optimize()
    assert status == "ok"
    _assert_capped_dispatch(n, "pk", availability)
    frame = peaker._parameters()["availability"]
    assert isinstance(frame, pd.DataFrame)
    assert frame.columns.name == "peaker"
    pd.testing.assert_series_equal(
        frame["pk"], availability, check_names=False, check_freq=False
    )
    assert isinstance(peaker._parameters()["p_nom"], pd.Series)


@pytest.mark.usefixtures("v1_semantics")
@pytest.mark.parametrize("as_frame", [True, False])
def test_mixed_time_varying_and_scalar_instances(n, availability, as_frame):
    peaker = n.composites.register(PEAKER)
    if as_frame:
        values = pd.DataFrame({"a": availability, "b": 0.5})
        peaker.add(["a", "b"], bus="elec", availability=values)
    else:
        peaker.add("a", bus="elec", availability=availability)
        peaker.add("b", bus="elec", availability=0.5)
    frame = peaker._parameters()["availability"]
    assert frame.columns.tolist() == ["a", "b"]
    assert frame["a"].tolist() == availability.tolist()
    assert (frame["b"] == 0.5).all()
    status, _ = n.optimize()
    assert status == "ok"
    _assert_capped_dispatch(n, "a", availability)
    _assert_capped_dispatch(n, "b", pd.Series(0.5, index=n.snapshots), cap=25)


@pytest.mark.usefixtures("v1_semantics")
@pytest.mark.parametrize("via", ["netcdf", "copy"])
def test_time_varying_parameter_roundtrip(n, availability, tmp_path, via):
    peaker = n.composites.register(PEAKER)
    peaker.add("pk", bus="elec", availability=availability)
    if via == "netcdf":
        n.export_to_netcdf(tmp_path / "n.nc")
        m = pypsa.Network(tmp_path / "n.nc")
        m.composites.register(PEAKER)
    else:
        m = n.copy()
    n.optimize()
    m.optimize()
    np.testing.assert_allclose(
        m.c.generators.dynamic.p["pk-gen"], n.c.generators.dynamic.p["pk-gen"]
    )


def test_overwrite_replaces_time_varying_with_scalar(n, availability):
    peaker = n.composites.register(PEAKER)
    peaker.add("pk", bus="elec", availability=availability)
    assert "pk-gen" in n.c.generators.dynamic.composite_param_availability
    peaker.add("pk", bus="elec", availability=0.25, overwrite=True)
    assert "pk-gen" not in n.c.generators.dynamic.composite_param_availability
    param = peaker._parameters()["availability"]
    assert isinstance(param, pd.Series)
    assert param.tolist() == [0.25]


@pytest.mark.usefixtures("v1_semantics")
def test_fragment_cannot_broadcast_time_varying_parameter(n, availability):
    definition = {
        **PEAKER,
        "math": {
            **PEAKER["math"],
            "constraints": {
                "cap": {
                    "foreach": ["peaker"],
                    "expression": "sum(sum(Generator_p, by=gen), over=snapshot) <= availability * p_nom",
                }
            },
        },
    }
    peaker = n.composites.register(definition)
    peaker.add("pk", bus="elec", availability=availability)
    with pytest.raises(
        Exception, match="dims \\['snapshot'\\] that are not in foreach"
    ):
        n.optimize.create_model()


def test_time_varying_parameter_rejects_investment_periods(n, availability):
    n.investment_periods = [2030]
    peaker = n.composites.register(PEAKER)
    peaker.add("pk", bus="elec", availability=availability.set_axis(n.snapshots))
    with pytest.raises(NotImplementedError, match="investment periods"):
        peaker._parameters()
