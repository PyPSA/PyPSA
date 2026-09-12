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


def test_remove_rejects_unknown_instance(n):
    battery = n.composites.register(BATTERY)
    with pytest.raises(ValueError, match="no instance"):
        battery.remove("ghost")


def test_register_rejects_duplicate_name(n):
    n.composites.register(BATTERY)
    with pytest.raises(ValueError, match="already registered"):
        n.composites.register(BATTERY)


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
