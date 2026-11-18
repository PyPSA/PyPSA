# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for Sankey diagrams of the energy balance."""

import pandas as pd
import pytest

import pypsa

ENERGY = ["electricity", "battery", "gas", "heat"]
WIND = (40 + 80 + 10 - 6.4) / 0.9


@pytest.fixture(scope="module")
def n():
    """Solved sector-coupled network with gains, losses, storage and emissions."""
    n = pypsa.Network()
    n.set_snapshots(range(4))
    colors = {"electricity": "#110d63", "heat": "#cc1f1f", "wind": "#235ebc"}
    for carrier in [*ENERGY, "co2", "wind", "heat pump", "boiler", "HVDC"]:
        n.add("Carrier", carrier, color=colors.get(carrier, ""))
    n.add("Carrier", ["charger", "discharger"])
    n.add("Bus", ["north", "south"], carrier="electricity")
    n.add("Bus", ENERGY[1:] + ["co2"], carrier=ENERGY[1:] + ["co2"])
    n.c.buses.static.loc[ENERGY[1:] + ["north", "south"], "unit"] = "MWh"

    n.add("Generator", "wind", bus="north", carrier="wind", p_nom=100)
    n.add("Generator", "gas", bus="gas", carrier="gas", p_nom=100, marginal_cost=20)
    n.add("Load", "electricity", bus="south", carrier="electricity", p_set=10)
    n.add("Load", "heat", bus="heat", carrier="heat", p_set=[60, 60, 60, 150])
    n.add("Store", "co2", bus="co2", carrier="co2", e_nom=1e6)
    n.add("Store", "battery", bus="battery", carrier="battery", e_nom=20)
    n.add(
        "Link",
        "HVDC",
        bus0="north",
        bus1="south",
        carrier="HVDC",
        p_nom=100,
        efficiency=0.9,
    )
    n.add(
        "Link",
        ["charger", "discharger"],
        bus0=["south", "battery"],
        bus1=["battery", "south"],
        carrier=["charger", "discharger"],
        p_nom=10,
        efficiency=0.8,
        p_max_pu=pd.DataFrame(
            {"charger": [1, 0, 0, 0], "discharger": [0, 1, 0, 0]}, index=n.snapshots
        ),
        marginal_cost=[-1, -1],
    )
    n.add(
        "Link",
        "heat pump",
        bus0="south",
        bus1="heat",
        carrier="heat pump",
        p_nom=20,
        efficiency=3,
    )
    n.add(
        "Process",
        "boiler",
        bus0="gas",
        bus1="heat",
        bus2="co2",
        carrier="boiler",
        rate0=-1,
        rate1=0.9,
        rate2=0.2,
        p_nom=100,
    )
    n.optimize(log_to_console=False)
    n.model.solver_model = None
    return n


def flows(n, **kwargs) -> pd.Series:
    """Get the flows of a Sankey diagram by source, target and carrier label."""
    defaults = {"bus_carrier": ENERGY, "threshold": 0, "split": False}
    kwargs = defaults | {"annotate": False} | kwargs
    (trace,) = n.statistics.sankey(**kwargs).data
    labels = pd.Series(trace.node.label)
    index = [labels[list(trace.link.source)], labels[list(trace.link.target)]]
    index = pd.MultiIndex.from_arrays([*index, trace.link.label])
    return pd.Series(trace.link.value, index.set_names(["source", "target", "carrier"]))


def imbalance(fig) -> pd.Series:
    """Get inflow minus outflow of the nodes with inflow and outflow."""
    (trace,) = fig.data
    value = pd.Series(trace.link.value)
    inflow = value.groupby(list(trace.link.target)).sum()
    outflow = value.groupby(list(trace.link.source)).sum()
    return (inflow - outflow).dropna()


def test_flows(n):
    f = flows(n)
    assert f["wind", "electricity", "wind"] == pytest.approx(WIND)
    assert f["electricity", "heat pump", "heat pump"] == pytest.approx(80)
    assert f["gains", "heat pump", "heat pump"] == pytest.approx(160)
    assert f["heat pump", "heat", "heat pump"] == pytest.approx(240)
    assert f["boiler", "losses", "boiler"] == pytest.approx(10)
    assert f["electricity", "losses", "HVDC"] == pytest.approx(0.1 * WIND)
    assert f["charger", "losses", "charger"] == pytest.approx(2)
    assert f["heat", "heat", "heat"] == pytest.approx(330)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"collapse": True},
        {"collapse": ["heat pump"]},
        {"threshold": 0.1},
        {"threshold": 0.1, "collapse": True},
        {"merge": {"battery": "electricity"}},
        {"merge": {"battery": "electricity"}, "collapse": True, "threshold": 0.05},
        {"split": True},
    ],
)
def test_balanced(n, kwargs):
    through = imbalance(n.statistics.sankey(bus_carrier=ENERGY, **kwargs))
    assert not through.empty
    assert through.abs().max() < 1e-6


def test_collapse(n):
    f = flows(n, collapse=True)
    assert f["electricity", "heat", "heat pump"] == pytest.approx(80)
    assert f["gains", "heat", "heat pump"] == pytest.approx(160)
    assert f["gas", "heat", "boiler"] == pytest.approx(90)
    assert f["gas", "losses", "boiler"] == pytest.approx(10)
    assert "heat pump" not in f.index.unique("source")


def test_merge(n):
    f = flows(n, merge={"battery": "electricity"})
    assert "battery" not in f.index.unique("source")
    assert f["electricity", "losses", "charger"] == pytest.approx(2)
    assert f["electricity", "losses", "discharger"] == pytest.approx(1.6)


def test_threshold(n):
    f = flows(n, threshold=0.15)
    assert f["electricity", "other", "other"] == pytest.approx(40 + 10 - 6.4)
    assert f["electricity", "losses", "other"] == pytest.approx(0.1 * WIND)
    assert "battery" not in f.index.unique("source")
    assert f["wind", "electricity", "wind"] == pytest.approx(WIND)


def test_bus_carrier(n):
    f = flows(n, bus_carrier="co2")
    assert f.to_dict() == pytest.approx(
        {("boiler", "co2", "boiler"): 20, ("co2", "co2", "co2"): 20}
    )


def test_weights(n):
    f = flows(n, bus_carrier=["co2", "gas"], weights={"gas": 0.2})
    assert f["gas", "boiler", "boiler"] == pytest.approx(20)
    assert f["boiler", "co2", "boiler"] == pytest.approx(20)
    assert "losses" not in f.index.unique("target")


@pytest.mark.parametrize(("split", "expected"), [(["heat"], "heat"), (True, "battery")])
def test_split(n, split, expected):
    fig = n.statistics.sankey(bus_carrier=ENERGY, split=split, annotate=False)
    (trace,) = fig.data
    labels = pd.Series(trace.node.label)
    copies = labels.index[labels == expected]
    supplying = copies.intersection(trace.link.source)
    receiving = copies.intersection(trace.link.target)
    assert len(supplying) > 0
    assert len(receiving) > 0
    assert supplying.intersection(receiving).empty
    assert imbalance(fig).abs().max() < 1e-6


def test_labels_and_colors(n):
    kwargs = {"bus_carrier": ENERGY, "losses": "waste", "gains": "ambient heat"}
    node = n.statistics.sankey(**kwargs, annotate=False).data[0].node
    colors = dict(zip(node.label, node.color, strict=True))
    assert colors["wind"] == "rgba(35,94,188,1)"
    assert colors["waste"] == colors["ambient heat"] == "rgba(179,179,179,1)"
    assert colors["heat pump"] != colors["boiler"]


def test_units_and_annotation(n):
    (trace,) = n.statistics.sankey(bus_carrier="heat").data
    assert trace.valuesuffix == " MWh"
    assert "heat pump 240 MWh" in trace.node.label
    (trace,) = n.statistics.sankey(bus_carrier="heat", scale=1e-3).data
    assert trace.valuesuffix == ""
    assert "heat pump 0.24" in trace.node.label
    (trace,) = n.statistics.sankey(bus_carrier="heat", scale=1e-3, unit="GWh").data
    assert trace.valuesuffix == " GWh"
    assert "heat pump 0.24 GWh" in trace.node.label


def test_nice_names(n):
    n = n.copy()
    n.c.carriers.static.loc[["battery", "electricity"], "nice_name"] = "Power"
    n.c.carriers.static.loc["HVDC", "nice_name"] = "Interconnector"
    f = flows(n)
    assert f["wind", "Power", "wind"] == pytest.approx(WIND)
    assert f["Power", "losses", "Interconnector"] == pytest.approx(0.1 * WIND)
    merged = flows(n, merge={"gas": "Power"})
    assert merged.equals(flows(n, merge={"gas": "electricity"}))
    assert "gas" not in merged.index.unique("target")
    assert "battery" not in f.index.unique("source")
    f = flows(n, nice_names=False)
    assert "battery" in f.index.unique("source")


def test_example_network(ac_dc_network_r):
    fig = ac_dc_network_r.statistics.sankey()
    assert imbalance(fig).abs().max() < 1e-3


def test_periods(ac_dc_network):
    n = ac_dc_network
    n.set_investment_periods([2020, 2030])
    n.c.loads.dynamic.p_set.loc[2030] *= 0.5
    n.optimize(multi_investment_periods=True, log_to_console=False)
    kwargs = {"bus_carrier": None, "nice_names": False}
    first, second = (flows(n, period=period, **kwargs) for period in [2020, 2030])
    total = flows(n, **kwargs)
    assert first.sum() == pytest.approx(2 * second.sum(), rel=0.05)
    assert total.sum() == pytest.approx(first.sum() + second.sum())


def test_scenarios(stochastic_network):
    n = stochastic_network
    n.optimize(log_to_console=False)
    kwargs = {"bus_carrier": None, "nice_names": False}
    gas = ("gas", "AC", "gas")
    expected = sum(
        weight * flows(n, scenario=scenario, **kwargs)[gas]
        for scenario, weight in n.scenario_weightings.weight.items()
    )
    assert flows(n, **kwargs)[gas] == pytest.approx(expected)


def test_errors(n, model_energy_network, network_collection):
    with pytest.raises(ValueError, match="Cannot split"):
        n.statistics.sankey(split=["wind"])
    with pytest.raises(ValueError, match="energy balance is empty"):
        model_energy_network.statistics.sankey()
    with pytest.raises(NotImplementedError):
        network_collection.statistics.sankey()
