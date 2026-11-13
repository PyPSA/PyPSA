# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for deriving a zonal PTDF from a nodal sub-network (GSK projection)."""

import numpy as np
import pandas as pd
import pytest

import pypsa

ZONES = pd.Series({"n1": "A", "n2": "A", "n3": "B"})
IDENTITY = pd.Series({"n1": "n1", "n2": "n2", "n3": "n3"})


def triangle_network(p_nom=(100.0, 300.0, 50.0), **line_kwargs) -> pypsa.Network:
    """Three-bus triangle, cheap generation at n1 and load at n2 and n3."""
    n = pypsa.Network()
    n.add("Bus", ["n1", "n2", "n3"])
    n.add(
        "Line",
        ["l12", "l23", "l13"],
        bus0=["n1", "n2", "n1"],
        bus1=["n2", "n3", "n3"],
        x=0.1,
        s_nom=[100, 400, 400],
        **line_kwargs,
    )
    n.add(
        "Generator",
        ["g1", "g2", "g3"],
        bus=["n1", "n2", "n3"],
        p_nom=list(p_nom),
        marginal_cost=[10, 80, 50],
        carrier=["wind", "gas", "coal"],
    )
    return n


def sub_network(n: pypsa.Network) -> pypsa.SubNetwork:
    n.determine_network_topology()
    return n.c.sub_networks.static.obj.iloc[0]


# GSK builders


@pytest.mark.parametrize(
    ("builder", "kwargs", "p_nom", "weight_n1"),
    [
        (pypsa.SubNetwork.gsk_uniform, {}, (100, 300, 50), 0.5),
        (pypsa.SubNetwork.gsk_by_capacity, {}, (100, 300, 50), 0.25),  # 100 / 400
        (
            pypsa.SubNetwork.gsk_by_capacity,
            {"carrier": ["wind", "coal"]},
            (100, 300, 50),
            1.0,
        ),
        (pypsa.SubNetwork.gsk_by_capacity, {"attr": "p_nom_opt"}, (100, 300, 50), 0.75),
        (pypsa.SubNetwork.gsk_by_capacity, {}, (0, 0, 50), 0.5),  # uniform fallback
    ],
)
def test_gsk_weights(builder, kwargs, p_nom, weight_n1):
    n = triangle_network(p_nom=p_nom)
    sub = sub_network(n)
    sub.n.c.generators.static["p_nom_opt"] = [300.0, 100.0, 50.0]
    g = builder(sub, ZONES, **kwargs)
    np.testing.assert_allclose(g.sum(), 1.0)
    assert g.loc["n1", "A"] == pytest.approx(weight_n1)
    assert g.loc["n2", "A"] == pytest.approx(1 - weight_n1)
    assert g.loc["n3", "B"] == pytest.approx(1.0)


# Zonal PTDF


def test_cnecs_cover_both_directions():
    n = triangle_network()
    sub = sub_network(n)
    zp = sub.calculate_zonal_PTDF(ZONES)
    assert list(zp.columns) == ["A", "B"]
    plus = zp.loc[["l12 (+)", "l23 (+)", "l13 (+)"]]
    minus = zp.loc[["l12 (-)", "l23 (-)", "l13 (-)"]]
    np.testing.assert_allclose(plus.values, -minus.values)
    monitored = sub.calculate_zonal_PTDF(ZONES, branches=["l23"])
    assert list(monitored.index) == ["l23 (+)", "l23 (-)"]


@pytest.mark.parametrize("gsk", ["uniform", "capacity"])
def test_flows_match_linear_power_flow(gsk):
    """Zonal PTDF x NP equals the LPF flows of the GSK-distributed injections."""
    n = triangle_network()
    n.add("Bus", "n4")
    n.add("Transformer", "t34", bus0="n3", bus1="n4", x=0.1, s_nom=100)
    zones = pd.concat([ZONES, pd.Series({"n4": "B"})])
    sub = sub_network(n)
    net_position = pd.Series({"A": 120.0, "B": -120.0})
    builders = {"uniform": sub.gsk_uniform, "capacity": sub.gsk_by_capacity}
    injection = builders[gsk](zones) @ net_position
    buses = n.c.buses.static.index
    n.add("Load", buses, bus=buses, p_set=-injection)
    zp = sub.calculate_zonal_PTDF(zones, gsk=gsk)
    n.lpf()
    expected = pd.concat(
        [n.c[c].dynamic["p0"].iloc[0] for c in ("Line", "Transformer")]
    )
    np.testing.assert_allclose(
        (zp.loc[expected.index + " (+)"] @ net_position).values,
        expected.values,
        atol=1e-9,
    )


def test_flow_based_reproduces_nodal_optimum():
    """With one bus per zone, the flow-based domain equals the nodal LOPF."""
    nodal, zonal = (triangle_network(p_nom=(1000.0, 1000.0, 1000.0)) for _ in range(2))
    for n in (nodal, zonal):
        n.add("Load", ["d2", "d3"], bus=["n2", "n3"], p_set=[300, 300])
    zonal.remove("Line", zonal.c.lines.static.index)
    s_nom = nodal.c.lines.static.s_nom
    zonal.c.global_constraints.add_flow_based(
        sub_network(nodal).calculate_zonal_PTDF(IDENTITY),
        ram=pd.concat([s_nom.add_suffix(" (+)"), s_nom.add_suffix(" (-)")]),
    )
    nodal.optimize()
    zonal.optimize()

    assert (nodal.c.lines.dynamic["p0"].abs().iloc[0] >= s_nom - 1e-6).any()
    assert zonal.objective == pytest.approx(nodal.objective)
    pd.testing.assert_frame_equal(
        zonal.c.generators.dynamic["p"], nodal.c.generators.dynamic["p"], atol=1e-6
    )


def test_single_node_subnetwork_is_empty():
    n = pypsa.Network()
    n.add("Bus", "iso")
    zp = sub_network(n).calculate_zonal_PTDF(pd.Series({"iso": "Z"}))
    assert zp.empty
    assert list(zp.columns) == ["Z"]


# Validation


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"node_to_zone": ZONES.drop("n3")}, "every sub-network bus"),
        ({"gsk": "bogus"}, "Unknown GSK scheme"),
        (
            {"gsk": pd.DataFrame({"A": [1, 1, 0], "B": [0, 0, 1]}, index=ZONES.index)},
            "sum to one",
        ),
    ],
)
def test_invalid_input_raises(kwargs, match):
    kwargs = {"node_to_zone": ZONES} | kwargs
    n = triangle_network()
    with pytest.raises(ValueError, match=match):
        sub_network(n).calculate_zonal_PTDF(**kwargs)


def test_zone_across_sub_networks_raises():
    n = triangle_network()
    n.add("Bus", "island")
    n.determine_network_topology()
    sub = n.c.sub_networks.static.obj[n.c.buses.static.at["n1", "sub_network"]]
    with pytest.raises(ValueError, match="several sub-networks"):
        sub.calculate_zonal_PTDF(pd.concat([ZONES, pd.Series({"island": "B"})]))


@pytest.mark.parametrize(
    ("component", "gsk"), [("Line", "uniform"), ("Generator", "capacity")]
)
def test_multi_period_needs_period_invariant_assets(component, gsk):
    n = triangle_network()
    n.set_snapshots(pd.MultiIndex.from_product([[2030, 2040], [0]]))
    n.set_investment_periods([2030, 2040])
    sub = sub_network(n)
    single_period = triangle_network()
    pd.testing.assert_frame_equal(
        sub.calculate_zonal_PTDF(ZONES, gsk=gsk),
        sub_network(single_period).calculate_zonal_PTDF(ZONES, gsk=gsk),
    )
    n.c[component].static["build_year"] = [2020, 2020, 2035]
    n.c[component].static["lifetime"] = 100
    with pytest.raises(NotImplementedError, match="investment period"):
        sub.calculate_zonal_PTDF(ZONES, gsk=gsk)


def test_stochastic_network_raises():
    n = triangle_network()
    n.set_scenarios({"s1": 0.5, "s2": 0.5})
    with pytest.raises(NotImplementedError, match="stochastic"):
        sub_network(n).calculate_zonal_PTDF(ZONES)
