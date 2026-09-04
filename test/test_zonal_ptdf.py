# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for deriving a zonal PTDF from a nodal sub-network (GSK projection)."""

import pandas as pd
import pytest

import pypsa

ZONES = pd.Series({"n1": "A", "n2": "A", "n3": "B"})


def _triangle(p_nom=(100.0, 300.0, 50.0)) -> pypsa.SubNetwork:
    """Three-bus triangle (one sub-network), zones A={n1,n2} and B={n3}."""
    n = pypsa.Network()
    n.add("Bus", ["n1", "n2", "n3"])
    n.add(
        "Line",
        ["l12", "l23", "l13"],
        bus0=["n1", "n2", "n1"],
        bus1=["n2", "n3", "n3"],
        x=0.1,
        s_nom=100,
    )
    n.add("Generator", ["g1", "g2", "g3"], bus=["n1", "n2", "n3"], p_nom=list(p_nom))
    n.determine_network_topology()
    return n.c.sub_networks.static.obj.iloc[0]


def test_gsk_uniform_weights_and_normalisation():
    """Uniform GSK spreads a zone evenly over its buses; columns sum to one."""
    g = _triangle().gsk_uniform(ZONES)
    assert (g.sum().round(9) == 1.0).all()
    assert g.loc["n1", "A"] == pytest.approx(0.5)
    assert g.loc["n2", "A"] == pytest.approx(0.5)
    assert g.loc["n3", "B"] == pytest.approx(1.0)


def test_gsk_by_capacity_weights_follow_p_nom():
    """Capacity GSK weights each bus by generator p_nom within its zone."""
    g = _triangle().gsk_by_capacity(ZONES)
    assert (g.sum().round(9) == 1.0).all()
    assert g.loc["n1", "A"] == pytest.approx(0.25)  # 100 / 400
    assert g.loc["n2", "A"] == pytest.approx(0.75)  # 300 / 400
    assert g.loc["n3", "B"] == pytest.approx(1.0)


def test_gsk_by_capacity_zero_capacity_zone_falls_back_to_uniform():
    """A zone with no capacity uses a uniform key instead of dividing by zero."""
    g = _triangle(p_nom=(0.0, 0.0, 50.0)).gsk_by_capacity(ZONES)
    assert g.loc["n1", "A"] == pytest.approx(0.5)  # uniform fallback in zone A
    assert g.loc["n2", "A"] == pytest.approx(0.5)
    assert g.loc["n3", "B"] == pytest.approx(1.0)


def test_calculate_zonal_PTDF_shape_and_labels():
    """The zonal PTDF is a branch x zone frame with a (type, name) row index."""
    zp = _triangle().calculate_zonal_PTDF(ZONES)
    assert zp.shape == (3, 2)
    assert list(zp.columns) == ["A", "B"]
    assert zp.index.names == ["type", "name"]
    assert ("Line", "l12") in zp.index


def test_calculate_zonal_PTDF_two_bus_physical():
    """On a single line, all of a zone's injection flows through it (slack zone -> 0)."""
    n = pypsa.Network()
    n.add("Bus", ["b0", "b1"])
    n.add("Line", "L", bus0="b0", bus1="b1", x=0.1, s_nom=100)
    n.determine_network_topology()
    sub = n.c.sub_networks.static.obj.iloc[0]
    zp = sub.calculate_zonal_PTDF(pd.Series({"b0": "Z0", "b1": "Z1"}))
    row = zp.loc[("Line", "L")].abs().round(9)
    assert sorted(row) == [0.0, 1.0]  # slack zone 0, the other +/-1


def test_calculate_zonal_PTDF_accepts_a_ready_gsk_frame():
    """Passing a built GSK frame gives the same result as naming its scheme."""
    sub = _triangle()
    by_frame = sub.calculate_zonal_PTDF(ZONES, gsk=sub.gsk_uniform(ZONES))
    by_name = sub.calculate_zonal_PTDF(ZONES, gsk="uniform")
    pd.testing.assert_frame_equal(by_frame, by_name)


def test_calculate_zonal_PTDF_fails_on_uncovered_bus():
    """Every sub-network bus must be mapped to a zone."""
    with pytest.raises(ValueError, match="every sub-network bus"):
        _triangle().calculate_zonal_PTDF(pd.Series({"n1": "A", "n2": "A"}))


def test_calculate_zonal_PTDF_unknown_scheme_raises():
    """An unknown GSK scheme name fails fast."""
    with pytest.raises(ValueError, match="Unknown GSK scheme"):
        _triangle().calculate_zonal_PTDF(ZONES, gsk="bogus")


def test_calculate_zonal_PTDF_single_node_subnetwork():
    """A branch-free (single-node) sub-network yields an empty branch frame, not a crash."""
    n = pypsa.Network()
    n.add("Bus", "iso")
    n.add("Generator", "g", bus="iso", p_nom=10)
    n.determine_network_topology()
    sub = n.c.sub_networks.static.obj.iloc[0]
    zp = sub.calculate_zonal_PTDF(pd.Series({"iso": "Z"}))
    assert zp.empty
    assert list(zp.columns) == ["Z"]


def test_zonal_PTDF_feeds_a_flow_based_constraint():
    """The derived zonal PTDF drops straight into a FlowBasedConstraint."""
    zp = _triangle().calculate_zonal_PTDF(ZONES, gsk="capacity")
    zp.index = [f"{t}_{name}" for t, name in zp.index]  # flat CNEC names
    m = pypsa.Network()
    m.add("Bus", ["A", "B"])
    m.add("FlowBasedConstraint", zp.index, zonal_ptdf=zp, ram=100.0)
    assert list(m.c.flow_based_constraints.zonal_ptdf.columns) == ["A", "B"]
