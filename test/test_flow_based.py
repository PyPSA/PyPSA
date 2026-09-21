# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for the native flow-based market-coupling domain."""

import pandas as pd
import pytest

import pypsa

FB = {"type": "flow_based", "sense": "<="}  # a flow-based global constraint row

ZONES = ["A", "B", "C"]
LOADS = pd.Series({"A": 500.0, "B": 1500.0, "C": 1000.0})
COST = pd.Series({"A": 10.0, "B": 80.0, "C": 50.0})
# equal-reactance triangle PTDF with C as reference; each line monitored both directions
_PTDF = {
    "AB": {"A": 1 / 3, "B": -1 / 3, "C": 0.0},
    "BC": {"A": 1 / 3, "B": 2 / 3, "C": 0.0},
    "AC": {"A": 2 / 3, "B": 1 / 3, "C": 0.0},
}


def _domain(ram: dict[str, float]) -> pd.DataFrame:
    """Signed +/- domain rows PTDF . NP <= RAM for the triangle toy."""
    pos = pd.DataFrame(_PTDF).T
    neg = -pos
    d = pd.concat([pos.rename(lambda s: f"{s}+"), neg.rename(lambda s: f"{s}-")])
    d["RAM"] = [ram[i] for i in d.index]
    return d.sort_index()


def _dynamic_ptdf(
    domain: pd.DataFrame, cols: list, snapshots: pd.Index
) -> pd.DataFrame:
    """Same domain columns repeated across snapshots as a (snapshot, cnec) frame."""
    return pd.concat(dict.fromkeys(snapshots, domain[cols]), names=["snapshot", "name"])


def _add_domain(
    n: pypsa.Network, domain: pd.DataFrame, one_at_a_time: bool, dynamic: bool = False
) -> None:
    """Attach the domain: static (bulk or one CNEC at a time) or repeated time-varying."""
    c = n.c.global_constraints
    if dynamic:
        c.add_flow_based(_dynamic_ptdf(domain, ZONES, n.snapshots), domain["RAM"])
    elif one_at_a_time:
        for cnec in domain.index:
            c.add_flow_based(domain.loc[[cnec], ZONES], domain.loc[[cnec], "RAM"])
    else:
        c.add_flow_based(domain[ZONES], domain["RAM"])


def _network(
    domain: pd.DataFrame, one_at_a_time: bool = False, dynamic: bool = False
) -> pypsa.Network:
    n = pypsa.Network()
    if dynamic:
        n.set_snapshots([0, 1])
    n.add("Bus", ZONES)
    n.add("Load", ZONES, bus=ZONES, p_set=LOADS)
    n.add("Generator", ZONES, bus=ZONES, p_nom=4000, marginal_cost=COST)
    _add_domain(n, domain, one_at_a_time, dynamic)
    return n


def _net_positions(n: pypsa.Network) -> pd.Series:
    return (n.generators_t.p.iloc[0] - LOADS)[ZONES].round(0)


def _np_variable(n: pypsa.Network) -> pd.Series:
    """Solved net-position variable of the first snapshot."""
    return n.model["Bus-net_position"].solution.isel(snapshot=0).to_pandas()


SYMMETRIC: dict[str, float] = {
    "AB+": 1000,
    "AB-": 1000,
    "BC+": 1500,
    "BC-": 1500,
    "AC+": 2000,
    "AC-": 2000,
}


@pytest.mark.parametrize("one_at_a_time", [False, True])
def test_symmetric_domain_reproduces_toy(one_at_a_time):
    """The clearing lands on the AB+ edge with the canonical net positions and prices."""
    n = _network(_domain(SYMMETRIC), one_at_a_time=one_at_a_time)
    n.optimize(log_to_console=False)

    assert _net_positions(n).to_dict() == {"A": 2000.0, "B": -1000.0, "C": -1000.0}
    prices = n.buses_t.marginal_price.iloc[0][ZONES].round(1)
    assert prices.to_dict() == {"A": 10.0, "B": 80.0, "C": 45.0}
    mu = (
        n.model.constraints["GlobalConstraint-flow_based"]
        .dual.isel(snapshot=0)
        .to_pandas()
    )
    binding = mu[mu.abs() > 1e-4].round(1)
    assert binding.index.tolist() == ["AB+"]
    assert binding.iloc[0] == pytest.approx(-105.0)


def test_asymmetric_ram_shifts_the_optimum():
    """A tighter AB+ margin curbs A's export below the symmetric case."""
    n = _network(
        _domain(
            {
                "AB+": 600,
                "AB-": 1000,
                "BC+": 1500,
                "BC-": 1500,
                "AC+": 2000,
                "AC-": 2000,
            }
        )
    )
    n.optimize(log_to_console=False)
    assert _net_positions(n)["A"] < 2000.0


def test_shadow_prices_reproduce_zonal_price_spreads():
    """A meshed asymmetric domain: mu reproduces the zonal price spreads.

    The KKT price identity ``pi_z = lambda + sum_c mu_c PTDF[c,z]`` implies that zonal
    price *differences* equal ``sum_c mu_c (PTDF[c,z1] - PTDF[c,z2])`` (lambda cancels).
    """
    n = _network(
        _domain(
            {
                "AB+": 600,
                "AB-": 1200,
                "BC+": 1500,
                "BC-": 1500,
                "AC+": 1400,
                "AC-": 2000,
            }
        )
    )
    n.optimize(log_to_console=False, assign_all_duals=True)
    c = n.c.global_constraints

    mu = c.dynamic["mu"].iloc[0]  # per-CNEC shadow price via generic assignment
    assert (mu.abs() > 1e-4).any()  # at least one CNEC binds
    assert (mu <= 1e-6).all()  # <= constraint duals are non-positive

    implied = c.zonal_ptdf.T @ mu  # zone -> sum_c mu_c PTDF[c,z]
    prices = n.buses_t.marginal_price.iloc[0][ZONES]
    for z1 in ZONES:
        for z2 in ZONES:
            spread = prices[z1] - prices[z2]
            assert spread == pytest.approx(implied[z1] - implied[z2], abs=1e-3)


def test_dual_assignment_is_clean():
    """mu is assigned (per CNEC); the bus/scalar objects create no junk frames."""
    n = _network(_domain(SYMMETRIC))
    n.optimize(log_to_console=False, assign_all_duals=True)
    dynamic = n.c.global_constraints.dynamic
    assert list(dynamic["mu"].columns) == sorted(SYMMETRIC)  # per-CNEC
    assert "net_position" not in dynamic  # the net position is the bus injection
    assert "mu_balance" not in dynamic  # scalar zero-sum dual has no component slot


@pytest.mark.parametrize(
    ("comp", "kwargs"),
    [
        ("Link", {"bus0": "A", "bus1": "B", "p_nom": 1000}),
        ("Line", {"bus0": "A", "bus1": "B", "x": 0.1, "s_nom": 500}),
        ("Transformer", {"bus0": "A", "bus1": "B", "x": 0.1, "s_nom": 500}),
    ],
)
def test_validation_rejects_cross_zone_branch(comp, kwargs):
    """A branch between two zone buses (not a declared FB column) is rejected."""
    n = _network(_domain(SYMMETRIC))
    n.add(comp, "A-B", **kwargs)
    with pytest.raises(ValueError, match="two zone buses"):
        n.optimize(log_to_console=False)


def _ahc_evfb_network(dynamic: bool = False, zones: list[str] = ZONES):
    """Three zones A,B,C plus external X, with an EvFB link (A-B) and an AHC link (C-X)."""
    n = pypsa.Network()
    if dynamic:
        n.set_snapshots([0, 1])
    n.add("Bus", ["A", "B", "C", "X"])
    n.add(
        "Load",
        ["A", "B", "C", "X"],
        bus=["A", "B", "C", "X"],
        p_set=[500.0, 1500.0, 1000.0, 200.0],
    )
    n.add(
        "Generator",
        ["A", "B", "C"],
        bus=["A", "B", "C"],
        p_nom=4000,
        marginal_cost=[10.0, 80.0, 50.0],
    )
    n.add("Generator", "Xgen", bus="X", p_nom=4000, marginal_cost=15.0)
    n.add("Link", "AB_hvdc", bus0="A", bus1="B", p_nom=800)  # EvFB (two zones)
    n.add("Link", "CX_hvdc", bus0="C", bus1="X", p_nom=600)  # AHC (zone to external)
    d = _domain(SYMMETRIC)
    d["AB_hvdc"], d["CX_hvdc"] = 0.2, 0.15
    cols = [*zones, "AB_hvdc", "CX_hvdc"]
    ptdf = _dynamic_ptdf(d, cols, n.snapshots) if dynamic else d[cols]
    n.c.global_constraints.add_flow_based(ptdf, d["RAM"])
    return n


def test_link_columns_reconstruct_cnec_loading():
    """AHC and EvFB link flows enter the constraint via Link-p in the bus0->bus1 sign.

    Reconstructing each CNEC loading from the zone net positions and the link flows must
    stay within RAM and hit RAM exactly on
    the binding CNECs. The corridor loads its CNECs only through its own column - not also
    smeared through the adjacent zone's net position (no double count).
    """
    n = _ahc_evfb_network()
    n.optimize(log_to_console=False, assign_all_duals=True)
    c = n.c.global_constraints
    zp = c.zonal_ptdf
    zone_cols = [col for col in zp.columns if col in n.c.buses.static.index]
    link_cols = [col for col in zp.columns if col in n.c.links.static.index]
    np_var = _np_variable(n)
    loading = (
        zp[zone_cols] @ np_var[zone_cols]
        + zp[link_cols] @ n.links_t.p0.iloc[0][link_cols]
    )
    ram = c.static["constant"]
    assert (loading <= ram + 1e-6).all()  # feasible
    mu = c.dynamic["mu"].iloc[0]
    for cnec in mu[mu.abs() > 1e-3].index:
        assert loading[cnec] == pytest.approx(ram[cnec], abs=1e-3)  # binding -> at RAM


@pytest.mark.parametrize("zones", [ZONES, ["C", "A", "B"]])
def test_net_position_is_the_bus_net_injection_for_any_zone_order(zones):
    """A zone's net position is gen - load plus corridor inflow, i.e. its bus injection."""
    n = _ahc_evfb_network(zones=zones)
    n.optimize(log_to_console=False)
    net_pos = _np_variable(n)[ZONES].round(3)
    assert net_pos.to_dict() == n.buses_t.p.iloc[0][ZONES].round(3).to_dict()
    gen_load = (n.generators_t.p.iloc[0] - LOADS)[ZONES]
    assert not net_pos.round(0).equals(gen_load.round(0))  # corridors really flow


def test_ahc_link_column_is_the_sensitivity_beyond_the_net_position():
    """An AHC import is counted once: in the zone net position and via the link column.

    Zone A imports from cheap external X. The corridor lands at a node with hub
    sensitivity 0.3 while zone A's PTDF is 0.4, so the link column is 0.3 - 0.4 = -0.1:
    with the import already inside NP_A, the corridor relieves the CNEC slightly. Using
    the raw hub sensitivity 0.3 instead would double count and block the import.
    """
    n = pypsa.Network()
    n.add("Bus", ["A", "B", "X"])
    n.add("Generator", "gA", bus="A", p_nom=1000, marginal_cost=10)
    n.add("Generator", "gX", bus="X", p_nom=1000, marginal_cost=5)
    n.add("Load", ["lA", "lB"], bus=["A", "B"], p_set=[200.0, 800.0])
    n.add("Link", "X-A", bus0="X", bus1="A", p_nom=500, p_min_pu=-1)
    ptdf = pd.DataFrame({"A": [0.4], "B": [-0.6], "X-A": [0.3 - 0.4]}, index=["c1"])
    n.c.global_constraints.add_flow_based(ptdf, pd.Series(800.0, ptdf.index))
    n.optimize(log_to_console=False)

    assert n.links_t.p0.iloc[0]["X-A"] == pytest.approx(500.0)  # import flows
    assert n.objective == pytest.approx(7500.0)  # cheap import used, not local gen
    net_pos = _np_variable(n)
    assert net_pos["A"] == pytest.approx(800.0)  # gen 500 - load 200 + import 500
    assert net_pos["A"] == pytest.approx(n.buses_t.p.iloc[0]["A"])
    assert net_pos["A"] + net_pos["B"] == pytest.approx(0.0)


def test_ahc_export_enters_the_net_position_and_the_plate_closes():
    """Zone A exports over an AHC link on bus0: NP_A = gen - load - export, sum(NP) = 0."""
    n = pypsa.Network()
    n.add("Bus", ["A", "B", "X"])
    n.add(
        "Generator", ["gA", "gB"], bus=["A", "B"], p_nom=2000, marginal_cost=[5.0, 50.0]
    )
    n.add("Load", ["lA", "lB", "lX"], bus=["A", "B", "X"], p_set=[100.0, 500.0, 400.0])
    n.add("Link", "A-X", bus0="A", bus1="X", p_nom=500, p_min_pu=-1)
    ptdf = pd.DataFrame({"A": [0.3], "B": [-0.3], "A-X": [0.2]}, index=["c1"])
    n.c.global_constraints.add_flow_based(ptdf, pd.Series(5000.0, ptdf.index))
    n.optimize(log_to_console=False)

    net_pos = _np_variable(n)
    genA = n.generators_t.p.iloc[0]["gA"]
    F = n.links_t.p0.iloc[0]["A-X"]
    assert F == pytest.approx(400.0)  # cheap A serves the external load
    assert net_pos["A"] == pytest.approx(genA - 100.0 - F)
    assert net_pos[["A", "B"]].sum() == pytest.approx(0.0)


def test_time_varying_constant_only_for_flow_based_rows():
    """Other global constraint types have one constant per row, not per snapshot."""
    n = _network(_domain(SYMMETRIC), dynamic=True)
    n.add(
        "GlobalConstraint",
        "co2",
        sense="<=",
        constant=pd.Series([1.0, 2.0], n.snapshots),
    )
    with pytest.raises(ValueError, match="time-varying `constant`"):
        n.optimize(log_to_console=False)


def test_link_column_without_zone_end_raises():
    """A link column must touch the flow-based region (AHC: one zone end, EvFB: two)."""
    n = _network(_domain(SYMMETRIC))
    n.add("Bus", ["X", "Y"])
    n.add("Link", "XY", bus0="X", bus1="Y", p_nom=1000)
    c = n.c.global_constraints
    c.add_flow_based(c.zonal_ptdf.assign(XY=0.1), c.static["constant"], overwrite=True)
    with pytest.raises(ValueError, match="no flow-based zone"):
        n.optimize(log_to_console=False)


def test_unknown_domain_column_raises():
    """A domain column that is neither a bus nor a link fails fast at build time."""
    n = _network(_domain(SYMMETRIC))
    n.c.global_constraints.static["ptdf_ghost"] = 0.1  # not a bus or link
    with pytest.raises(ValueError, match="neither"):
        n.optimize(log_to_console=False)


def test_bus_takes_priority_over_link_on_name_clash():
    """A column that names both a bus and a link is treated as a zone (net position)."""
    n = pypsa.Network()
    n.add("Bus", [*ZONES, "gas"])
    n.add("Link", "C", bus0="A", bus1="gas", p_nom=100)  # link named like bus "C"
    n.add("Load", ZONES, bus=ZONES, p_set=LOADS)
    n.add("Generator", ZONES, bus=ZONES, p_nom=4000, marginal_cost=COST)
    d = _domain(SYMMETRIC)
    n.c.global_constraints.add_flow_based(d[ZONES], d["RAM"])
    n.optimize(log_to_console=False)
    assert "C" in n.model["Bus-net_position"].indexes["name"]  # zone, not link


def test_non_zone_link_is_allowed():
    """A link to a non-zone bus (e.g. a gas pipeline) does not trip validation."""
    n = _network(_domain(SYMMETRIC))
    n.add("Bus", "gas")
    n.add("Link", "A-gas", bus0="A", bus1="gas", p_nom=1000)
    n.optimize(log_to_console=False)
    assert _net_positions(n)["A"] == pytest.approx(2000.0)


def test_zonal_ptdf_views_are_pandas_and_xarray():
    """Zonal PTDF is public pandas (cnec x zone) and internal xarray (name, bus)."""
    c = _network(_domain(SYMMETRIC)).c.global_constraints
    assert isinstance(c.zonal_ptdf, pd.DataFrame)
    assert list(c.zonal_ptdf.columns) == ZONES
    assert c.zonal_ptdf.loc["AB+", "A"] == pytest.approx(1 / 3)
    assert set(c.da.zonal_ptdf.dims) == {"name", "bus"}


def test_readd_with_overwrite_replaces_the_row():
    """Re-adding a CNEC with overwrite updates its PTDF and RAM without duplicating it."""
    n = _network(_domain(SYMMETRIC))
    c = n.c.global_constraints
    row = pd.DataFrame({"A": 0.9, "B": 0.0, "C": 0.0}, index=["AB+"])
    c.add_flow_based(row, pd.Series(99.0, row.index), overwrite=True)
    assert len(c.static) == 6  # not duplicated
    assert c.zonal_ptdf.loc["AB+", "A"] == pytest.approx(0.9)
    assert c.static.loc["AB+", "constant"] == pytest.approx(99.0)


@pytest.mark.parametrize("dynamic", [False, True])
def test_incremental_add_unions_zones_with_zero_fill(dynamic):
    """CNECs added separately with different zone sets share one zero-filled frame."""
    n = pypsa.Network()
    n.add("Bus", ZONES)
    rows = {"c1": {"A": 0.5, "B": -0.5}, "c2": {"A": 0.2, "C": 0.3}}
    for cnec, ptdf in rows.items():
        row = pd.DataFrame(ptdf, index=[cnec])
        if dynamic:
            row = _dynamic_ptdf(row, list(ptdf), n.snapshots)
        n.c.global_constraints.add_flow_based(row, pd.Series(100.0, [cnec]))
    z = n.c.global_constraints.zonal_ptdf
    if dynamic:
        z = z.loc[n.snapshots[0]]
    assert set(z.columns) == set(ZONES)
    assert z.loc["c1", "C"] == 0.0  # zone absent for c1 -> zero sensitivity
    assert z.loc["c2", "B"] == 0.0
    assert z.loc["c2", "A"] == 0.2


def test_no_domain_optimizes_normally():
    """Without a flow-based domain the machinery is a clean no-op."""
    n = pypsa.Network()
    n.add("Bus", "b")
    n.add("Load", "l", bus="b", p_set=10.0)
    n.add("Generator", "g", bus="b", p_nom=20.0, marginal_cost=5.0)
    n.optimize(log_to_console=False)
    assert "GlobalConstraint-flow_based" not in n.model.constraints
    assert n.generators_t.p.iloc[0]["g"] == pytest.approx(10.0)


def test_copy_preserves_domain_and_re_solves():
    """n.copy() carries the zonal PTDF columns; the copy solves independently."""
    n = _network(_domain(SYMMETRIC))
    m = n.copy()
    assert m.c.global_constraints.zonal_ptdf.equals(n.c.global_constraints.zonal_ptdf)
    m.optimize(log_to_console=False)
    assert _net_positions(m).to_dict() == {"A": 2000.0, "B": -1000.0, "C": -1000.0}


def _static_np(domain: pd.DataFrame) -> pd.Series:
    """Net positions of a single-snapshot solve of a static domain."""
    n = _network(domain)
    n.optimize(log_to_console=False)
    return n.buses_t.p.iloc[0][ZONES].round(0)


def _dynamic_network(dA: pd.DataFrame, dB: pd.DataFrame) -> pypsa.Network:
    """Two-snapshot toy whose zonal PTDF is ``dA`` in hour 0 and ``dB`` in hour 1."""
    n = pypsa.Network()
    n.set_snapshots([0, 1])
    n.add("Bus", ZONES)
    n.add("Load", ZONES, bus=ZONES, p_set=LOADS)
    n.add("Generator", ZONES, bus=ZONES, p_nom=4000, marginal_cost=COST)
    ptdf = pd.concat({0: dA[ZONES], 1: dB[ZONES]}, names=["snapshot", "name"])
    n.c.global_constraints.add_flow_based(ptdf, dA["RAM"])
    return n


def test_time_varying_zonal_ptdf_frontend_and_da():
    """A time-varying PTDF is public pandas (MultiIndex) and internal xarray (+snapshot)."""
    dA = _domain(SYMMETRIC)
    dB = dA.copy()
    dB[ZONES] = dB[ZONES] * 2.0
    c = _dynamic_network(dA, dB).c.global_constraints
    assert isinstance(c.zonal_ptdf.index, pd.MultiIndex)
    assert list(c.zonal_ptdf.columns) == ZONES
    assert set(c.da.zonal_ptdf.dims) == {"snapshot", "name", "bus"}
    pd.testing.assert_frame_equal(
        c.zonal_ptdf.loc[0][ZONES],
        dA[ZONES],
        check_like=True,
        check_names=False,
        check_index_type=False,
    )


def test_snapshot_subset_copy_keeps_time_varying_ptdf():
    """n.copy(snapshots=...) keeps the PTDF of the selected hour, like any dynamic attribute."""
    dA = _domain(SYMMETRIC)
    dB = dA.copy()
    dB[ZONES] = dB[ZONES] * 2.0
    m = _dynamic_network(dA, dB).copy(snapshots=[1])
    pd.testing.assert_frame_equal(
        m.c.global_constraints.zonal_ptdf.loc[1][ZONES],
        dB[ZONES],
        check_names=False,
        check_index_type=False,
    )


def test_static_and_time_varying_zone_columns_mix():
    """A time-varying column for one zone broadcasts against static columns of the others."""
    dA = _domain(SYMMETRIC)
    dB = dA.copy()
    dB["A"] = dB["A"] * 2.0  # only zone A changes in hour 1
    ref = _dynamic_network(dA, dB)
    ref.optimize(log_to_console=False)

    n = _dynamic_network(dA, dA)
    n.remove("GlobalConstraint", dA.index)
    ptdf_A = pd.DataFrame({0: dA["A"], 1: dB["A"]}).T.rename_axis(index="snapshot")
    n.add(  # the raw form: one ptdf_<zone> attribute per zone, static or time-varying
        "GlobalConstraint",
        dA.index,
        **FB,
        constant=dA["RAM"].values,
        **dA[["B", "C"]].add_prefix("ptdf_"),
        ptdf_A=ptdf_A,
    )
    n.optimize(log_to_console=False)
    pd.testing.assert_frame_equal(n.buses_t.p, ref.buses_t.p)


def test_time_varying_zonal_ptdf_matches_per_snapshot_static():
    """Each hour clears exactly like a static domain built from that hour's PTDF."""
    dA = _domain(SYMMETRIC)
    dB = dA.copy()
    dB[ZONES] = dB[ZONES] * 2.0  # tighter half-spaces in hour 1
    refA, refB = _static_np(dA), _static_np(dB)
    assert not refA.equals(refB)  # the two hours really differ

    n = _dynamic_network(dA, dB)
    n.optimize(log_to_console=False)
    npos = n.buses_t.p[ZONES].round(0)
    assert npos.loc[0].equals(refA)
    assert npos.loc[1].equals(refB)


def test_time_varying_zonal_ptdf_round_trips(tmp_path):
    """Export/import through netCDF preserves the time-varying frame and clearing."""
    dA = _domain(SYMMETRIC)
    dB = dA.copy()
    dB[ZONES] = dB[ZONES] * 2.0
    n = _dynamic_network(dA, dB)
    path = tmp_path / "dynamic.nc"
    n.export_to_netcdf(path)
    m = pypsa.Network(path)
    pd.testing.assert_frame_equal(
        m.c.global_constraints.zonal_ptdf,
        n.c.global_constraints.zonal_ptdf,
        check_index_type=False,  # cnec labels come back object vs StringDtype (PyPSA-wide)
    )
    m.optimize(log_to_console=False)  # the recovered domain still solves


def test_investment_period_selects_the_domain():
    """Rows with an investment period only bind in that period's snapshots."""
    d2030, d2040 = _domain({**SYMMETRIC, "AB+": 600}), _domain(SYMMETRIC)
    n = pypsa.Network()
    n.set_snapshots(pd.MultiIndex.from_product([[2030, 2040], [0]]))
    n.set_investment_periods([2030, 2040])
    n.add("Bus", ZONES)
    n.add("Load", ZONES, bus=ZONES, p_set=LOADS)
    n.add("Generator", ZONES, bus=ZONES, p_nom=4000, marginal_cost=COST)
    for year, d in [(2030, d2030), (2040, d2040)]:
        d = d.set_axis(d.index + f" {year}")
        n.c.global_constraints.add_flow_based(
            d[ZONES], d["RAM"], investment_period=year
        )
    n.optimize(log_to_console=False, multi_investment_periods=True)

    npos = n.buses_t.p[ZONES].round(0)
    assert npos.loc[(2030, 0)].equals(_static_np(d2030))
    assert npos.loc[(2040, 0)].equals(_static_np(d2040))


def test_unknown_investment_period_raises():
    """A flow-based row tied to a period the network does not have fails fast."""
    n = _network(_domain(SYMMETRIC))
    n.set_snapshots(pd.MultiIndex.from_product([[2030], [0]]))
    n.set_investment_periods([2030])
    n.c.global_constraints.static["investment_period"] = 2050.0
    with pytest.raises(ValueError, match="investment periods"):
        n.optimize(log_to_console=False, multi_investment_periods=True)


def test_time_varying_ram_matches_per_snapshot_static():
    """A per-snapshot RAM clears each hour like a static domain with that hour's RAM."""
    d0 = _domain(SYMMETRIC)
    d1 = _domain({**SYMMETRIC, "AB+": 600})  # tighter AB+ margin in hour 1
    ref0, ref1 = _static_np(d0), _static_np(d1)
    assert not ref0.equals(ref1)

    n = pypsa.Network()
    n.set_snapshots([0, 1])
    n.add("Bus", ZONES)
    n.add("Load", ZONES, bus=ZONES, p_set=LOADS)
    n.add("Generator", ZONES, bus=ZONES, p_nom=4000, marginal_cost=COST)
    ram = pd.DataFrame({0: d0["RAM"], 1: d1["RAM"]}).T  # snapshot x cnec
    n.c.global_constraints.add_flow_based(d0[ZONES], ram)
    n.optimize(log_to_console=False, assign_all_duals=True)

    npos = n.buses_t.p[ZONES].round(0)
    assert npos.loc[0].equals(ref0)
    assert npos.loc[1].equals(ref1)
    # one dual per snapshot
    assert len(n.c.global_constraints.dynamic["mu"]) == 2


def test_time_varying_ptdf_with_link_column_broadcasts():
    """A time-varying PTDF carrying an EvFB link column broadcasts per hour."""
    dA = _domain(SYMMETRIC)
    dA["AB_hvdc"] = 0.2
    dB = dA.copy()
    dB[[*ZONES, "AB_hvdc"]] *= 2.0  # tighter half-spaces in hour 1
    cols = [*ZONES, "AB_hvdc"]

    n = pypsa.Network()
    n.set_snapshots([0, 1])
    n.add("Bus", ZONES)
    n.add("Load", ZONES, bus=ZONES, p_set=LOADS)
    n.add("Generator", ZONES, bus=ZONES, p_nom=4000, marginal_cost=COST)
    n.add("Link", "AB_hvdc", bus0="A", bus1="B", p_nom=800)  # EvFB (two zones)
    ptdf = pd.concat({0: dA[cols], 1: dB[cols]}, names=["snapshot", "name"])
    n.c.global_constraints.add_flow_based(ptdf, dA["RAM"])
    n.optimize(log_to_console=False)

    npos = n.model["Bus-net_position"].solution.to_pandas()[ZONES].round(0)
    pd.testing.assert_frame_equal(
        npos, n.buses_t.p[ZONES].round(0), check_names=False, check_column_type=False
    )
    assert not npos.loc[0].equals(npos.loc[1])  # the link-loaded hours really differ


@pytest.mark.parametrize(
    "make",
    [lambda dynamic: _network(_domain(SYMMETRIC), dynamic=dynamic), _ahc_evfb_network],
    ids=["zone-domain", "link-domain"],
)
def test_time_varying_reproduces_static(make):
    """A domain repeated across snapshots reproduces the static net positions and duals."""
    ref = make(False)
    ref.optimize(log_to_console=False, assign_all_duals=True)
    n = make(True)
    n.optimize(log_to_console=False, assign_all_duals=True)

    ref_np = ref.buses_t.p.iloc[0][ZONES].round(0)
    ref_mu = ref.c.global_constraints.dynamic["mu"].iloc[0].round(3)
    mu = n.c.global_constraints.dynamic["mu"]
    for sns in n.snapshots:
        assert n.buses_t.p.loc[sns][ZONES].round(0).equals(ref_np)
        assert mu.loc[sns].round(3).equals(ref_mu)
