# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for the native flow-based market-coupling domain."""

import pandas as pd
import pytest

import pypsa

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
    if dynamic:
        ptdf = _dynamic_ptdf(domain, ZONES, n.snapshots)
        n.add(
            "FlowBasedDomain", domain.index, zonal_ptdf=ptdf, ram=domain["RAM"].values
        )
        return
    if one_at_a_time:
        for cnec in domain.index:
            n.add(
                "FlowBasedDomain",
                cnec,
                zonal_ptdf=domain.loc[cnec, ZONES],
                ram=float(domain.loc[cnec, "RAM"]),
            )
    else:
        n.add(
            "FlowBasedDomain",
            domain.index,
            zonal_ptdf=domain[ZONES],
            ram=domain["RAM"].values,
        )


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


SYMMETRIC = {"AB+": 1000, "AB-": 1000, "BC+": 1500, "BC-": 1500, "AC+": 2000, "AC-": 2000}


@pytest.mark.parametrize("one_at_a_time", [False, True])
def test_symmetric_domain_reproduces_toy(one_at_a_time):
    """The clearing lands on the AB+ edge with the canonical net positions and prices."""
    n = _network(_domain(SYMMETRIC), one_at_a_time=one_at_a_time)
    n.optimize(log_to_console=False)

    assert _net_positions(n).to_dict() == {"A": 2000.0, "B": -1000.0, "C": -1000.0}
    prices = n.buses_t.marginal_price.iloc[0][ZONES].round(1)
    assert prices.to_dict() == {"A": 10.0, "B": 80.0, "C": 45.0}
    mu = n.model.constraints["FlowBasedDomain-domain"].dual.isel(snapshot=0).to_pandas()
    binding = mu[mu.abs() > 1e-4].round(1)
    assert binding.index.tolist() == ["AB+"]
    assert binding.iloc[0] == pytest.approx(-105.0)


def test_net_position_output_matches_domain_variable():
    """buses_t.net_position exposes the domain net position; equals buses_t.p without corridors."""
    n = _network(_domain(SYMMETRIC))
    n.optimize(log_to_console=False)
    np_out = n.buses_t.net_position.iloc[0][ZONES].round(0)
    assert np_out.to_dict() == {"A": 2000.0, "B": -1000.0, "C": -1000.0}
    assert (np_out - n.buses_t.p.iloc[0][ZONES]).abs().max() == pytest.approx(0.0)


def test_prices_come_from_nodal_balance():
    """No auxiliary components: zonal prices are the nodal-balance duals, not reconstructed."""
    n = _network(_domain(SYMMETRIC))
    n.optimize(log_to_console=False)
    assert not n.buses_t.marginal_price.empty
    # net positions sum to zero (global balance)
    assert _net_positions(n).sum() == pytest.approx(0.0)


def test_asymmetric_ram_shifts_the_optimum():
    """A tighter AB+ margin curbs A's export below the symmetric case."""
    n = _network(_domain({"AB+": 600, "AB-": 1000, "BC+": 1500, "BC-": 1500, "AC+": 2000, "AC-": 2000}))
    n.optimize(log_to_console=False)
    assert _net_positions(n)["A"] < 2000.0


def test_shadow_prices_reproduce_zonal_price_spreads():
    """A meshed asymmetric domain: mu_domain reproduces the zonal price spreads.

    The KKT price identity ``pi_z = lambda + sum_c mu_c PTDF[c,z]`` implies that zonal
    price *differences* equal ``sum_c mu_c (PTDF[c,z1] - PTDF[c,z2])`` (lambda cancels).
    """
    n = _network(
        _domain(
            {"AB+": 600, "AB-": 1200, "BC+": 1500, "BC-": 1500, "AC+": 1400, "AC-": 2000}
        )
    )
    n.optimize(log_to_console=False, assign_all_duals=True)
    c = n.c.flow_based_domains

    mu = c.dynamic["mu_domain"].iloc[0]  # per-CNEC shadow price via generic assignment
    assert (mu.abs() > 1e-4).any()  # at least one CNEC binds
    assert (mu <= 1e-6).all()  # <= constraint duals are non-positive

    implied = c.zonal_ptdf.T @ mu  # zone -> sum_c mu_c PTDF[c,z]
    prices = n.buses_t.marginal_price.iloc[0][ZONES]
    for z1 in ZONES:
        for z2 in ZONES:
            spread = prices[z1] - prices[z2]
            assert spread == pytest.approx(implied[z1] - implied[z2], abs=1e-3)


def test_dual_assignment_is_clean():
    """mu_domain is assigned (per CNEC); the bus/scalar objects create no junk frames."""
    n = _network(_domain(SYMMETRIC))
    n.optimize(log_to_console=False, assign_all_duals=True)
    dynamic = n.c.flow_based_domains.dynamic
    assert list(dynamic["mu_domain"].columns) == sorted(SYMMETRIC)  # per-CNEC
    assert "net_position" not in dynamic  # net position lives in n.buses_t.net_position
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


def _ahc_evfb_network(dynamic: bool = False):
    """Three zones A,B,C plus external X, with an EvFB link (A-B) and an AHC link (C-X)."""
    n = pypsa.Network()
    if dynamic:
        n.set_snapshots([0, 1])
    n.add("Bus", ["A", "B", "C", "X"])
    n.add("Load", ["A", "B", "C", "X"], bus=["A", "B", "C", "X"], p_set=[500.0, 1500.0, 1000.0, 200.0])
    n.add("Generator", ["A", "B", "C"], bus=["A", "B", "C"], p_nom=4000, marginal_cost=[10.0, 80.0, 50.0])
    n.add("Generator", "Xgen", bus="X", p_nom=4000, marginal_cost=15.0)
    n.add("Link", "AB_hvdc", bus0="A", bus1="B", p_nom=800)  # EvFB (two zones)
    n.add("Link", "CX_hvdc", bus0="C", bus1="X", p_nom=600)  # AHC (zone to external)
    d = _domain(SYMMETRIC)
    d["AB_hvdc"], d["CX_hvdc"] = 0.2, 0.15
    cols = [*ZONES, "AB_hvdc", "CX_hvdc"]
    ptdf = _dynamic_ptdf(d, cols, n.snapshots) if dynamic else d[cols]
    n.c.flow_based_domains.add(d.index, zonal_ptdf=ptdf, ram=d["RAM"].values)
    return n


def test_link_columns_reconstruct_cnec_loading():
    """AHC and EvFB link flows enter the constraint via Link-p in the bus0->bus1 sign.

    Reconstructing each CNEC loading from the zone net positions (gen - load, read from the
    net-position variable) and the link flows must stay within RAM and hit RAM exactly on
    the binding CNECs. The corridor loads its CNECs only through its own column - not also
    smeared through the adjacent zone's net position (no double count).
    """
    n = _ahc_evfb_network()
    n.optimize(log_to_console=False, assign_all_duals=True)
    c = n.c.flow_based_domains
    zp = c.zonal_ptdf
    zone_cols = [col for col in zp.columns if col in n.buses.index]
    link_cols = [col for col in zp.columns if col in n.links.index]
    np_var = n.model["FlowBasedDomain-net_position"].solution.isel(snapshot=0).to_pandas()
    # the net position is gen - load, unaffected by the corridor flows
    gen_load = (n.generators_t.p.iloc[0].groupby(n.generators.bus).sum() - n.loads_t.p.iloc[0].groupby(n.loads.bus).sum())
    assert np_var[zone_cols].round(1).to_dict() == gen_load[zone_cols].round(1).to_dict()
    loading = zp[zone_cols] @ np_var[zone_cols] + zp[link_cols] @ n.links_t.p0.iloc[0][link_cols]
    ram = c.static["ram"]
    assert (loading <= ram + 1e-6).all()  # feasible
    mu = c.dynamic["mu_domain"].iloc[0]
    for cnec in mu[mu.abs() > 1e-3].index:
        assert loading[cnec] == pytest.approx(ram[cnec], abs=1e-3)  # binding -> at RAM


def test_ahc_import_not_double_counted():
    """A binding corridor CNEC must not phantom-block cheap AHC imports (no leak).

    Zone A imports from cheap external X over an AHC border; a single CNEC binds. Under the
    old leak the corridor loaded the CNEC at ptdf_A + ptdf_link (0.7) and blocked the
    import; with the Core-side term cut it loads once (0.3), so the import flows.
    """
    n = pypsa.Network()
    n.add("Bus", ["A", "B", "X"])
    n.add("Generator", "gA", bus="A", p_nom=1000, marginal_cost=10)
    n.add("Generator", "gX", bus="X", p_nom=1000, marginal_cost=5)
    n.add("Load", ["lA", "lB"], bus=["A", "B"], p_set=[200.0, 800.0])
    n.add("Link", "X-A", bus0="X", bus1="A", p_nom=500, p_min_pu=-1)
    ptdf = pd.DataFrame({"A": [0.4], "B": [-0.6], "X-A": [0.3]}, index=pd.Index(["c1"], name="name"))
    n.c.flow_based_domains.add("c1", zonal_ptdf=ptdf, ram=800.0)
    n.optimize(log_to_console=False)

    assert n.links_t.p0.iloc[0]["X-A"] == pytest.approx(500.0)  # import flows (was blocked)
    assert n.objective == pytest.approx(7500.0)  # cheap import used, not local gen
    np_var = n.model["FlowBasedDomain-net_position"].solution.isel(snapshot=0).to_pandas()
    assert np_var["A"] == pytest.approx(300.0)  # gen - load, no corridor leak


def test_ahc_export_keeps_net_position_and_plate_sign():
    """Core exports over an AHC border on bus0: net position stays gen-load, plate closes."""
    n = pypsa.Network()
    n.add("Bus", ["A", "B", "X"])
    n.add("Generator", ["gA", "gB"], bus=["A", "B"], p_nom=2000, marginal_cost=[5.0, 50.0])
    n.add("Load", ["lA", "lB", "lX"], bus=["A", "B", "X"], p_set=[100.0, 500.0, 400.0])
    n.add("Link", "A-X", bus0="A", bus1="X", p_nom=500, p_min_pu=-1)  # Core (bus0) -> external
    ptdf = pd.DataFrame({"A": [0.3], "B": [-0.3], "A-X": [0.2]}, index=pd.Index(["c1"], name="name"))
    n.c.flow_based_domains.add("c1", zonal_ptdf=ptdf, ram=5000.0)
    n.optimize(log_to_console=False)

    np_var = n.model["FlowBasedDomain-net_position"].solution.isel(snapshot=0).to_pandas()
    genA = n.generators_t.p.iloc[0]["gA"]
    F = n.links_t.p0.iloc[0]["A-X"]
    assert F == pytest.approx(400.0)  # cheap A serves the external load
    assert np_var["A"] == pytest.approx(genA - 100.0)  # gen - load, no leak
    assert np_var.sum() == pytest.approx(F)  # plate: Core exports F over the border


def test_evfb_stays_off_the_plate():
    """An internal EvFB corridor moves no zonal energy: sum(NP)=0 and NP stays gen-load."""
    n = pypsa.Network()
    n.add("Bus", ZONES)
    n.add("Load", ZONES, bus=ZONES, p_set=LOADS)
    n.add("Generator", ZONES, bus=ZONES, p_nom=4000, marginal_cost=COST)
    n.add("Link", "AB", bus0="A", bus1="B", p_nom=800)  # EvFB (both ends zones)
    d = _domain(SYMMETRIC)
    d["AB"] = 0.2
    n.c.flow_based_domains.add(d.index, zonal_ptdf=d[[*ZONES, "AB"]], ram=d["RAM"].values)
    n.optimize(log_to_console=False)

    np_var = n.model["FlowBasedDomain-net_position"].solution.isel(snapshot=0).to_pandas()
    assert np_var.sum() == pytest.approx(0.0)  # internal corridor -> no net Core exchange
    assert np_var[ZONES].round(0).tolist() == _net_positions(n).tolist()  # gen - load


def test_fully_external_link_column_is_constrained_not_cut():
    """A link with neither end a zone loads the CNECs via its column but touches no NP."""
    n = pypsa.Network()
    n.add("Bus", [*ZONES, "X", "Y"])
    n.add("Load", ZONES, bus=ZONES, p_set=LOADS)
    n.add("Generator", ZONES, bus=ZONES, p_nom=4000, marginal_cost=COST)
    n.add("Generator", ["gX", "gY"], bus=["X", "Y"], p_nom=1000, marginal_cost=[1.0, 100.0])
    n.add("Load", "lY", bus="Y", p_set=300.0)
    n.add("Link", "XY", bus0="X", bus1="Y", p_nom=1000, p_min_pu=-1)  # fully external
    d = _domain(SYMMETRIC)
    d["XY"] = 0.0
    d.loc["ext"] = {"A": 0.0, "B": 0.0, "C": 0.0, "XY": 1.0, "RAM": 200.0}
    n.c.flow_based_domains.add(d.index, zonal_ptdf=d[[*ZONES, "XY"]], ram=d["RAM"].values)
    n.optimize(log_to_console=False)

    assert n.links_t.p0.iloc[0]["XY"] == pytest.approx(200.0)  # capped by its own CNEC
    assert _net_positions(n).to_dict() == {"A": 2000.0, "B": -1000.0, "C": -1000.0}
    np_var = n.model["FlowBasedDomain-net_position"].solution.isel(snapshot=0).to_pandas()
    assert np_var.sum() == pytest.approx(0.0)  # external link is not on the plate


def test_buses_p_is_physical_injection_hub_np_is_link_flow():
    """With a corridor, buses_t.p is the physical injection; the hub NP is read from Link-p0."""
    n = pypsa.Network()
    n.add("Bus", ["A", "B", "X"])
    n.add("Generator", ["gA", "gX"], bus=["A", "X"], p_nom=1000, marginal_cost=[10.0, 5.0])
    n.add("Load", ["lA", "lB"], bus=["A", "B"], p_set=[200.0, 800.0])
    n.add("Link", "X-A", bus0="X", bus1="A", p_nom=500, p_min_pu=-1)
    ptdf = pd.DataFrame({"A": [0.4], "B": [-0.6], "X-A": [0.3]}, index=pd.Index(["c1"], name="name"))
    n.c.flow_based_domains.add("c1", zonal_ptdf=ptdf, ram=800.0)
    n.optimize(log_to_console=False)

    np_var = n.model["FlowBasedDomain-net_position"].solution.isel(snapshot=0).to_pandas()
    assert np_var["A"] == pytest.approx(300.0)  # domain net position = gen - load
    assert n.buses_t.net_position.iloc[0]["A"] == pytest.approx(300.0)  # exposed output = NP var
    assert n.buses_t.p.iloc[0]["A"] == pytest.approx(800.0)  # physical injection = gen-load+import
    assert n.links_t.p0.iloc[0]["X-A"] == pytest.approx(500.0)  # virtual hub NP = link flow
    assert "X" not in n.buses_t.net_position.columns[n.buses_t.net_position.iloc[0] != 0]  # zones only


def test_evfb_cross_zone_link_column_is_allowed():
    """A cross-zone link that is a declared domain column (EvFB) passes validation."""
    n = _ahc_evfb_network()  # AB_hvdc connects zones A and B and is a column
    n.optimize(log_to_console=False)
    assert "FlowBasedDomain-domain" in n.model.constraints


def test_unknown_domain_column_raises():
    """A domain column that is neither a bus nor a link fails fast at build time."""
    n = _network(_domain(SYMMETRIC))
    n.c.flow_based_domains.zonal_ptdf["ghost"] = 0.1  # not a bus or link
    with pytest.raises(ValueError, match="neither"):
        n.optimize(log_to_console=False)


def test_bus_takes_priority_over_link_on_name_clash():
    """A column that names both a bus and a link is treated as a zone (net position)."""
    n = pypsa.Network()
    n.add("Bus", [*ZONES, "gas"])
    n.add("Link", "C", bus0="A", bus1="gas", p_nom=100)  # link named like bus "C"
    n.add("Load", ZONES, bus=ZONES, p_set=LOADS)
    n.add("Generator", ZONES, bus=ZONES, p_nom=4000, marginal_cost=COST)
    n.c.flow_based_domains.add(_domain(SYMMETRIC).index, zonal_ptdf=_domain(SYMMETRIC)[ZONES], ram=_domain(SYMMETRIC)["RAM"].values)
    n.optimize(log_to_console=False)
    assert "C" in n.model["FlowBasedDomain-net_position"].indexes["bus"]  # zone, not link


def test_non_zone_link_is_allowed():
    """A link to a non-zone bus (e.g. a gas pipeline) does not trip validation."""
    n = _network(_domain(SYMMETRIC))
    n.add("Bus", "gas")
    n.add("Link", "A-gas", bus0="A", bus1="gas", p_nom=1000)
    n.optimize(log_to_console=False)
    assert _net_positions(n)["A"] == pytest.approx(2000.0)


def test_inactive_domain_is_ignored():
    """Deactivating all constraints leaves an unconstrained copper-plate clearing."""
    n = _network(_domain(SYMMETRIC))
    n.flow_based_domains["active"] = False
    n.optimize(log_to_console=False)
    # cheapest generator (A) serves all demand; no binding domain
    assert "FlowBasedDomain-domain" not in n.model.constraints


def test_zonal_ptdf_views_are_pandas_and_xarray():
    """Zonal PTDF is public pandas (cnec x zone) and internal xarray (name, bus)."""
    c = _network(_domain(SYMMETRIC)).c.flow_based_domains
    assert isinstance(c.zonal_ptdf, pd.DataFrame)
    assert list(c.zonal_ptdf.columns) == ZONES
    assert c.zonal_ptdf.loc["AB+", "A"] == pytest.approx(1 / 3)
    assert set(c.da.zonal_ptdf.dims) == {"name", "bus"}


def test_deactivating_one_cnec_relaxes_the_domain():
    """An inactive CNEC drops out of the constraint and no longer limits trade."""
    n = _network(_domain(SYMMETRIC))
    n.c.flow_based_domains.static.loc["AB+", "active"] = False
    n.optimize(log_to_console=False)
    con = n.model.constraints["FlowBasedDomain-domain"]
    assert "AB+" not in con.dual.indexes["name"]  # excluded from the constraint
    assert _net_positions(n)["A"] > 2000.0  # AB+ no longer binds -> more export


def test_readd_with_overwrite_replaces_the_row():
    """Re-adding a CNEC with overwrite updates its PTDF and RAM without duplicating it."""
    n = _network(_domain(SYMMETRIC))
    c = n.c.flow_based_domains
    n.add(
        "FlowBasedDomain",
        "AB+",
        zonal_ptdf=pd.Series({"A": 0.9, "B": 0.0, "C": 0.0}),
        ram=99.0,
        overwrite=True,
    )
    assert len(c.static) == 6  # not duplicated
    assert c.zonal_ptdf.loc["AB+", "A"] == pytest.approx(0.9)
    assert c.static.loc["AB+", "ram"] == pytest.approx(99.0)


def test_incremental_add_unions_zones_with_zero_fill():
    """CNECs added separately with different zone sets share one zero-filled frame."""
    n = pypsa.Network()
    n.add("Bus", ZONES)
    n.c.flow_based_domains.add("c1", zonal_ptdf=pd.Series({"A": 0.5, "B": -0.5}), ram=100.0)
    n.c.flow_based_domains.add("c2", zonal_ptdf=pd.Series({"A": 0.2, "C": 0.3}), ram=200.0)
    z = n.c.flow_based_domains.zonal_ptdf
    assert set(z.columns) == set(ZONES)
    assert z.loc["c1", "C"] == 0.0  # zone absent for c1 -> zero sensitivity
    assert z.loc["c2", "B"] == 0.0


def test_no_domain_optimizes_normally():
    """Without a flow-based domain the machinery is a clean no-op."""
    n = pypsa.Network()
    n.add("Bus", "b")
    n.add("Load", "l", bus="b", p_set=10.0)
    n.add("Generator", "g", bus="b", p_nom=20.0, marginal_cost=5.0)
    n.optimize(log_to_console=False)
    assert "FlowBasedDomain-domain" not in n.model.constraints
    assert n.generators_t.p.iloc[0]["g"] == pytest.approx(10.0)


def test_copy_preserves_domain_and_re_solves():
    """n.copy() deep-copies the zonal PTDF store; the copy solves independently."""
    n = _network(_domain(SYMMETRIC))
    m = n.copy()
    assert m.c.flow_based_domains.zonal_ptdf.equals(n.c.flow_based_domains.zonal_ptdf)
    assert m.c.flow_based_domains.zonal_ptdf is not n.c.flow_based_domains.zonal_ptdf
    m.optimize(log_to_console=False)
    assert _net_positions(m).to_dict() == {"A": 2000.0, "B": -1000.0, "C": -1000.0}


def test_single_and_bulk_add_agree():
    """Adding CNECs one at a time gives the same PTDF frame and clearing as one bulk add."""
    n_bulk = _network(_domain(SYMMETRIC))
    n_single = _network(_domain(SYMMETRIC), one_at_a_time=True)
    pd.testing.assert_frame_equal(
        n_single.c.flow_based_domains.zonal_ptdf.sort_index(),
        n_bulk.c.flow_based_domains.zonal_ptdf.sort_index(),
    )
    n_bulk.optimize(log_to_console=False)
    n_single.optimize(log_to_console=False)
    assert _net_positions(n_single).to_dict() == _net_positions(n_bulk).to_dict()


def _static_np(domain: pd.DataFrame) -> pd.Series:
    """Net positions of a single-snapshot solve of a static domain."""
    n = _network(domain)
    n.optimize(log_to_console=False)
    return n.buses_t.net_position.iloc[0][ZONES].round(0)


def _dynamic_network(dA: pd.DataFrame, dB: pd.DataFrame) -> pypsa.Network:
    """Two-snapshot toy whose zonal PTDF is ``dA`` in hour 0 and ``dB`` in hour 1."""
    n = pypsa.Network()
    n.set_snapshots([0, 1])
    n.add("Bus", ZONES)
    n.add("Load", ZONES, bus=ZONES, p_set=LOADS)
    n.add("Generator", ZONES, bus=ZONES, p_nom=4000, marginal_cost=COST)
    ptdf = pd.concat({0: dA[ZONES], 1: dB[ZONES]}, names=["snapshot", "name"])
    n.add("FlowBasedDomain", dA.index, zonal_ptdf=ptdf, ram=dA["RAM"].values)
    return n


def test_time_varying_zonal_ptdf_frontend_and_da():
    """A time-varying PTDF is public pandas (MultiIndex) and internal xarray (+snapshot)."""
    dA = _domain(SYMMETRIC)
    dB = dA.copy()
    dB[ZONES] = dB[ZONES] * 2.0
    c = _dynamic_network(dA, dB).c.flow_based_domains
    assert isinstance(c.zonal_ptdf.index, pd.MultiIndex)
    assert list(c.zonal_ptdf.columns) == ZONES
    assert set(c.da.zonal_ptdf.dims) == {"snapshot", "name", "bus"}
    pd.testing.assert_frame_equal(
        c.zonal_ptdf.loc[0][ZONES], dA[ZONES], check_like=True, check_names=False
    )


def test_time_varying_zonal_ptdf_matches_per_snapshot_static():
    """Each hour clears exactly like a static domain built from that hour's PTDF."""
    dA = _domain(SYMMETRIC)
    dB = dA.copy()
    dB[ZONES] = dB[ZONES] * 2.0  # tighter half-spaces in hour 1
    refA, refB = _static_np(dA), _static_np(dB)
    assert not refA.equals(refB)  # the two hours really differ

    n = _dynamic_network(dA, dB)
    n.optimize(log_to_console=False)
    npos = n.buses_t.net_position[ZONES].round(0)
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
        m.c.flow_based_domains.zonal_ptdf,
        n.c.flow_based_domains.zonal_ptdf,
        check_index_type=False,  # cnec labels come back object vs StringDtype (PyPSA-wide)
    )
    m.optimize(log_to_console=False)  # the recovered domain still solves


def test_copy_preserves_time_varying_domain():
    """n.copy() carries the time-varying zonal PTDF frame."""
    dA = _domain(SYMMETRIC)
    dB = dA.copy()
    dB[ZONES] = dB[ZONES] * 2.0
    n = _dynamic_network(dA, dB)
    m = n.copy()
    pd.testing.assert_frame_equal(
        m.c.flow_based_domains.zonal_ptdf, n.c.flow_based_domains.zonal_ptdf
    )


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
    n.add("FlowBasedDomain", d0.index, zonal_ptdf=d0[ZONES], ram=ram)
    n.optimize(log_to_console=False, assign_all_duals=True)

    npos = n.buses_t.net_position[ZONES].round(0)
    assert npos.loc[0].equals(ref0)
    assert npos.loc[1].equals(ref1)
    # one dual per snapshot
    assert len(n.c.flow_based_domains.dynamic["mu_domain"]) == 2


def test_time_varying_ptdf_with_link_column_broadcasts():
    """A time-varying PTDF carrying an EvFB link column broadcasts and stays gen - load."""
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
    n.add("FlowBasedDomain", dA.index, zonal_ptdf=ptdf, ram=dA["RAM"].values)
    n.optimize(log_to_console=False)

    npos = n.buses_t.net_position[ZONES].round(0)
    # EvFB cut applied per hour, so the net position stays generation - load
    gen_minus_load = (n.generators_t.p[ZONES] - LOADS).round(0)
    pd.testing.assert_frame_equal(npos, gen_minus_load, check_names=False)
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

    ref_np = ref.buses_t.net_position.iloc[0][ZONES].round(0)
    ref_mu = ref.c.flow_based_domains.dynamic["mu_domain"].iloc[0].round(3)
    mu = n.c.flow_based_domains.dynamic["mu_domain"]
    for sns in n.snapshots:
        assert n.buses_t.net_position.loc[sns][ZONES].round(0).equals(ref_np)
        assert mu.loc[sns].round(3).equals(ref_mu)
