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


def _add_domain(n: pypsa.Network, domain: pd.DataFrame, one_at_a_time: bool) -> None:
    """Attach the domain either as one bulk add or one CNEC at a time."""
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


def _network(domain: pd.DataFrame, one_at_a_time: bool = False) -> pypsa.Network:
    n = pypsa.Network()
    n.add("Bus", ZONES)
    n.add("Load", ZONES, bus=ZONES, p_set=LOADS)
    n.add("Generator", ZONES, bus=ZONES, p_nom=4000, marginal_cost=COST)
    _add_domain(n, domain, one_at_a_time)
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
    assert "net_position" not in dynamic  # net position lives in n.buses_t.p
    assert "mu_balance" not in dynamic  # scalar zero-sum dual has no component slot


def test_validation_rejects_cross_zone_electrical_link():
    """A link between two zone buses must be removed; the domain replaces it."""
    n = _network(_domain(SYMMETRIC))
    n.add("Link", "A-B", bus0="A", bus1="B", p_nom=1000)
    with pytest.raises(ValueError, match="cross-zone"):
        n.optimize(log_to_console=False)


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
