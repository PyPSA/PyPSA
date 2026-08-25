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


def _network(domain: pd.DataFrame) -> pypsa.Network:
    n = pypsa.Network()
    n.add("Bus", ZONES)
    n.add("Load", ZONES, bus=ZONES, p_set=LOADS)
    n.add("Generator", ZONES, bus=ZONES, p_nom=4000, marginal_cost=COST)
    n.add("FlowBasedDomain", domain.index, ram=domain["RAM"].values)
    n.flow_based_domains[ZONES] = domain[ZONES]
    return n


def _net_positions(n: pypsa.Network) -> pd.Series:
    return (n.generators_t.p.iloc[0] - LOADS)[ZONES].round(0)


def test_symmetric_domain_reproduces_toy():
    """The clearing lands on the AB+ edge with the canonical net positions and prices."""
    n = _network(_domain({"AB+": 1000, "AB-": 1000, "BC+": 1500, "BC-": 1500, "AC+": 2000, "AC-": 2000}))
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
    n = _network(_domain({"AB+": 1000, "AB-": 1000, "BC+": 1500, "BC-": 1500, "AC+": 2000, "AC-": 2000}))
    n.optimize(log_to_console=False)
    assert not n.buses_t.marginal_price.empty
    # net positions sum to zero (global balance)
    assert _net_positions(n).sum() == pytest.approx(0.0)


def test_asymmetric_ram_shifts_the_optimum():
    """A tighter AB+ margin curbs A's export below the symmetric case."""
    n = _network(_domain({"AB+": 600, "AB-": 1000, "BC+": 1500, "BC-": 1500, "AC+": 2000, "AC-": 2000}))
    n.optimize(log_to_console=False)
    assert _net_positions(n)["A"] < 2000.0


def test_validation_rejects_cross_zone_electrical_link():
    """A link between two zone buses must be removed; the domain replaces it."""
    n = _network(_domain({"AB+": 1000, "AB-": 1000, "BC+": 1500, "BC-": 1500, "AC+": 2000, "AC-": 2000}))
    n.add("Link", "A-B", bus0="A", bus1="B", p_nom=1000)
    with pytest.raises(ValueError, match="cross-zone"):
        n.optimize(log_to_console=False)


def test_non_zone_link_is_allowed():
    """A link to a non-zone bus (e.g. a gas pipeline) does not trip validation."""
    n = _network(_domain({"AB+": 1000, "AB-": 1000, "BC+": 1500, "BC-": 1500, "AC+": 2000, "AC-": 2000}))
    n.add("Bus", "gas")
    n.add("Link", "A-gas", bus0="A", bus1="gas", p_nom=1000)
    n.optimize(log_to_console=False)
    assert _net_positions(n)["A"] == pytest.approx(2000.0)


def test_inactive_domain_is_ignored():
    """Deactivating all constraints leaves an unconstrained copper-plate clearing."""
    n = _network(_domain({"AB+": 1000, "AB-": 1000, "BC+": 1500, "BC-": 1500, "AC+": 2000, "AC-": 2000}))
    n.flow_based_domains["active"] = False
    n.optimize(log_to_console=False)
    # cheapest generator (A) serves all demand; no binding domain
    assert "FlowBasedDomain-domain" not in n.model.constraints
