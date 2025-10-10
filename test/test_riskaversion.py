# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import pytest
from numpy.testing import assert_array_almost_equal as almost_equal

import pypsa


def test_set_risk_preference_requires_scenarios():
    n = pypsa.Network()
    # No scenarios defined: should raise RuntimeError
    with pytest.raises(ValueError, match=r"set_scenarios\(\)"):
        n.set_risk_preference(alpha=0.1, omega=0.5)


def test_set_and_get_risk_preference_dict_and_flags():
    n = pypsa.Network()
    # Define scenarios
    n.set_scenarios(["s1", "s2"])  # equal weights 0.5/0.5

    # Initially, should not have risk preference
    assert n.has_risk_preference is False
    assert n.risk_preference is None

    # Set valid preferences
    n.set_risk_preference(alpha=0.2, omega=0.7)

    # Getter returns dict
    rp = n.risk_preference
    assert isinstance(rp, dict)
    assert pytest.approx(rp["alpha"]) == 0.2
    assert pytest.approx(rp["omega"]) == 0.7

    # Boolean flag flips to True
    assert n.has_risk_preference is True


@pytest.mark.parametrize(
    ("alpha", "omega", "err", "pattern"),
    [
        (-0.1, 0.5, ValueError, "Alpha must be between 0 and 1"),
        (1.0, 0.5, ValueError, "Alpha must be between 0 and 1"),
        (0.2, -0.01, ValueError, "Omega must be between 0 and 1"),
        (0.2, 1.01, ValueError, "Omega must be between 0 and 1"),
    ],
)
def test_set_risk_preference_validation(alpha, omega, err, pattern):
    n = pypsa.Network()
    n.set_scenarios({"a": 0.4, "b": 0.6})
    with pytest.raises(err, match=pattern):
        n.set_risk_preference(alpha=alpha, omega=omega)


@pytest.fixture
def toy_network():
    n = pypsa.examples.model_energy()
    n.add(
        "Generator",
        "gas",
        carrier="gas",
        bus="electricity",
        p_nom=5000,
        efficiency=0.5,
        marginal_cost=100,
    )
    # downsample to first week of July (7 days at 3-hourly resolution -> 56 snapshots)
    week_idx = n.snapshots[(n.snapshots.month == 7) & (n.snapshots.day <= 7)]
    n.snapshots = week_idx
    n.snapshot_weightings = n.snapshot_weightings * 8760 / len(week_idx)
    return n


def _capacities_mw(n: pypsa.Network):
    # A quick helper to return installed capacities as MW Series indexed by (component, carrier) for almost_equal check
    cap = n.statistics.optimal_capacity()
    if "scenario" in cap.index.names:
        without_scen = [lvl for lvl in cap.index.names if lvl != "scenario"]
        cap = cap.groupby(level=without_scen).first()
    cap = cap.groupby(level=["component", "carrier"]).sum()
    return cap.sort_index()


def _objective_bil(n: pypsa.Network) -> float:
    assert n.objective is not None
    return n.objective / 1e9


def test_risk_neutral_equivalence_omega_zero(toy_network):
    n = toy_network.copy()

    # Stochastic risk-neutral (omega=0)
    n.set_scenarios({"volcano": 0.1, "no_volcano": 0.9})
    # degrade solar in volcano
    n.generators_t.p_max_pu.loc[:, ("volcano", "solar")] *= 0.3

    # Risk neutral run
    n.optimize(log_to_console=False)
    cap_stoch_neutral = _capacities_mw(n)
    obj_stoch_neutral = _objective_bil(n)

    # CVaR with omega=0 should equal risk-neutral (same model/objective)
    n.set_risk_preference(alpha=0.2, omega=0.0)
    n.optimize(log_to_console=False)
    cap_cvar_w0 = _capacities_mw(n)
    obj_cvar_w0 = _objective_bil(n)

    almost_equal(cap_stoch_neutral.values, cap_cvar_w0.values)
    assert obj_stoch_neutral == pytest.approx(obj_cvar_w0, rel=1e-8, abs=1e-8)


def test_worst_case_equivalence_omega_one_single_tail(toy_network):
    n = toy_network.copy()

    # Deterministic: volcano-only as worst case baseline
    nd = toy_network.copy()
    # degrade solar in baseline w/o scenarios
    nd.generators_t.p_max_pu.loc[:, ("solar")] *= 0.3
    nd.optimize(log_to_console=False)
    cap_det_worst = _capacities_mw(nd)
    obj_det_worst = _objective_bil(nd)

    # Stochastic with two scenarios
    n.set_scenarios({"volcano": 0.1, "no_volcano": 0.9})
    n.generators_t.p_max_pu.loc[:, ("volcano", "solar")] *= 0.3

    # CVaR with omega=1 and alpha capturing the worst scenario only
    # For p_worst = 0.1, set alpha = 1 - p_worst = 0.9 so 1/(1-alpha)*p_worst = 1
    p_worst = n.scenario_weightings.loc["volcano", "weight"]
    n.set_risk_preference(alpha=float(1.0 - p_worst), omega=1.0)
    n.optimize(log_to_console=False)
    cap_cvar_w1 = _capacities_mw(n)
    obj_cvar_w1 = _objective_bil(n)

    almost_equal(cap_det_worst.values, cap_cvar_w1.values)
    assert obj_det_worst == pytest.approx(obj_cvar_w1, rel=1e-8, abs=1e-8)


def test_monotone_objective_vs_omega(toy_network):
    n = toy_network.copy()

    # Setup stochastic
    n.set_scenarios({"volcano": 0.1, "no_volcano": 0.9})
    n.generators_t.p_max_pu.loc[:, ("volcano", "solar")] *= 0.3

    # Risk-neutral baseline
    n.optimize(log_to_console=False)
    obj_neutral = _objective_bil(n)

    # Evaluate CVaR objective for increasing omega values
    alpha = 0.9
    objectives = []
    for omega in [0.25, 0.5, 0.75]:
        n.set_risk_preference(alpha=alpha, omega=omega)
        n.optimize(log_to_console=False)
        objectives.append(_objective_bil(n))

    obj_025, obj_050, obj_075 = objectives
    eps = 1e-8

    # Ensure strict monotonicity: obj_neutral < obj(0.25) < obj(0.5) < obj(0.75)
    assert obj_neutral < obj_025 - eps
    assert obj_025 < obj_050 - eps
    assert obj_050 < obj_075 - eps


def test_cvar_constraints_cover_many_settings():
    """Test CVaR optimization for multiple components and settings, incl. storage_units and stores, spillage, and unit commitment.

    This test builds a tiny stochastic network with:
    - Store with non-zero marginal_cost_storage (uses Store-e)
    - StorageUnit with non-zero spill_cost (uses StorageUnit-spill)
    - Committable Generator with stand_by_cost and start/shutdown costs

    Then enables CVaR and checks that CVaR constraints exist in the model.
    """
    import pandas as pd

    n = pypsa.Network(snapshots=pd.RangeIndex(0, 4))
    n.add("Carrier", ["elec", "gas"])  # just to be explicit
    n.add("Bus", "b", carrier="elec")

    # Load
    n.add("Load", "d", bus="b", p_set=[80, 120, 90, 110])

    # Generator with marginal and stand_by costs
    n.add(
        "Generator",
        "gen_quad",
        bus="b",
        p_nom=200,
        marginal_cost=5,
        carrier="elec",
        stand_by_cost=1.0,
    )

    # Store with marginal_cost_storage
    n.add(
        "Store",
        "store1",
        bus="b",
        e_nom=100,
        p_nom=100,
        marginal_cost_storage=1.5,  # currency/MWh/h
    )

    # StorageUnit with spill cost
    n.add(
        "StorageUnit",
        "su1",
        bus="b",
        p_nom=120,
        max_hours=2,
        spill_cost=3.0,
    )

    # Stochastic setup + CVaR
    n.set_scenarios({"s1": 0.5, "s2": 0.5})
    n.set_risk_preference(alpha=0.5, omega=0.2)

    # Small scenario-specific variation to avoid degenerate equivalence
    n.c.loads.dynamic.p_set.loc[:, ("s2", "d")] = [90, 130, 95, 115]

    status, cond = n.optimize(log_to_console=False, solver_name="highs")
    assert status == "ok"
    assert cond == "optimal"

    # CVaR constraints created correctly
    assert "CVaR-def" in n.model.constraints
    for s in n.scenarios:
        key = f"CVaR-excess-{s}"
        assert key in n.model.constraints


def test_cvar_constraints_multiperiod_opt():
    """Cover CVaR weighting path for multi-period investment optimization."""
    n = pypsa.Network(snapshots=range(3))
    n.investment_periods = [2020, 2030]

    n.add("Carrier", "elec")
    n.add("Bus", "b", carrier="elec")

    # Two extendable generators in different periods
    n.add(
        "Generator",
        "g20",
        bus="b",
        build_year=2020,
        lifetime=40,
        p_nom_extendable=True,
        capital_cost=50,
        marginal_cost=1,
    )
    n.add(
        "Generator",
        "g30",
        bus="b",
        build_year=2030,
        lifetime=40,
        p_nom_extendable=True,
        capital_cost=20,
        marginal_cost=5,
    )

    n.add("Load", "d", bus="b", p_set=[100, 120, 110, 90, 105, 95])

    # Scenarios + CVaR
    n.set_scenarios({"high": 0.1, "low": 0.9})
    n.set_risk_preference(alpha=0.9, omega=0.5)

    n.c.generators.static.loc[("low", "g20"), "marginal_cost"] = 1.0
    n.c.generators.static.loc[("high", "g20"), "marginal_cost"] = 100.0

    status, cond = n.optimize(multi_investment_periods=True, log_to_console=False)
    assert status == "ok"
    assert cond == "optimal"

    # CVaR constraints present
    assert "CVaR-def" in n.model.constraints
    for s in n.scenarios:
        assert f"CVaR-excess-{s}" in n.model.constraints


def test_cvar_with_quadratic_opex_raises():
    """Ensure we fail fast when CVaR is enabled together with quadratic marginal costs.

    The guard should raise a ValueError with a clear message before model solve
    when `marginal_cost_quadratic` terms are present (unsupported with CVaR).
    """
    import pandas as pd

    n = pypsa.Network(snapshots=pd.RangeIndex(0, 3))
    n.add("Carrier", "elec")
    n.add("Bus", "b", carrier="elec")
    n.add("Load", "d", bus="b", p_set=[50, 60, 55])

    # Generator with quadratic marginal cost
    n.add(
        "Generator",
        "gq",
        bus="b",
        p_nom=200,
        marginal_cost=0.0,
        marginal_cost_quadratic=0.01,  # triggers quadratic OPEX calc
    )

    # Stochastic setup + CVaR
    n.set_scenarios({"s1": 0.5, "s2": 0.5})
    n.set_risk_preference(alpha=0.5, omega=0.3)

    with pytest.raises(ValueError, match=r"CVaR with quadratic operational costs"):
        n.optimize(log_to_console=False)


def test_objective_without_any_costs_raises():
    """Cover optimize.py guard when neither CAPEX nor OPEX terms exist."""
    n = pypsa.Network(snapshots=range(2))
    n.add("Carrier", "elec")
    n.add("Bus", "b", carrier="elec")
    n.add("Load", "d", bus="b", p_set=[1.0, 1.0])
    n.add("Generator", "g", bus="b", p_nom=5.0)

    # No capex/opex terms — objective cannot be formed
    with pytest.raises(ValueError, match=r"Objective function could not be created"):
        n.optimize(log_to_console=False)


def test_objective_includes_standby_cost_for_committable():
    """Trigger more OPEX terms in aux constraints"""

    n = pypsa.Network(snapshots=range(2))
    n.add("Bus", "b", carrier="elec")
    # Remember that at t=0, we do not charge a start_up by default
    n.add("Load", "d", bus="b", p_set=[0, 50])

    # Committable generator with non-zero stand-by cost and start-up cost
    n.add(
        "Generator",
        "gc",
        bus="b",
        p_nom=200,
        marginal_cost=10.0,
        committable=True,
        p_min_pu=0.1,
        stand_by_cost=1.0,
        start_up_cost=5.0,
    )

    status, cond = n.optimize(log_to_console=False, solver_name="highs")
    assert status == "ok"
    assert cond == "optimal"
    assert n.objective == 506.0  # 10*50 + 5 + 1
