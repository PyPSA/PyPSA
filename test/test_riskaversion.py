import pytest
from numpy.testing import assert_array_almost_equal as almost_equal

import pypsa


def test_set_risk_preference_requires_scenarios():
    n = pypsa.Network()
    # No scenarios defined: should raise
    with pytest.raises(RuntimeError, match=r"set_scenarios\(\)"):
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
