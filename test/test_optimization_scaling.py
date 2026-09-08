# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for numerical scaling in n.optimize(scaling=...)."""

from unittest.mock import patch

import linopy
import numpy as np
import pytest

NETWORKS = ["ac_dc_network", "storage_hvdc_network"]


def _solve_fails_on(model):
    """Patch `linopy.Model.solve` to raise for `model` only, the tuner keeps working."""
    orig = linopy.Model.solve

    def solve(self, *args, **kwargs):
        if self is model:
            raise RuntimeError("boom")
        return orig(self, *args, **kwargs)

    return patch("linopy.Model.solve", solve)


def _solve(n, **kw):
    n.optimize(**kw)
    return n


@pytest.mark.parametrize("network", NETWORKS)
@pytest.mark.parametrize(
    "scaling",
    [
        True,
        {"energy": 100, "cost": 1e3},
        {"energy": 100, "cost": 1e3, "constraint_factors": {"Bus-nodal_balance": 1e-2}},
    ],
    ids=["auto", "manual", "manual_rows"],
)
def test_scaling_equivalence(request, network, scaling):
    """Scaled and unscaled solves must agree in original units."""
    n = request.getfixturevalue(network)
    ref = _solve(n.copy(), scaling=False)
    got = _solve(n, scaling=scaling)

    np.testing.assert_allclose(got.objective, ref.objective, rtol=1e-5)
    np.testing.assert_allclose(
        got.objective_constant, ref.objective_constant, rtol=1e-5
    )

    for c, attr in [
        ("Generator", "p_nom_opt"),
        ("Link", "p_nom_opt"),
        ("Line", "s_nom_opt"),
    ]:
        a = got.components[c].static.get(attr)
        b = ref.components[c].static.get(attr)
        if a is not None and len(a):
            np.testing.assert_allclose(a.values, b.values, rtol=1e-5, atol=1e-6)

    for c, attr in [
        ("Generator", "p"),
        ("Link", "p"),
        ("Line", "s"),
        ("StorageUnit", "p"),
    ]:
        a = got.components[c].dynamic.get(attr)
        b = ref.components[c].dynamic.get(attr)
        if a is not None and a.shape[1]:
            if c == "StorageUnit":
                # individual dispatch is degenerate across identical units
                np.testing.assert_allclose(
                    a.values.sum(axis=1), b.values.sum(axis=1), rtol=1e-5, atol=1e-4
                )
            else:
                np.testing.assert_allclose(a.values, b.values, rtol=1e-5, atol=1e-6)

    np.testing.assert_allclose(
        got.c.buses.dynamic.marginal_price.values,
        ref.c.buses.dynamic.marginal_price.values,
        rtol=1e-5,
        atol=1e-6,
    )

    # mu_upper dual on a passive branch
    a = got.components["Line"].dynamic.get("mu_upper")
    b = ref.components["Line"].dynamic.get("mu_upper")
    if a is not None and a.shape[1]:
        np.testing.assert_allclose(a.values, b.values, rtol=1e-5, atol=1e-6)

    # global constraint mu
    gc_a, gc_b = got.c.global_constraints.static, ref.c.global_constraints.static
    if len(gc_a) and "mu" in gc_a:
        np.testing.assert_allclose(
            gc_a["mu"].values, gc_b["mu"].values, rtol=1e-5, atol=1e-6
        )


@pytest.mark.parametrize("network", NETWORKS)
def test_scaling_inputs_restored(request, network):
    """Network inputs are never touched by model scaling."""
    n = request.getfixturevalue(network)
    before = n.c.generators.static["capital_cost"].copy()
    before_mc = n.c.generators.dynamic["marginal_cost"].copy()
    n.optimize(scaling=True)
    np.testing.assert_array_equal(
        n.c.generators.static["capital_cost"].values, before.values
    )
    if before_mc.shape[1]:
        np.testing.assert_array_equal(
            n.c.generators.dynamic["marginal_cost"].values, before_mc.values
        )


@pytest.mark.parametrize("network", NETWORKS)
def test_scaling_exception_safety(request, network):
    """A solve failure leaves inputs and model data untouched, factors set."""
    n = request.getfixturevalue(network)
    before = n.c.generators.static["capital_cost"].copy()
    n.optimize.create_model(scaling=True)
    coeffs = {k: c.data["coeffs"].copy() for k, c in n.model.constraints.items()}
    with (
        _solve_fails_on(n.model),
        pytest.raises(RuntimeError),
    ):
        n.optimize.solve_model()
    np.testing.assert_array_equal(
        n.c.generators.static["capital_cost"].values, before.values
    )
    for k, c in n.model.constraints.items():
        assert c.data["coeffs"].equals(coeffs[k])
    assert n._scaling_factors is not None
    assert any(float(v.scaling.max()) != 1 for _, v in n.model.variables.items())


def test_scaling_unit_commitment():
    """Commitment binaries stay unscaled, their costs still round-trip."""
    import pypsa

    def build():
        n = pypsa.Network()
        n.set_snapshots(range(6))
        n.add("Bus", "b")
        n.add("Load", "l", bus="b", p_set=[50, 200, 60, 210, 40, 180])
        n.add("Generator", "base", bus="b", p_nom=100, marginal_cost=20)
        n.add(
            "Generator",
            "peak",
            bus="b",
            p_nom=300,
            marginal_cost=80,
            committable=True,
            p_min_pu=0.3,
            start_up_cost=5000,
            shut_down_cost=2000,
            stand_by_cost=100,
        )
        return n

    ref = _solve(build(), scaling=False)
    got = _solve(build(), scaling=True)
    np.testing.assert_allclose(got.objective, ref.objective, rtol=1e-6)
    np.testing.assert_array_equal(
        got.components["Generator"].dynamic["status"].values,
        ref.components["Generator"].dynamic["status"].values,
    )


def test_scaling_modular():
    """Integer module counts stay unscaled next to scaled capacities."""
    import pypsa

    def build():
        n = pypsa.Network()
        n.set_snapshots(range(4))
        n.add("Bus", "b")
        n.add("Load", "l", bus="b", p_set=[400, 600, 800, 500])
        n.add(
            "Generator",
            "modular_gas",
            bus="b",
            p_nom_extendable=True,
            committable=True,
            p_nom_mod=200,
            p_nom_max=1000,
            p_min_pu=0.3,
            marginal_cost=50,
            capital_cost=50000,
            start_up_cost=100,
            shut_down_cost=50,
        )
        return n

    ref = _solve(build(), scaling=False)
    got = _solve(build(), scaling=True)
    np.testing.assert_allclose(got.objective, ref.objective, rtol=1e-5)
    np.testing.assert_allclose(
        got.c["Generator"].static["p_nom_opt"].values,
        ref.c["Generator"].static["p_nom_opt"].values,
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        got.model.variables["Generator-n_mod"].solution.values,
        ref.model.variables["Generator-n_mod"].solution.values,
    )


def test_scaling_emissions():
    """A primary_energy CO2 cap: objective, shadow price (cost/emissions) and the
    binding-cap dispatch must match the unscaled solve."""
    import pypsa

    def build():
        n = pypsa.Network()
        n.set_snapshots(range(4))
        n.add("Bus", "b")
        n.add("Carrier", "gas", co2_emissions=0.2)
        n.add("Carrier", "clean", co2_emissions=0.0)
        n.add("Load", "l", bus="b", p_set=[100, 120, 90, 110])
        n.add("Generator", "gas", bus="b", carrier="gas", p_nom=200, marginal_cost=20)
        n.add(
            "Generator", "clean", bus="b", carrier="clean", p_nom=200, marginal_cost=80
        )
        # Cap forces some clean dispatch (gas-only would emit ~84 tCO2).
        n.add("GlobalConstraint", "co2", type="primary_energy", constant=40.0)
        return n

    ref = _solve(build(), scaling=False)
    got = _solve(build(), scaling=True)

    np.testing.assert_allclose(got.objective, ref.objective, rtol=1e-6)
    np.testing.assert_allclose(
        got.c.global_constraints.static["mu"].values,
        ref.c.global_constraints.static["mu"].values,
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        got.c.generators.dynamic.p.values,
        ref.c.generators.dynamic.p.values,
        rtol=1e-5,
        atol=1e-6,
    )


def test_scaling_storage():
    """A storage network (StorageUnit + Store, with >1 snapshot weighting and
    standing losses) exercises max_hours and standing_loss. SoC, store e, bus
    prices and the store energy-balance dual must round-trip."""
    import pypsa

    def build():
        n = pypsa.Network()
        n.set_snapshots(range(6))
        n.snapshot_weightings.loc[:, :] = 3.0  # >1 hour weighting
        n.add("Bus", "b")
        n.add("Load", "l", bus="b", p_set=[100, 300, 150, 350, 80, 250])
        n.add("Generator", "g", bus="b", p_nom=250, marginal_cost=50)
        n.add("Generator", "peak", bus="b", p_nom=300, marginal_cost=200)
        n.add(
            "StorageUnit",
            "su",
            bus="b",
            p_nom=120,
            max_hours=8,
            marginal_cost=2,
            standing_loss=0.01,  # nonlinear in the (rescaled) stores weighting
            state_of_charge_initial=200,
        )
        n.add("Bus", "e_bus", carrier="energy")
        n.add("Link", "chg", bus0="b", bus1="e_bus", p_nom=150, efficiency=0.9)
        n.add(
            "Store",
            "st",
            bus="e_bus",
            e_nom=2000,
            marginal_cost=1,
            standing_loss=0.02,
            e_initial=1000,
        )
        return n

    ref = _solve(build(), scaling=False, assign_all_duals=True)
    got = _solve(build(), scaling={"energy": 1e3, "cost": 1e6}, assign_all_duals=True)

    np.testing.assert_allclose(got.objective, ref.objective, rtol=1e-5)
    np.testing.assert_allclose(
        got.c.storage_units.dynamic.state_of_charge.values,
        ref.c.storage_units.dynamic.state_of_charge.values,
        rtol=1e-5,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        got.c.stores.dynamic.e.values,
        ref.c.stores.dynamic.e.values,
        rtol=1e-5,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        got.c.buses.dynamic.marginal_price.values,
        ref.c.buses.dynamic.marginal_price.values,
        rtol=1e-5,
        atol=1e-5,
    )
    # Store energy-balance dual verifies the cost/energy factor. (The StorageUnit
    # SoC-balance dual is skipped, it is degenerate whenever the unit sits at
    # SoC=0, so scaled/unscaled land on equivalent alternate dual vertices.)
    np.testing.assert_allclose(
        got.c.stores.dynamic.mu_energy_balance.values,
        ref.c.stores.dynamic.mu_energy_balance.values,
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("network", NETWORKS)
def test_scaling_two_step_matches_one_shot(request, network):
    """`create_model(scaling=...) + solve_model()` must match `optimize(scaling=...)`."""
    one = request.getfixturevalue(network).copy()
    one.optimize(scaling=True, assign_all_duals=True)

    two = request.getfixturevalue(network).copy()
    before = two.c.generators.static["capital_cost"].copy()
    two.optimize.create_model(scaling=True)
    # Inputs restored once the model is built (context exited).
    np.testing.assert_array_equal(
        two.c.generators.static["capital_cost"].values, before.values
    )
    two.optimize.solve_model(assign_all_duals=True)
    # And still restored after solving.
    np.testing.assert_array_equal(
        two.c.generators.static["capital_cost"].values, before.values
    )

    np.testing.assert_allclose(two.objective, one.objective, rtol=1e-5)
    np.testing.assert_allclose(
        two.objective_constant, one.objective_constant, rtol=1e-5
    )

    for c, attr in [
        ("Generator", "p_nom_opt"),  # extendable
        ("Link", "p_nom_opt"),  # fixed-nominal in these fixtures
        ("Line", "s_nom_opt"),
    ]:
        a = two.components[c].static.get(attr)
        b = one.components[c].static.get(attr)
        if a is not None and len(a):
            np.testing.assert_allclose(a.values, b.values, rtol=1e-5, atol=1e-6)

    for c, attr in [("Generator", "p"), ("Link", "p"), ("Line", "s")]:
        a = two.components[c].dynamic.get(attr)
        b = one.components[c].dynamic.get(attr)
        if a is not None and a.shape[1]:
            np.testing.assert_allclose(a.values, b.values, rtol=1e-5, atol=1e-6)

    np.testing.assert_allclose(
        two.c.buses.dynamic.marginal_price.values,
        one.c.buses.dynamic.marginal_price.values,
        rtol=1e-5,
        atol=1e-6,
    )


def test_scaling_two_step_fixed_nominal():
    """A non-extendable component gets the right p_nom_opt via the two-step path."""
    import pypsa

    def build():
        n = pypsa.Network()
        n.set_snapshots(range(3))
        n.add("Bus", "b")
        n.add("Load", "l", bus="b", p_set=[100, 150, 120])
        n.add("Generator", "fixed", bus="b", p_nom=200, marginal_cost=30)
        n.add(
            "Generator",
            "ext",
            bus="b",
            p_nom_extendable=True,
            capital_cost=1e5,
            marginal_cost=10,
        )
        return n

    one = build()
    one.optimize(scaling=True)
    two = build()
    two.optimize.create_model(scaling=True)
    two.optimize.solve_model()

    np.testing.assert_allclose(
        two.c.generators.static["p_nom_opt"].values,
        one.c.generators.static["p_nom_opt"].values,
        rtol=1e-5,
        atol=1e-6,
    )
    # Fixed generator keeps its nominal capacity exactly.
    assert two.c.generators.static.at["fixed", "p_nom_opt"] == 200


def _build_transformer_network(variable=False):
    import pypsa

    n = pypsa.Network()
    n.set_snapshots([0, 1])
    n.add("Carrier", "AC")
    n.add("Bus", "A", v_nom=1.0, carrier="AC")
    n.add("Bus", "B", v_nom=1.0, carrier="AC")
    n.add("Generator", "gen_A", bus="A", p_nom=100, marginal_cost=10.0, carrier="AC")
    n.add("Load", "load_B", bus="B", p_set=[50.0, 50.0])
    n.add("Line", "L1", bus0="A", bus1="B", x=0.01, r=1e-6, s_nom=100, carrier="AC")
    bounds = {"phase_shift_min": -20.0, "phase_shift_max": 20.0} if variable else {}
    n.add(
        "Transformer",
        "T1",
        bus0="A",
        bus1="B",
        x=1.0,  # x_pu = x / s_nom = 0.01
        r=1e-6,
        s_nom=100,
        phase_shift=10.0,
        **bounds,
    )
    return n


@pytest.mark.parametrize("variable", [False, True], ids=["fixed", "variable"])
def test_scaling_transformer_phase_shift(variable):
    """Transformer KVL terms and phase_shift readback must survive scaling.

    The phase-shift angle is a dimensionless column mixed into energy rows."""
    ref = _solve(_build_transformer_network(variable), scaling=False)
    got = _solve(_build_transformer_network(variable), scaling=True)

    np.testing.assert_allclose(got.objective, ref.objective, rtol=1e-6)
    for c in ("lines", "transformers"):
        np.testing.assert_allclose(
            got.c[c].dynamic["p0"].values,
            ref.c[c].dynamic["p0"].values,
            rtol=1e-5,
            atol=1e-6,
        )
    if variable:
        np.testing.assert_allclose(
            got.c.transformers.dynamic["phase_shift_opt"].values,
            ref.c.transformers.dynamic["phase_shift_opt"].values,
            rtol=1e-5,
            atol=1e-6,
        )
    # Persisted per-unit impedances must be the true values, not scaled ones.
    np.testing.assert_allclose(
        got.c.transformers.static["x_pu"].values,
        ref.c.transformers.static["x_pu"].values,
        rtol=1e-12,
    )


def test_scaling_overnight_cost():
    """overnight_cost and fom_cost enter the objective and must scale like
    capital_cost."""
    import pypsa

    def build():
        n = pypsa.Network()
        n.set_snapshots(range(3))
        n.add("Bus", "b")
        n.add("Load", "l", bus="b", p_set=[100, 150, 120])
        n.add("Generator", "fixed", bus="b", p_nom=100, marginal_cost=30)
        n.add(
            "Generator",
            "ext",
            bus="b",
            p_nom_extendable=True,
            overnight_cost=1e6,
            discount_rate=0.07,
            lifetime=25,
            fom_cost=2e4,
            marginal_cost=10,
        )
        return n

    ref = _solve(build(), scaling=False)
    got = _solve(build(), scaling=True)
    np.testing.assert_allclose(got.objective, ref.objective, rtol=1e-6)
    np.testing.assert_allclose(
        got.c.generators.static["p_nom_opt"].values,
        ref.c.generators.static["p_nom_opt"].values,
        rtol=1e-5,
        atol=1e-6,
    )


def test_scaling_piecewise_marginal_cost():
    """Piecewise chords mix column classes in one row and must round-trip."""
    import pandas as pd

    import pypsa

    def build():
        costs = pd.DataFrame(
            {"p_pu": [0.0, 0.5, 1.0], "marginal_cost": [0.0, 20.0, 40.0]}
        )
        n = pypsa.Network()
        n.add("Bus", "b")
        n.add("Generator", "gen", bus="b", p_nom=100, marginal_cost=costs)
        n.add("Load", "l", bus="b", p_set=50)
        return n

    ref = _solve(build(), scaling=False)
    got = _solve(build(), scaling=True)
    np.testing.assert_allclose(got.objective, ref.objective, rtol=1e-6)
    np.testing.assert_allclose(
        got.c.generators.dynamic.p.values, ref.c.generators.dynamic.p.values, rtol=1e-6
    )


def test_scaling_quadratic_objective_skipped(caplog):
    """A quadratic objective is left unscaled with a warning."""
    import pypsa

    n = pypsa.Network()
    n.set_snapshots(range(2))
    n.add("Bus", "b")
    n.add("Load", "l", bus="b", p_set=[100, 150])
    n.add(
        "Generator",
        "g",
        bus="b",
        p_nom=300,
        marginal_cost=10,
        marginal_cost_quadratic=0.1,
    )
    with caplog.at_level("WARNING", logger="pypsa.optimization.scaling"):
        n.optimize(scaling=True)
    assert "quadratic objective" in caplog.text
    assert n._scaling_factors is None


def test_scaling_frozen_constraints_skipped(ac_dc_network, caplog):
    """Frozen (CSR) constraints cannot be scaled in place, so scaling steps aside."""
    n = ac_dc_network
    ref = n.copy()
    ref.optimize(scaling=False)
    with caplog.at_level("WARNING", logger="pypsa.optimization.scaling"):
        n.optimize(scaling=True, model_kwargs={"freeze_constraints": True})
    assert "frozen constraints" in caplog.text
    assert n._scaling_factors is None
    np.testing.assert_allclose(n.objective, ref.objective, rtol=1e-6)


def test_scaling_restores_infinite_rhs_rows():
    """Rows the solver-side sanitizer would relabel still restore bit-exactly."""
    import pypsa

    def build():
        n = pypsa.Network()
        n.set_snapshots(range(2))
        n.add("Bus", "b")
        n.add("Load", "l", bus="b", p_set=[100, 150])
        n.add("Generator", "g", bus="b", p_nom=np.inf, marginal_cost=10)
        return n

    ref = build()
    ref.optimize.create_model()
    ref.model.constraints.sanitize_zeros()
    ref.model.constraints.sanitize_infinities()
    n = build()
    n.optimize(scaling={"energy": 8, "cost": 16})
    for k, c in n.model.constraints.items():
        assert c.data["coeffs"].equals(ref.model.constraints[k].data["coeffs"])


def test_scaling_keeps_stale_objective_value(ac_dc_network):
    """A failing re-solve must leave the previous objective value untouched."""
    n = ac_dc_network
    n.optimize(scaling={"energy": 8, "cost": 16})
    value = n.model.objective.value
    with (
        _solve_fails_on(n.model),
        pytest.raises(RuntimeError),
    ):
        n.optimize.solve_model()
    assert n.model.objective.value == value


def _factors_for(m, energy=2.0**9, cost=2.0**17):
    from pypsa.optimization.scaling import ScalingFactors

    return ScalingFactors(
        energy,
        cost,
        {name: 2.0 ** ((i % 5) - 2) for i, name in enumerate(m.constraints)},
    )


def _model_snapshot(m):
    return {
        "coeffs": {k: c.data["coeffs"].copy() for k, c in m.constraints.items()},
        "rhs": {k: c.data["rhs"].copy() for k, c in m.constraints.items()},
        "lower": {k: v.data["lower"].copy() for k, v in m.variables.items()},
        "upper": {k: v.data["upper"].copy() for k, v in m.variables.items()},
        "objective": m.objective.expression.data["coeffs"].copy(),
    }


def _assert_snapshot_equal(m, snap):
    for k, c in m.constraints.items():
        assert c.data["coeffs"].equals(snap["coeffs"][k])
        assert c.data["rhs"].equals(snap["rhs"][k])
    for k, v in m.variables.items():
        assert v.data["lower"].equals(snap["lower"][k])
        assert v.data["upper"].equals(snap["upper"][k])
    assert m.objective.expression.data["coeffs"].equals(snap["objective"])


def test_apply_factors_sets_factors_leaves_data(ac_dc_network):
    """User-side data stays untouched, linopy factors follow the spec."""
    from pypsa.optimization.scaling import apply_factors, classify_columns

    n = ac_dc_network
    n.optimize.create_model(include_objective_constant=False)
    m = n.model
    snap = _model_snapshot(m)
    factors = _factors_for(m)
    apply_factors(m, factors)
    _assert_snapshot_equal(m, snap)
    classes = classify_columns(m)
    col = {"energy": 1 / factors.energy, "cost": 1 / factors.cost, "none": 1.0}
    for name, var in m.variables.items():
        assert (
            float(var.scaling.min()) == float(var.scaling.max()) == col[classes[name]]
        )
    for name, con in m.constraints.items():
        assert float(con.scaling.max()) == factors.constraint_factors[name]
    assert m.objective.scaling == 1 / factors.cost


def test_apply_factors_solve_matches_plain(ac_dc_network):
    from pypsa.optimization.scaling import apply_factors

    n = ac_dc_network
    # a zero p_nom_opt makes the bound duals degenerate, floor it
    n.c.generators.static["p_nom_min"] = 10.0
    ref = n.copy()
    ref.optimize.create_model(include_objective_constant=False)
    ref.model.solve()

    n.optimize.create_model(include_objective_constant=False)
    m = n.model
    apply_factors(m, _factors_for(m))
    m.solve()

    np.testing.assert_allclose(m.objective.value, ref.model.objective.value, rtol=1e-6)
    for name, var in m.variables.items():
        np.testing.assert_allclose(
            var.solution.values,
            ref.model.variables[name].solution.values,
            rtol=1e-6,
            atol=1e-6,
        )
    for name, con in m.constraints.items():
        np.testing.assert_allclose(
            con.dual.values,
            ref.model.constraints[name].dual.values,
            rtol=1e-6,
            atol=1e-6,
        )


def test_apply_factors_ones_untouched(ac_dc_network):
    from pypsa.optimization.scaling import ScalingFactors, apply_factors

    n = ac_dc_network
    n.optimize.create_model(include_objective_constant=False)
    m = n.model
    before = {k: c.data["coeffs"].values for k, c in m.constraints.items()}
    apply_factors(m, ScalingFactors(1.0, 1.0, dict.fromkeys(m.constraints, 1.0)))
    for k, c in m.constraints.items():
        assert np.shares_memory(c.data["coeffs"].values, before[k])
        assert float(c.scaling.max()) == 1.0
    assert m.objective.scaling == 1.0


def test_scaling_factors_property(ac_dc_network):
    n = ac_dc_network
    assert n._scaling_factors is None
    n.optimize(scaling=True)
    factors = n._scaling_factors
    assert set(factors) == {"energy", "cost", "constraint_factors"}
    assert set(factors["constraint_factors"]) == set(n.model.constraints)
    values = [
        factors["energy"],
        factors["cost"],
        *factors["constraint_factors"].values(),
    ]
    assert all(float(np.log2(v)).is_integer() for v in values)
    n.optimize(scaling=False)
    assert n._scaling_factors is None


def test_scaling_objective_constant(ac_dc_network):
    """Scaling leaves the objective-constant default alone and round-trips it."""
    import pypsa

    n = ac_dc_network
    ref, got = n.copy(), n.copy()
    pypsa.options.params.optimize.include_objective_constant = None
    with pytest.warns(FutureWarning, match="include_objective_constant"):
        n.optimize(scaling=True)
    assert "objective_constant" in n.model.variables

    ref.optimize(scaling=False, include_objective_constant=True)
    got.optimize(scaling=True, include_objective_constant=True)
    assert "objective_constant" in got.model.variables
    np.testing.assert_allclose(got.objective, ref.objective, rtol=1e-6)
    np.testing.assert_allclose(got.objective_constant, ref.objective_constant)


def test_scaling_extra_functionality(ac_dc_network):
    """Constraints added in extra_functionality can be scaled."""

    def tiny(n, sns):
        n.model.add_constraints(1e-7 * n.model["Generator-p"] >= 0, name="tiny")

    n = ac_dc_network
    n.optimize(scaling={"constraint_factors": {"tiny": 1e6}}, extra_functionality=tiny)
    assert n._scaling_factors["constraint_factors"]["tiny"] == 1e6
    assert float(n.model.constraints["tiny"].scaling.max()) == 1e6


def test_scaling_unknown_constraint_group(ac_dc_network):
    n = ac_dc_network
    with pytest.raises(ValueError, match="unknown constraint group"):
        n.optimize(scaling={"constraint_factors": {"nope": 2.0}})


def test_tune_scaling_round_trip(ac_dc_network):
    """tune_scaling() returns what scaling=True applies, in dict form."""
    n = ac_dc_network
    n.optimize.create_model(include_objective_constant=False)
    tuned = n.optimize.tune_scaling()
    assert set(tuned) == {"energy", "cost", "constraint_factors"}
    assert set(tuned["constraint_factors"]) == set(n.model.constraints)
    n.optimize(scaling=tuned, include_objective_constant=False)
    manual = n._scaling_factors
    n.optimize(scaling=True, include_objective_constant=False)
    assert manual == n._scaling_factors
    rows = n.optimize.tune_scaling(constraint_factors=True)["constraint_factors"]
    assert all(float(np.log2(v)).is_integer() for v in rows.values())


def test_scaling_mga(ac_dc_network):
    """MGA with unequal energy/cost factors matches the unscaled MGA optimum."""
    ref = ac_dc_network.copy()
    ref.optimize()
    ref.optimize.optimize_mga(slack=0.05)

    got = ac_dc_network.copy()
    scaling = {"energy": 100, "cost": 1e6}
    got.optimize(scaling=scaling)
    got.optimize.optimize_mga(slack=0.05, scaling=scaling)

    # The MGA objective (total generator capacity) is unique even when the
    # individual capacities are degenerate.
    np.testing.assert_allclose(
        got.c.generators.static["p_nom_opt"].sum(),
        ref.c.generators.static["p_nom_opt"].sum(),
        rtol=1e-5,
    )


# --- resolver, column classifier, exponent chooser ----------------------------


def test_resolve_scaling():
    from pypsa.optimization.scaling import ScalingFactors, resolve_scaling

    assert resolve_scaling(False) is None
    assert resolve_scaling(None) is None
    assert resolve_scaling(True) is True
    assert resolve_scaling({"energy": 1000}) == ScalingFactors(1000.0, 1.0, {})
    assert resolve_scaling({"constraint_factors": {"a": 2}}) == ScalingFactors(
        1.0, 1.0, {"a": 2.0}
    )
    assert resolve_scaling({"cost": 65536.0}).cost == 65536.0
    assert resolve_scaling({"energy": np.int64(1000)}).energy == 1000.0
    for bad in (
        "big",
        {"energy": "x"},
        {"constraint_factors": True},
        {"constraint_factors": {"a": "x"}},
        {"emissions": 1},
        {"columns": True},
    ):
        with pytest.raises(TypeError):
            resolve_scaling(bad)
    for bad in (
        {"energy": 0},
        {"energy": float("inf")},
        {"cost": float("nan")},
        {"constraint_factors": {"a": 0}},
    ):
        with pytest.raises(ValueError):
            resolve_scaling(bad)
    with pytest.raises(TypeError, match="power"):
        resolve_scaling({"power": 2})


def _build_uc_modular_network():
    import pypsa

    n = pypsa.Network()
    n.set_snapshots(range(4))
    n.add("Bus", "b")
    n.add("Load", "l", bus="b", p_set=[400, 600, 800, 500])
    n.add(
        "Generator",
        "modular_gas",
        bus="b",
        p_nom_extendable=True,
        committable=True,
        p_nom_mod=200,
        p_nom_max=1000,
        p_min_pu=0.3,
        marginal_cost=50,
        capital_cost=50000,
        start_up_cost=100,
        shut_down_cost=50,
    )
    return n


def test_classify_columns():
    from pypsa.optimization.scaling import classify_columns

    n = _build_uc_modular_network()
    n.optimize.create_model()
    classes = classify_columns(n.model)
    assert classes["Generator-status"] == "none"
    assert classes["Generator-n_mod"] == "none"
    assert classes["Generator-p"] == "energy"
    assert classes["Generator-p_nom"] == "energy"


def test_classify_columns_cvar(stochastic_network):
    from pypsa.optimization.scaling import classify_columns

    n = stochastic_network
    n.set_risk_preference(alpha=0.2, omega=0.5)
    n.optimize.create_model()
    classes = classify_columns(n.model)
    assert classes["CVaR-a"] == "cost"
    assert classes["CVaR"] == "cost"


def _window_violation(m, exps):
    """Sum of log2 window violations of every ILP quantity under `exps`."""
    from pypsa.optimization.scaling import (
        WINDOW,
        _group_ranges,
        _quantities,
        classify_columns,
    )

    classes = classify_columns(m)
    logv, A, cats = _quantities(m, classes, _group_ranges(m, classes))
    g = np.log2([exps.energy, exps.cost, *exps.constraint_factors.values()])
    scaled = logv + A.to_numpy() @ g
    lo = np.log2([WINDOW[c][0] for c in cats])
    hi = np.log2([WINDOW[c][1] for c in cats])
    return float(np.maximum(lo - scaled, 0).sum() + np.maximum(scaled - hi, 0).sum())


def test_choose_factors(ac_dc_network):
    from pypsa.optimization.scaling import (
        ScalingFactors,
        _group_ranges,
        _row_units,
        choose_factors,
        classify_columns,
    )

    n = ac_dc_network
    n.optimize.create_model(include_objective_constant=False)
    m = n.model

    tied = choose_factors(m, constraint_factors=False)
    assert set(tied.constraint_factors) == set(m.constraints)
    units = _row_units(_group_ranges(m, classify_columns(m)), list(m.constraints))
    follow = {"energy": 1 / tied.energy, "cost": 1 / tied.cost, "none": 1.0}
    assert all(
        tied.constraint_factors[name] == follow[u]
        for name, u in zip(m.constraints, units, strict=True)
    )

    auto = choose_factors(m, constraint_factors=True)
    assert auto == choose_factors(m, constraint_factors=True)
    assert isinstance(auto, ScalingFactors)
    assert all(
        float(np.log2(v)).is_integer()
        for v in (auto.energy, auto.cost, *auto.constraint_factors.values())
    )
    zero = ScalingFactors(1.0, 1.0, dict.fromkeys(m.constraints, 1.0))
    viol_auto, viol_zero = _window_violation(m, auto), _window_violation(m, zero)
    assert viol_auto == 0 or viol_auto < viol_zero


def test_resolve_factors_indicator(caplog):
    import pypsa
    from pypsa.optimization.scaling import resolve_factors

    n = pypsa.Network()
    n.set_snapshots(range(2))
    n.add("Bus", "b")
    n.add("Load", "l", bus="b", p_set=[50, 80])
    n.add("Generator", "g", bus="b", p_nom=100, committable=True, marginal_cost=10)
    n.optimize.create_model()
    m = n.model
    status = m.variables["Generator-status"]
    p = m.variables["Generator-p"]
    m.add_indicator_constraints(status, 1, 1 * p, ">=", 0, name="ind")
    with caplog.at_level("WARNING"):
        factors = resolve_factors(m, True)
    assert factors is None
    assert "indicator" in caplog.text
    with pytest.raises(ValueError, match="indicator"):
        n.optimize.tune_scaling()


def test_scaling_units_keep_coefficients(ac_dc_network):
    """Unlisted rows follow their unit.

    Row factor 1/512 against column factor 1/512 cancels in linopy's export
    `Sc A Sx^-1`, so the solver sees the coefficients as built.
    """
    from pypsa.optimization.scaling import (
        ScalingFactors,
        apply_factors,
        resolve_factors,
    )

    n = ac_dc_network
    n.optimize.create_model(include_objective_constant=False)
    m = n.model
    factors = resolve_factors(m, ScalingFactors(512.0, 3.0, {}))
    assert factors.constraint_factors["Bus-nodal_balance"] == 1 / 512
    apply_factors(m, factors)
    assert float(m.constraints["Bus-nodal_balance"].scaling.max()) == 1 / 512
    assert float(m.variables["Generator-p"].scaling.max()) == 1 / 512
