"""Test the committable+extendable functionality using big-M formulation."""

import time

import numpy as np
import pytest

import pypsa
from pypsa.optimization.constraints import _infer_big_m_scale


def test_committable_extendable_modular_generator():
    """Test a generator that is committable, extendable, AND modular.

    This tests compatibility between:
    - Unit commitment (binary status variables)
    - Capacity expansion (continuous p_nom variable)
    - Modular expansion (integer n_mod variable, p_nom = n_mod * p_nom_mod)
    """
    n = pypsa.Network()
    n.set_snapshots(range(4))  # 4 hours for simplicity

    n.add("Carrier", "gas")
    n.add("Bus", "bus", carrier="AC")

    # Simple load pattern
    load_profile = [400, 600, 800, 500]
    n.add("Load", "load", bus="bus", p_set=load_profile)

    # Add a committable + extendable + modular generator
    # Module size: 200 MW, max 5 modules = 1000 MW
    n.add(
        "Generator",
        "modular_gas",
        bus="bus",
        carrier="gas",
        p_nom_extendable=True,
        committable=True,
        p_nom_mod=200,  # Modular: must build in 200 MW increments
        p_nom_max=1000,
        p_min_pu=0.3,  # 30% minimum stable generation when online
        marginal_cost=50,
        capital_cost=50000,
        start_up_cost=100,
        shut_down_cost=50,
    )

    # Solve
    status, termination_code = n.optimize(solver_name="highs")

    assert status == "ok", f"Optimization failed with status: {status}"

    # Verify capacity is a multiple of p_nom_mod
    p_nom_opt = n.c["Generator"].static.loc["modular_gas", "p_nom_opt"]
    p_nom_mod = n.c["Generator"].static.loc["modular_gas", "p_nom_mod"]

    assert p_nom_opt > 0, "Should have built some capacity"

    # Check that p_nom_opt is a multiple of p_nom_mod (within numerical tolerance)
    n_modules = p_nom_opt / p_nom_mod
    assert abs(n_modules - round(n_modules)) < 1e-3, (
        f"Capacity {p_nom_opt} MW should be a multiple of module size {p_nom_mod} MW"
    )

    # Verify the n_mod variable exists and has the correct value
    assert "Generator-n_mod" in n.model.variables, "Modular variable should exist"
    n_mod_solution = n.model.variables["Generator-n_mod"].solution.loc["modular_gas"]
    assert n_mod_solution == round(n_modules), (
        f"n_mod solution {n_mod_solution} should match {round(n_modules)}"
    )

    # Verify unit commitment constraints are respected
    status_values = n.model.variables["Generator-status"].solution.sel(
        name="modular_gas"
    )
    dispatch_values = n.c["Generator"].dynamic["p"]["modular_gas"]
    p_min_pu = n.c["Generator"].static.loc["modular_gas", "p_min_pu"]
    min_power = p_min_pu * p_nom_opt

    for t in range(4):
        if status_values.isel(snapshot=t).item() < 0.5:  # Offline
            assert dispatch_values.iloc[t] < 1e-6, f"At t={t}, offline but dispatching"
        else:  # Online
            assert dispatch_values.iloc[t] >= min_power - 1e-3, (
                f"At t={t}, online but below minimum power"
            )

    # Verify power balance
    total_load = sum(load_profile)
    total_generation = dispatch_values.sum()
    assert abs(total_load - total_generation) < 1e-3, "Power balance violated"


def test_committable_extendable_modular_link():
    """Test a link that is committable, extendable, AND modular."""
    n = pypsa.Network()
    n.set_snapshots(range(3))

    n.add("Bus", "bus0", carrier="AC")
    n.add("Bus", "bus1", carrier="AC")

    # Load on bus1
    n.add("Load", "load", bus="bus1", p_set=[300, 500, 400])

    # Cheap generator on bus0
    n.add(
        "Generator",
        "gen0",
        bus="bus0",
        p_nom=1000,
        marginal_cost=10,
    )

    # Committable + extendable + modular link
    n.add(
        "Link",
        "modular_link",
        bus0="bus0",
        bus1="bus1",
        p_nom_extendable=True,
        committable=True,
        p_nom_mod=150,  # 150 MW modules
        p_nom_max=600,
        p_min_pu=0.2,
        marginal_cost=5,
        capital_cost=30000,
        start_up_cost=50,
    )

    status, termination_code = n.optimize(solver_name="highs")

    assert status == "ok"

    # Verify modular capacity
    p_nom_opt = n.c["Link"].static.loc["modular_link", "p_nom_opt"]
    p_nom_mod = n.c["Link"].static.loc["modular_link", "p_nom_mod"]
    n_modules = p_nom_opt / p_nom_mod

    assert p_nom_opt > 0
    assert abs(n_modules - round(n_modules)) < 1e-3

    # Verify n_mod variable
    assert "Link-n_mod" in n.model.variables
    n_mod_solution = n.model.variables["Link-n_mod"].solution.loc["modular_link"]
    assert n_mod_solution == round(n_modules)


def test_committable_extendable_generator():
    """Test a generator that is both committable and extendable."""
    # Create a simple network
    n = pypsa.Network()
    n.set_snapshots(range(24))  # 24 hours

    n.add("Carrier", "gas")
    n.add("Bus", "bus", carrier="el")

    # Create load pattern
    load_profile = [
        2000,
        1800,
        1600,
        1400,
        1300,
        1400,
        1600,
        1800,
        2200,
        2400,
        2600,
        2800,
        2900,
        2800,
        2700,
        2600,
        2400,
        2600,
        2800,
        3000,
        2800,
        2600,
        2400,
        2200,
    ]
    n.add("Load", "load", bus="bus", p_set=load_profile)

    # Add a generator that is both committable and extendable
    n.add(
        "Generator",
        "gas_gen",
        bus="bus",
        p_nom_extendable=True,
        committable=True,
        carrier="gas",
        marginal_cost=50,
        capital_cost=100000,
        p_nom_max=3000,
        p_min_pu=0.4,  # Minimum load when running
        start_up_cost=1000,
        shut_down_cost=500,
    )

    # Solve the optimization
    status, termination_code = n.optimize(solver_name="highs")

    # Check that optimization was successful
    assert status == "ok"

    # Check that the generator was built with some capacity
    assert n.c["Generator"].static.loc["gas_gen", "p_nom_opt"] > 0

    # Check that status variables were created
    assert "Generator-status" in n.model.variables
    assert "Generator-start_up" in n.model.variables
    assert "Generator-shut_down" in n.model.variables

    # Get status and dispatch for verification
    status_values = n.model.variables["Generator-status"].solution
    dispatch_values = n.c["Generator"].dynamic["p"]["gas_gen"]

    # Check that when status is 0, dispatch is 0
    # And when status is 1, dispatch is >= p_min_pu * p_nom_opt
    p_nom_opt = n.c["Generator"].static.loc["gas_gen", "p_nom_opt"]
    p_min_pu = n.c["Generator"].static.loc["gas_gen", "p_min_pu"]
    min_power = p_min_pu * p_nom_opt

    for t in range(24):
        if status_values.loc[t, "gas_gen"] == 0:
            # When offline, dispatch should be 0
            assert abs(dispatch_values.iloc[t]) < 1e-6
        else:
            # When online, dispatch should be >= minimum power
            assert dispatch_values.iloc[t] >= min_power - 1e-6

    # Verify power balance (generation = load)
    total_load = sum(load_profile)
    total_generation = dispatch_values.sum()
    assert abs(total_load - total_generation) < 1e-3


def test_committable_extendable_multiple_generators():
    """Test multiple generators with different combinations of committable/extendable."""
    n = pypsa.Network()
    n.set_snapshots(range(8))  # 8 hours for faster test

    n.add("Carrier", ["coal", "gas"])
    n.add("Bus", "bus", carrier="el")

    # Simple load profile
    load_profile = [1000, 800, 600, 700, 900, 1100, 1200, 1000]
    n.add("Load", "load", bus="bus", p_set=load_profile)

    # Generator 1: committable only (not extendable)
    n.add(
        "Generator",
        "coal_fixed",
        bus="bus",
        p_nom=500,
        committable=True,
        carrier="coal",
        marginal_cost=30,
        p_min_pu=0.5,
        start_up_cost=500,
    )

    # Generator 2: extendable only (not committable)
    n.add(
        "Generator",
        "gas_ext",
        bus="bus",
        p_nom_extendable=True,
        carrier="gas",
        marginal_cost=60,
        capital_cost=50000,
        p_nom_max=1000,
    )

    # Generator 3: both committable and extendable
    n.add(
        "Generator",
        "gas_com_ext",
        bus="bus",
        p_nom_extendable=True,
        committable=True,
        carrier="gas",
        marginal_cost=40,
        capital_cost=80000,
        p_nom_max=800,
        p_min_pu=0.3,
        start_up_cost=800,
    )

    # Solve the optimization
    status, termination_code = n.optimize(solver_name="highs")

    # Check that optimization was successful
    assert status == "ok"

    # Check that extendable generators got some capacity
    assert n.c["Generator"].static.loc["gas_ext", "p_nom_opt"] >= 0
    assert n.c["Generator"].static.loc["gas_com_ext", "p_nom_opt"] >= 0

    # Check that status variables only exist for committable generators
    if "Generator-status" in n.model.variables:
        status_vars = n.model.variables["Generator-status"]
        committable_gens = (
            n.c["Generator"].static.loc[n.c["Generator"].static["committable"]].index
        )
        for gen in committable_gens:
            assert gen in status_vars.coords["name"].values

    # Verify power balance
    total_load = sum(load_profile)
    total_generation = n.c["Generator"].dynamic["p"].sum(axis=1).sum()
    assert abs(total_load - total_generation) < 1e-3


def test_big_m_formulation_constraints():
    """Test that big-M constraints are correctly applied."""
    n = pypsa.Network()
    n.set_snapshots(range(3))  # Just 3 snapshots for constraint testing

    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=[100, 200, 150])

    # Add committable+extendable generator with specific parameters
    n.add(
        "Generator",
        "test_gen",
        bus="bus",
        p_nom_extendable=True,
        committable=True,
        marginal_cost=50,
        capital_cost=100000,
        p_nom_max=1000,  # This will be used for big-M calculation
        p_min_pu=0.4,
        p_max_pu=0.8,
    )

    # Build the model without solving
    n.optimize.create_model()

    # Check that the right constraint names exist for committable+extendable
    constraint_names = list(n.model.constraints)

    # Should have vectorized big-M constraints for committable+extendable generators
    big_m_lower_constraints = [
        name for name in constraint_names if "Generator-com-ext-p-lower" in name
    ]
    big_m_upper_constraints = [
        name for name in constraint_names if "Generator-com-ext-p-upper-bigM" in name
    ]
    capacity_upper_constraints = [
        name for name in constraint_names if "Generator-com-ext-p-upper-cap" in name
    ]

    assert len(big_m_lower_constraints) > 0, (
        f"Big-M lower constraints should exist. Found constraints: {constraint_names}"
    )
    assert len(big_m_upper_constraints) > 0, (
        f"Big-M upper constraints should exist. Found constraints: {constraint_names}"
    )
    assert len(capacity_upper_constraints) > 0, (
        f"Capacity upper constraints should exist. Found constraints: {constraint_names}"
    )


def test_big_m_scale_infers_peak_load():
    """The auto big-M scale uses the network peak load when available."""
    n = pypsa.Network()
    n.set_snapshots(range(3))
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=[100, 250, 180])

    inferred_scale = _infer_big_m_scale(n, "Generator")

    assert inferred_scale == pytest.approx(2500)


def test_big_m_scale_fallback_without_load():
    """Auto big-M falls back to a conservative default when no load exists."""
    n = pypsa.Network()
    n.set_snapshots([0])
    n.add("Bus", "bus")

    inferred_scale = _infer_big_m_scale(n, "Generator")

    assert inferred_scale == pytest.approx(1e6)


def test_non_negative_constraint_added():
    """Ensure non-negative dispatch constraint is added when min_pu >= 0."""
    n = pypsa.Network()
    n.set_snapshots([0])

    n.add("Bus", "bus", carrier="el")
    n.add("Load", "load", bus="bus", p_set=50)

    n.add(
        "Generator",
        "uc_gen",
        bus="bus",
        p_nom_extendable=True,
        committable=True,
        p_nom_max=200,
        p_min_pu=0.0,
        p_max_pu=0.8,
        marginal_cost=10,
        capital_cost=1000,
    )

    n.optimize.create_model()

    constraint_names = list(n.model.constraints)
    assert any("Generator-com-ext-p-lower-nonneg" in name for name in constraint_names)


def test_non_negative_constraint_skipped_for_negative_min_pu():
    """Ensure non-negative constraint is skipped when min_pu is negative."""
    n = pypsa.Network()
    n.set_snapshots([0])

    n.add("Bus", "bus", carrier="el")
    n.add("Load", "load", bus="bus", p_set=50)

    n.add(
        "Generator",
        "uc_gen",
        bus="bus",
        p_nom_extendable=True,
        committable=True,
        p_nom_max=200,
        p_min_pu=-0.1,
        p_max_pu=0.8,
        marginal_cost=10,
        capital_cost=1000,
    )

    n.optimize.create_model()

    constraint_names = list(n.model.constraints)
    assert not any(
        "Generator-com-ext-p-lower-nonneg" in name for name in constraint_names
    )


def test_committable_extendable_can_switch_off():
    """Ensure committable+extendable units can be offline without infeasibility."""
    n = pypsa.Network()
    n.set_snapshots(range(2))

    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=[100, 0])

    n.add(
        "Generator",
        "uc_gen",
        bus="bus",
        p_nom_extendable=True,
        committable=True,
        marginal_cost=40,
        capital_cost=60000,
        p_nom_max=200,
        p_min_pu=0.4,
        start_up_cost=200,
        shut_down_cost=100,
    )

    status, termination_code = n.optimize(solver_name="highs")

    assert status == "ok"
    assert n.c["Generator"].static.loc["uc_gen", "p_nom_opt"] > 0

    status_values = n.c["Generator"].dynamic["status"]["uc_gen"]
    dispatch_values = n.c["Generator"].dynamic["p"]["uc_gen"]

    assert status_values.iloc[0] > 0.5
    assert dispatch_values.iloc[0] == pytest.approx(100, rel=1e-6, abs=1e-6)

    assert status_values.iloc[1] < 0.5
    assert abs(dispatch_values.iloc[1]) < 1e-6


def test_big_m_warning_emitted(caplog):
    """Trigger the big-M warning when optimized capacity exceeds the bound."""
    n = pypsa.Network()
    n.set_snapshots([0])

    n.add("Bus", "bus")
    n.add(
        "Generator",
        "uc_gen",
        bus="bus",
        p_nom_extendable=True,
        committable=True,
        p_nom_max=100,
    )

    n.c["Generator"].static.loc["uc_gen", "p_nom_opt"] = 60
    n.c["Generator"].static.loc["uc_gen", "p_max_pu"] = 0.2
    n.c["Generator"].dynamic["p_max_pu"]["uc_gen"] = 0.2

    caplog.set_level("WARNING", logger="pypsa.optimization.optimize")
    n.optimize._warn_big_m_exceeded()

    assert any("big-M bounds" in record.message for record in caplog.records), (
        "Expected big-M warning was not emitted"
    )


def test_many_generators_performance():
    """Test performance with many committable+extendable generators."""
    n = pypsa.Network()
    n.set_snapshots(range(24))  # 24 hours

    n.add("Bus", "bus")

    # Create a realistic load pattern
    base_load = 1000
    load_pattern = [
        base_load * (0.6 + 0.4 * np.sin(2 * np.pi * t / 24 + np.pi / 4))
        for t in range(24)
    ]
    n.add("Load", "load", bus="bus", p_set=load_pattern)

    # Add many generators with different properties
    n_generators = 20
    for i in range(n_generators):
        # Mix of generator types
        is_committable = i % 3 != 0  # 2/3 are committable
        is_extendable = i % 2 == 0  # 1/2 are extendable

        gen_name = f"gen_{i}"

        if is_committable and is_extendable:
            # Committable + extendable
            n.add(
                "Generator",
                gen_name,
                bus="bus",
                p_nom_extendable=True,
                committable=True,
                marginal_cost=30 + i * 5,
                capital_cost=50000 + i * 10000,
                p_nom_max=200 + i * 50,
                p_min_pu=0.2 + 0.02 * i,
                start_up_cost=500 + i * 100,
                shut_down_cost=250 + i * 50,
            )
        elif is_committable:
            # Committable only
            n.add(
                "Generator",
                gen_name,
                bus="bus",
                p_nom=150 + i * 25,
                committable=True,
                marginal_cost=30 + i * 5,
                p_min_pu=0.3 + 0.02 * i,
                start_up_cost=500 + i * 100,
            )
        elif is_extendable:
            # Extendable only
            n.add(
                "Generator",
                gen_name,
                bus="bus",
                p_nom_extendable=True,
                marginal_cost=40 + i * 5,
                capital_cost=60000 + i * 8000,
                p_nom_max=300 + i * 40,
            )
        else:
            # Fixed capacity
            n.add(
                "Generator",
                gen_name,
                bus="bus",
                p_nom=100 + i * 20,
                marginal_cost=35 + i * 5,
            )

    # Time the optimization
    start_time = time.time()
    status, termination_code = n.optimize(solver_name="highs")
    optimization_time = time.time() - start_time

    # Check optimization was successful
    assert status == "ok", f"Optimization failed with status: {status}"

    # Verify results make sense
    total_load = sum(load_pattern)
    total_generation = n.c["Generator"].dynamic["p"].sum().sum()
    assert abs(total_load - total_generation) < 1e-3

    # Check that extendable generators got some capacity
    extendable_gens = (
        n.c["Generator"].static.loc[n.c["Generator"].static["p_nom_extendable"]].index
    )
    for gen in extendable_gens:
        assert n.c["Generator"].static.loc[gen, "p_nom_opt"] >= 0

    # Performance check: should complete in reasonable time
    assert optimization_time < 30, (
        f"Optimization took too long: {optimization_time:.2f}s"
    )


def test_extreme_parameters():
    """Test with extreme parameter values."""
    n = pypsa.Network()
    n.set_snapshots(range(12))

    n.add("Bus", "bus")
    n.add(
        "Load",
        "load",
        bus="bus",
        p_set=[100, 200, 300, 400, 500, 600, 500, 400, 300, 200, 150, 100],
    )

    # Generator with very small minimum load
    n.add(
        "Generator",
        "tiny_min",
        bus="bus",
        p_nom_extendable=True,
        committable=True,
        marginal_cost=50,
        capital_cost=80000,
        p_nom_max=1000,
        p_min_pu=0.01,  # Very small
        start_up_cost=100,
    )

    # Generator with very high minimum load
    n.add(
        "Generator",
        "high_min",
        bus="bus",
        p_nom_extendable=True,
        committable=True,
        marginal_cost=60,
        capital_cost=90000,
        p_nom_max=500,
        p_min_pu=0.9,  # Very high
        start_up_cost=1000,
    )

    # Generator with very large capacity limit
    n.add(
        "Generator",
        "huge_cap",
        bus="bus",
        p_nom_extendable=True,
        committable=True,
        marginal_cost=40,
        capital_cost=70000,
        p_nom_max=1e6,  # Very large
        p_min_pu=0.2,
        start_up_cost=5000,
    )

    status, termination_code = n.optimize(solver_name="highs")
    assert status == "ok"

    # Check that all constraints are satisfied
    for gen in n.c["Generator"].static.index:
        if n.c["Generator"].static.loc[gen, "committable"]:
            p_min_pu = n.c["Generator"].static.loc[gen, "p_min_pu"]
            p_nom_opt = n.c["Generator"].static.loc[gen, "p_nom_opt"]
            dispatch = n.c["Generator"].dynamic["p"][gen]
            status_vals = (
                n.c["Generator"].dynamic["status"][gen]
                if gen in n.c["Generator"].dynamic["status"].columns
                else None
            )

            if status_vals is not None:
                # When online, dispatch should be >= p_min_pu * p_nom_opt
                for t in range(len(dispatch)):
                    if status_vals.iloc[t] > 0.5:  # Online
                        min_power = p_min_pu * p_nom_opt
                        assert dispatch.iloc[t] >= min_power - 1e-6, (
                            f"Generator {gen} violates minimum power at time {t}"
                        )


def test_zero_minimum_load():
    """Test with zero minimum load (p_min_pu = 0)."""
    n = pypsa.Network()
    n.set_snapshots(range(6))

    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=[50, 100, 80, 60, 40, 30])

    n.add(
        "Generator",
        "flexible",
        bus="bus",
        p_nom_extendable=True,
        committable=True,
        marginal_cost=50,
        capital_cost=80000,
        p_nom_max=200,
        p_min_pu=0.0,  # Zero minimum load
        start_up_cost=500,
    )

    status, termination_code = n.optimize(solver_name="highs")
    assert status == "ok"


def test_maximum_minimum_load():
    """Test with maximum minimum load (p_min_pu = 1.0)."""
    n = pypsa.Network()
    n.set_snapshots(range(4))

    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=[200, 200, 200, 200])

    n.add(
        "Generator",
        "must_run",
        bus="bus",
        p_nom_extendable=True,
        committable=True,
        marginal_cost=40,
        capital_cost=100000,
        p_nom_max=250,
        p_min_pu=1.0,  # Must run at full capacity
        start_up_cost=2000,
    )

    status, termination_code = n.optimize(solver_name="highs")
    assert status == "ok"

    # Verify that when generator is on, it runs at full capacity
    if "must_run" in n.c["Generator"].dynamic["status"].columns:
        status_vals = n.c["Generator"].dynamic["status"]["must_run"]
        dispatch_vals = n.c["Generator"].dynamic["p"]["must_run"]
        p_nom_opt = n.c["Generator"].static.loc["must_run", "p_nom_opt"]

        for t in range(len(status_vals)):
            if status_vals.iloc[t] > 0.5:  # Online
                expected_power = p_nom_opt
                assert abs(dispatch_vals.iloc[t] - expected_power) < 1e-6


def test_infinite_capacity_limit():
    """Test with infinite capacity limit."""
    n = pypsa.Network()
    n.set_snapshots(range(3))

    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=[500, 1000, 750])

    n.add(
        "Generator",
        "unlimited",
        bus="bus",
        p_nom_extendable=True,
        committable=True,
        marginal_cost=60,
        capital_cost=50000,
        p_nom_max=np.inf,  # Infinite capacity
        p_min_pu=0.2,
        start_up_cost=1000,
    )

    status, termination_code = n.optimize(solver_name="highs")
    assert status == "ok"


def test_single_snapshot():
    """Test with single snapshot."""
    n = pypsa.Network()
    n.set_snapshots([0])

    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=500)

    n.add(
        "Generator",
        "single_snapshot",
        bus="bus",
        p_nom_extendable=True,
        committable=True,
        marginal_cost=50,
        capital_cost=80000,
        p_nom_max=800,
        p_min_pu=0.3,
        start_up_cost=1000,
    )

    status, termination_code = n.optimize(solver_name="highs")
    assert status == "ok"


def test_no_start_costs():
    """Test with zero startup/shutdown costs."""
    n = pypsa.Network()
    n.set_snapshots(range(6))

    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=[100, 300, 200, 400, 150, 250])

    # Add a cheaper baseload generator to make the problem feasible
    n.add(
        "Generator",
        "baseload",
        bus="bus",
        p_nom_extendable=True,
        marginal_cost=30,
        capital_cost=60000,
        p_nom_max=300,
    )

    n.add(
        "Generator",
        "no_start_cost",
        bus="bus",
        p_nom_extendable=True,
        committable=True,
        marginal_cost=50,
        capital_cost=80000,
        p_nom_max=500,
        p_min_pu=0.2,  # Lower minimum to avoid infeasibility
        start_up_cost=0,  # Zero startup cost
        shut_down_cost=0,
    )  # Zero shutdown cost

    status, termination_code = n.optimize(solver_name="highs")
    assert status in ["ok", "warning"], f"Optimization failed with status: {status}"


if __name__ == "__main__":
    # Run the tests
    test_committable_extendable_generator()
    test_committable_extendable_multiple_generators()
    test_big_m_formulation_constraints()
    print("All tests passed!")
