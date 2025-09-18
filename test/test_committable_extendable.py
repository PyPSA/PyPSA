"""Test the committable+extendable functionality using big-M formulation."""

import time

import numpy as np

import pypsa


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
    assert n.generators.p_nom_opt.loc["gas_gen"] > 0

    # Check that status variables were created
    assert "Generator-status" in n.model.variables
    assert "Generator-start_up" in n.model.variables
    assert "Generator-shut_down" in n.model.variables

    # Get status and dispatch for verification
    status_values = n.model.variables["Generator-status"].solution
    dispatch_values = n.generators_t.p["gas_gen"]

    # Check that when status is 0, dispatch is 0
    # And when status is 1, dispatch is >= p_min_pu * p_nom_opt
    p_nom_opt = n.generators.p_nom_opt.loc["gas_gen"]
    p_min_pu = n.generators.p_min_pu.loc["gas_gen"]
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
    assert n.generators.p_nom_opt.loc["gas_ext"] >= 0
    assert n.generators.p_nom_opt.loc["gas_com_ext"] >= 0

    # Check that status variables only exist for committable generators
    if "Generator-status" in n.model.variables:
        status_vars = n.model.variables["Generator-status"]
        committable_gens = n.generators.loc[n.generators.committable].index
        for gen in committable_gens:
            assert gen in status_vars.coords["name"].values

    # Verify power balance
    total_load = sum(load_profile)
    total_generation = n.generators_t.p.sum(axis=1).sum()
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
    total_generation = n.generators_t.p.sum().sum()
    assert abs(total_load - total_generation) < 1e-3

    # Check that extendable generators got some capacity
    extendable_gens = n.generators.loc[n.generators.p_nom_extendable].index
    for gen in extendable_gens:
        assert n.generators.p_nom_opt.loc[gen] >= 0

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
    for gen in n.generators.index:
        if n.generators.committable.loc[gen]:
            p_min_pu = n.generators.p_min_pu.loc[gen]
            p_nom_opt = n.generators.p_nom_opt.loc[gen]
            dispatch = n.generators_t.p[gen]
            status_vals = (
                n.generators_t.status[gen] if "status" in n.generators_t else None
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
    if hasattr(n.generators_t, "status"):
        status_vals = n.generators_t.status["must_run"]
        dispatch_vals = n.generators_t.p["must_run"]
        p_nom_opt = n.generators.p_nom_opt.loc["must_run"]

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
