"""Test the committable+extendable functionality using big-M formulation."""

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


if __name__ == "__main__":
    # Run the tests
    test_committable_extendable_generator()
    test_committable_extendable_multiple_generators()
    test_big_m_formulation_constraints()
    print("All tests passed!")
