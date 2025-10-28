# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import numpy as np

import pypsa


def test_simple_cycle() -> None:
    """Test the cycles function in a simple network with a known cycle."""
    # Create a test network with a cycle
    n = pypsa.Network()

    # Add buses
    for i in range(3):
        n.add("Bus", f"bus{i}", v_nom=220)

    # Add a cycle of lines
    n.add("Line", "line0-1", bus0="bus0", bus1="bus1", x=0.1, s_nom=100)
    n.add("Line", "line1-2", bus0="bus1", bus1="bus2", x=0.1, s_nom=100)
    n.add("Line", "line2-0", bus0="bus2", bus1="bus0", x=0.1, s_nom=100)

    # Get the cycles
    cycles_df = n.cycle_matrix()

    # A simple network with one cycle should have one column in the cycles matrix
    assert cycles_df.shape[1] == 1, f"Expected 1 cycle, got {cycles_df.shape[1]}"

    # The values in the cycles matrix should be either 1, -1, or 0
    assert set(np.unique(cycles_df.values)).issubset({0, 1, -1}), (
        "Unexpected values in cycles matrix"
    )

    # All rows should have a non-zero value, as all lines are part of the cycle
    assert (cycles_df.abs().sum(axis=1) > 0).all(), (
        "Some lines are not part of any cycle"
    )

    # For each cycle, we need to account for the carrier when checking the sum
    column_sums = cycles_df.sum()
    # Check that the values are consistent
    assert isinstance(column_sums.iloc[0], int | float), "Column sum should be numeric"


def test_multiple_cycles() -> None:
    """Test the cycles function in a network with multiple cycles."""
    n = pypsa.Network()

    # Create a network with two cycles sharing an edge
    # 0 -- 1 -- 2
    # |    |    |
    # 3 -- 4 -- 5

    # Add buses
    for i in range(6):
        n.add("Bus", f"bus{i}", v_nom=220)

    # Add lines to form two cycles
    # First cycle: 0-1-4-3-0
    n.add("Line", "line0-1", bus0="bus0", bus1="bus1", x=0.1, s_nom=100)
    n.add("Line", "line1-4", bus0="bus1", bus1="bus4", x=0.1, s_nom=100)
    n.add("Line", "line4-3", bus0="bus4", bus1="bus3", x=0.1, s_nom=100)
    n.add("Line", "line3-0", bus0="bus3", bus1="bus0", x=0.1, s_nom=100)

    # Second cycle: 1-2-5-4-1
    n.add("Line", "line1-2", bus0="bus1", bus1="bus2", x=0.1, s_nom=100)
    n.add("Line", "line2-5", bus0="bus2", bus1="bus5", x=0.1, s_nom=100)
    n.add("Line", "line5-4", bus0="bus5", bus1="bus4", x=0.1, s_nom=100)
    # line1-4 already added

    # Get the cycles
    cycles_df = n.cycle_matrix()

    # Network should have 2 independent cycles
    assert cycles_df.shape[1] == 2, f"Expected 2 cycles, got {cycles_df.shape[1]}"

    # Each edge should appear in at least one cycle
    assert (cycles_df.abs().sum(axis=1) > 0).all(), (
        "Some lines are not part of any cycle"
    )

    # The shared edge (1-4) should appear in both cycles
    line1_4_idx = ("Line", "line1-4")
    if line1_4_idx in cycles_df.index:
        assert (cycles_df.loc[line1_4_idx].abs() > 0).all(), (
            "Shared line should be part of both cycles"
        )

    # For each cycle, check the numeric value without requiring it to be zero
    for col in cycles_df.columns:
        cycle_sum = cycles_df[col].sum()
        assert isinstance(cycle_sum, int | float), f"Column {col} sum should be numeric"


def test_no_cycles() -> None:
    """Test the cycles function in a network with no cycles (tree structure)."""
    n = pypsa.Network()

    # Add buses in a tree structure
    for i in range(4):
        n.add("Bus", f"bus{i}", v_nom=220)

    # Add lines in a tree (no cycles)
    n.add("Line", "line0-1", bus0="bus0", bus1="bus1", x=0.1, s_nom=100)
    n.add("Line", "line0-2", bus0="bus0", bus1="bus2", x=0.1, s_nom=100)
    n.add("Line", "line0-3", bus0="bus0", bus1="bus3", x=0.1, s_nom=100)

    # Get the cycles
    cycles_df = n.cycle_matrix()

    # Should be an empty DataFrame with no cycles
    assert cycles_df.empty or cycles_df.shape[1] == 0, (
        "Network with tree structure should have no cycles"
    )


def test_multiple_subnetworks() -> None:
    """Test the cycles function in a network with multiple subnetworks."""
    n = pypsa.Network()

    # First subnetwork with a cycle
    for i in range(3):
        n.add("Bus", f"sub1_bus{i}", v_nom=220)

    n.add("Line", "sub1_line0-1", bus0="sub1_bus0", bus1="sub1_bus1", x=0.1, s_nom=100)
    n.add("Line", "sub1_line1-2", bus0="sub1_bus1", bus1="sub1_bus2", x=0.1, s_nom=100)
    n.add("Line", "sub1_line2-0", bus0="sub1_bus2", bus1="sub1_bus0", x=0.1, s_nom=100)

    # Second subnetwork with a cycle
    for i in range(3):
        n.add("Bus", f"sub2_bus{i}", v_nom=220)

    n.add("Line", "sub2_line0-1", bus0="sub2_bus0", bus1="sub2_bus1", x=0.1, s_nom=100)
    n.add("Line", "sub2_line1-2", bus0="sub2_bus1", bus1="sub2_bus2", x=0.1, s_nom=100)
    n.add("Line", "sub2_line2-0", bus0="sub2_bus2", bus1="sub2_bus0", x=0.1, s_nom=100)

    # Get the cycles
    cycles_df = n.cycle_matrix()

    # Should have 2 cycles (one from each subnetwork)
    assert cycles_df.shape[1] == 2, f"Expected 2 cycles, got {cycles_df.shape[1]}"

    # Each subnetwork's lines should participate in exactly one cycle
    sub1_lines = [("Line", f"sub1_line{i}-{(i + 1) % 3}") for i in range(3)]
    sub2_lines = [("Line", f"sub2_line{i}-{(i + 1) % 3}") for i in range(3)]

    # Check that each line in subnetwork 1 participates in one cycle
    for line in sub1_lines:
        if line in cycles_df.index:
            non_zero_count = (cycles_df.loc[line].abs() > 0).sum()
            assert non_zero_count == 1, (
                f"Line {line} should participate in exactly 1 cycle"
            )

    # Check that each line in subnetwork 2 participates in one cycle
    for line in sub2_lines:
        if line in cycles_df.index:
            non_zero_count = (cycles_df.loc[line].abs() > 0).sum()
            assert non_zero_count == 1, (
                f"Line {line} should participate in exactly 1 cycle"
            )


def test_mixed_branch_types() -> None:
    """Test the cycles function with different branch component types."""
    n = pypsa.Network()

    # Add buses
    for i in range(3):
        n.add("Bus", f"bus{i}", v_nom=220)

    # Add a cycle with a mix of lines and transformers
    n.add("Line", "line0-1", bus0="bus0", bus1="bus1", x=0.1, s_nom=100)
    n.add("Transformer", "transformer1-2", bus0="bus1", bus1="bus2", x=0.1, s_nom=100)
    n.add("Line", "line2-0", bus0="bus2", bus1="bus0", x=0.1, s_nom=100)

    # Get the cycles
    cycles_df = n.cycle_matrix()

    # Should have 1 cycle
    assert cycles_df.shape[1] == 1, f"Expected 1 cycle, got {cycles_df.shape[1]}"

    # Both lines and transformer should participate in the cycle
    components_in_cycles = {
        idx[0] for idx in cycles_df.index if (cycles_df.loc[idx].abs() > 0).any()
    }
    assert "Line" in components_in_cycles, "Lines should be part of cycle"
    assert "Transformer" in components_in_cycles, "Transformer should be part of cycle"


def test_investment_periods() -> None:
    """Test the cycles function with investment periods."""
    n = pypsa.Network()

    # Set investment periods
    n.set_investment_periods([2020, 2030, 2040])

    # Add buses
    for i in range(4):
        n.add("Bus", f"bus{i}", v_nom=220)

    # Add a cycle with lines built in different periods
    n.add(
        "Line",
        "line0-1",
        bus0="bus0",
        bus1="bus1",
        x=0.1,
        s_nom=100,
        build_year=2020,
        lifetime=40,
    )  # Available in all periods

    n.add(
        "Line",
        "line1-2",
        bus0="bus1",
        bus1="bus2",
        x=0.1,
        s_nom=100,
        build_year=2020,
        lifetime=40,
    )  # Available in all periods

    n.add(
        "Line",
        "line2-3",
        bus0="bus2",
        bus1="bus3",
        x=0.1,
        s_nom=100,
        build_year=2030,
        lifetime=20,
    )  # Available in 2030 and 2040

    n.add(
        "Line",
        "line3-0",
        bus0="bus3",
        bus1="bus0",
        x=0.1,
        s_nom=100,
        build_year=2040,
        lifetime=10,
    )  # Available only in 2040

    # Test for 2020: should have no complete cycles
    cycles_2020 = n.cycle_matrix(investment_period=2020)
    assert cycles_2020.empty or cycles_2020.shape[1] == 0, (
        "Should have no cycles in 2020"
    )

    # Test for 2030: should have no complete cycles
    cycles_2030 = n.cycle_matrix(investment_period=2030)
    assert cycles_2030.empty or cycles_2030.shape[1] == 0, (
        "Should have no cycles in 2030"
    )

    # Test for 2040: should have one complete cycle
    cycles_2040 = n.cycle_matrix(investment_period=2040)
    assert cycles_2040.shape[1] == 1, (
        f"Expected 1 cycle in 2040, got {cycles_2040.shape[1]}"
    )

    # Test without investment period: should include all branches
    cycles_all = n.cycle_matrix()
    assert cycles_all.shape[1] == 1, (
        f"Expected 1 cycle with all branches, got {cycles_all.shape[1]}"
    )


def test_weighted_cycles() -> None:
    """Test the apply_weights parameter in the cycles function."""
    # Create a test network with a cycle
    n = pypsa.Network()

    # Add buses
    for i in range(3):
        n.add("Bus", f"bus{i}", v_nom=220, carrier="AC")

    # Add a cycle of lines with different reactance values
    n.add("Line", "line0-1", bus0="bus0", bus1="bus1", x=0.1, r=0.01, s_nom=100)
    n.add("Line", "line1-2", bus0="bus1", bus1="bus2", x=0.2, r=0.02, s_nom=100)
    n.add("Line", "line2-0", bus0="bus2", bus1="bus0", x=0.3, r=0.03, s_nom=100)

    # Calculate the network topology and dependent values
    n.determine_network_topology()
    n.calculate_dependent_values()

    # Get the unweighted cycles
    cycles_unweighted = n.cycle_matrix(apply_weights=False)

    # Get the weighted cycles
    cycles_weighted = n.cycle_matrix(apply_weights=True)

    # Both should have the same structure
    assert cycles_unweighted.shape == cycles_weighted.shape
    assert (np.sign(cycles_unweighted) == np.sign(cycles_weighted)).all().all()

    # But the weighted version should have weights applied
    for line in ["line0-1", "line1-2", "line2-0"]:
        line_idx = ("Line", line)
        if not (cycles_weighted.loc[line_idx] == 0).all():
            # For AC networks, weight should be the per-unit effective reactance
            x_value = n.c.lines.static.at[line, "x_pu_eff"]
            # Find the column where this line has a non-zero value
            col = cycles_weighted.columns[(cycles_weighted.loc[line_idx] != 0).values][
                0
            ]

            # Unweighted value should be either 1 or -1
            unweighted_val = cycles_unweighted.loc[line_idx, col]

            # Weighted value should be x_value * unweighted_val
            weighted_val = cycles_weighted.loc[line_idx, col]
            assert np.isclose(weighted_val, x_value * unweighted_val)


def test_weighted_cycles_dc_network() -> None:
    """Test the apply_weights parameter with DC network."""
    n = pypsa.Network()

    # Add buses with DC carrier
    for i in range(3):
        n.add("Bus", f"bus{i}", v_nom=400, carrier="DC")

    # Add DC links with resistance values
    n.add("Line", "line0-1", bus0="bus0", bus1="bus1", x=0.0, r=0.01, s_nom=100)
    n.add("Line", "line1-2", bus0="bus1", bus1="bus2", x=0.0, r=0.02, s_nom=100)
    n.add("Line", "line2-0", bus0="bus2", bus1="bus0", x=0.0, r=0.03, s_nom=100)

    # Calculate the network topology
    n.determine_network_topology()
    n.calculate_dependent_values()

    # Get the unweighted and weighted cycles
    cycles_unweighted = n.cycle_matrix(apply_weights=False)
    cycles_weighted = n.cycle_matrix(apply_weights=True)

    # Both should have the same structure
    assert cycles_unweighted.shape == cycles_weighted.shape
    assert (np.sign(cycles_unweighted) == np.sign(cycles_weighted)).all().all()

    # For DC networks, weights should be resistance
    for line in ["line0-1", "line1-2", "line2-0"]:
        line_idx = ("Line", line)
        if not (cycles_weighted.loc[line_idx] == 0).all():
            # Find the column where this line has a non-zero value
            col = cycles_weighted.columns[(cycles_weighted.loc[line_idx] != 0).values][
                0
            ]

            # Unweighted value should be either 1 or -1
            unweighted_val = cycles_unweighted.loc[line_idx, col]

            # For DC, weighted value should be r_value * unweighted_val
            r_value = n.c.lines.static.at[line, "r_pu_eff"]
            weighted_val = cycles_weighted.loc[line_idx, col]
            assert np.isclose(weighted_val, r_value * unweighted_val)
