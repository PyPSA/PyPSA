# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""
Test warnings for cyclic storage overriding initial values.
"""

import logging

import pandas as pd

import pypsa


def test_warning_storage_unit_global_cyclic(caplog):
    """Test warning when global cyclic_state_of_charge overrides initial value."""
    n = pypsa.Network(snapshots=range(4))
    n.add("Bus", "bus")
    n.add("Carrier", "carrier")
    n.add("Load", "load", bus="bus", p_set=0.1)
    n.add("Generator", "gen", bus="bus", carrier="carrier", p_nom=1, marginal_cost=10)

    # Add storage with both cyclic=True and non-zero initial value
    n.add(
        "StorageUnit",
        "storage",
        bus="bus",
        carrier="carrier",
        p_nom=1,
        max_hours=1,
        state_of_charge_initial=0.5,  # Non-zero initial value
        cyclic_state_of_charge=True,  # Cyclic enabled
        marginal_cost=1,
    )

    with caplog.at_level(logging.WARNING):
        n.optimize()

    # Check that warning was issued
    assert any(
        "Cyclic state of charge constraint overrules initial storage level setting"
        in record.message
        and "storage" in record.message
        for record in caplog.records
    ), "Expected warning about cyclic overriding initial values not found"


def test_warning_storage_unit_per_period_cyclic(caplog):
    """Test warning when per-period cyclic overrides initial value in multi-investment."""
    n = pypsa.Network()
    n.set_snapshots(
        pd.MultiIndex.from_tuples(
            [(2030, 0), (2030, 1), (2040, 0), (2040, 1)], names=["period", "timestep"]
        )
    )
    n.set_investment_periods([2030, 2040])
    n.add("Bus", "bus")
    n.add("Carrier", "carrier")
    n.add("Load", "load", bus="bus", p_set=0.1)
    n.add("Generator", "gen", bus="bus", carrier="carrier", p_nom=1, marginal_cost=10)

    # Add storage with both per-period cyclic=True and per-period initial=True
    n.add(
        "StorageUnit",
        "storage",
        bus="bus",
        carrier="carrier",
        p_nom=1,
        max_hours=1,
        state_of_charge_initial=0.5,  # Non-zero initial value
        cyclic_state_of_charge_per_period=True,  # Per-period cyclic
        state_of_charge_initial_per_period=True,  # Per-period initial
        marginal_cost=1,
    )

    with caplog.at_level(logging.WARNING):
        n.optimize(multi_investment_periods=True)

    # Check that warning was issued
    assert any(
        "Cyclic state of charge constraint overrules initial storage level setting"
        in record.message
        and "storage" in record.message
        for record in caplog.records
    ), "Expected warning about cyclic overriding initial values not found"


def test_warning_store_global_cyclic(caplog):
    """Test warning when global e_cyclic overrides initial value."""
    n = pypsa.Network(snapshots=range(4))
    n.add("Bus", "bus")
    n.add("Carrier", "carrier")
    n.add("Load", "load", bus="bus", p_set=0.1)
    n.add("Generator", "gen", bus="bus", carrier="carrier", p_nom=1, marginal_cost=10)

    # Add store with both cyclic=True and non-zero initial value
    n.add(
        "Store",
        "store",
        bus="bus",
        carrier="carrier",
        e_nom=1,
        e_initial=0.5,  # Non-zero initial value
        e_cyclic=True,  # Cyclic enabled
        marginal_cost=1,
    )

    with caplog.at_level(logging.WARNING):
        n.optimize()

    # Check that warning was issued
    assert any(
        "Cyclic energy level constraint overrules initial value setting"
        in record.message
        and "store" in record.message
        for record in caplog.records
    ), "Expected warning about cyclic overriding initial values not found"


def test_warning_store_per_period_cyclic(caplog):
    """Test warning when per-period cyclic overrides initial value in multi-investment."""
    n = pypsa.Network()
    n.set_snapshots(
        pd.MultiIndex.from_tuples(
            [(2030, 0), (2030, 1), (2040, 0), (2040, 1)], names=["period", "timestep"]
        )
    )
    n.set_investment_periods([2030, 2040])
    n.add("Bus", "bus")
    n.add("Carrier", "carrier")
    n.add("Load", "load", bus="bus", p_set=0.1)
    n.add("Generator", "gen", bus="bus", carrier="carrier", p_nom=1, marginal_cost=10)

    # Add store with both per-period cyclic=True and per-period initial=True
    n.add(
        "Store",
        "store",
        bus="bus",
        carrier="carrier",
        e_nom=1,
        e_initial=0.5,  # Non-zero initial value
        e_cyclic_per_period=True,  # Per-period cyclic
        e_initial_per_period=True,  # Per-period initial
        marginal_cost=1,
    )

    with caplog.at_level(logging.WARNING):
        n.optimize(multi_investment_periods=True)

    # Check that warning was issued
    assert any(
        "Cyclic energy level constraint overrules initial value setting"
        in record.message
        and "store" in record.message
        for record in caplog.records
    ), "Expected warning about cyclic overriding initial values not found"


def test_warning_storage_unit_cp_overrides_c(caplog):
    """Test warning when per-period cyclic overrides global cyclic for StorageUnit."""
    n = pypsa.Network()
    n.set_snapshots(
        pd.MultiIndex.from_tuples(
            [(2030, 0), (2030, 1), (2040, 0), (2040, 1)], names=["period", "timestep"]
        )
    )
    n.set_investment_periods([2030, 2040])
    n.add("Bus", "bus")
    n.add("Carrier", "carrier")
    n.add("Load", "load", bus="bus", p_set=0.1)
    n.add("Generator", "gen", bus="bus", carrier="carrier", p_nom=1, marginal_cost=10)

    # Add storage with both global cyclic AND per-period cyclic (CP overrides C)
    n.add(
        "StorageUnit",
        "storage",
        bus="bus",
        carrier="carrier",
        p_nom=1,
        max_hours=1,
        cyclic_state_of_charge=True,  # Global cyclic
        cyclic_state_of_charge_per_period=True,  # Per-period cyclic (overrides global)
        marginal_cost=1,
    )

    with caplog.at_level(logging.WARNING):
        n.optimize(multi_investment_periods=True)

    # Check that warning was issued
    assert any(
        "Per-period cyclic (cyclic_state_of_charge_per_period=True) overrides global cyclic"
        in record.message
        and "storage" in record.message
        for record in caplog.records
    ), "Expected warning about CP overriding C not found"


def test_warning_store_cp_overrides_c(caplog):
    """Test warning when per-period cyclic overrides global cyclic for Store."""
    n = pypsa.Network()
    n.set_snapshots(
        pd.MultiIndex.from_tuples(
            [(2030, 0), (2030, 1), (2040, 0), (2040, 1)], names=["period", "timestep"]
        )
    )
    n.set_investment_periods([2030, 2040])
    n.add("Bus", "bus")
    n.add("Carrier", "carrier")
    n.add("Load", "load", bus="bus", p_set=0.1)
    n.add("Generator", "gen", bus="bus", carrier="carrier", p_nom=1, marginal_cost=10)

    # Add store with both global cyclic AND per-period cyclic (CP overrides C)
    n.add(
        "Store",
        "store",
        bus="bus",
        carrier="carrier",
        e_nom=1,
        e_cyclic=True,  # Global cyclic
        e_cyclic_per_period=True,  # Per-period cyclic (overrides global)
        marginal_cost=1,
    )

    with caplog.at_level(logging.WARNING):
        n.optimize(multi_investment_periods=True)

    # Check that warning was issued
    assert any(
        "Per-period cyclic (e_cyclic_per_period=True) overrides global cyclic"
        in record.message
        and "store" in record.message
        for record in caplog.records
    ), "Expected warning about CP overriding C not found"
