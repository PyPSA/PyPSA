"""Real-world scenario tests for committable+extendable functionality."""

import numpy as np

import pypsa


def test_power_system_expansion_planning():
    """Test a realistic power system expansion planning scenario."""
    n = pypsa.Network()

    # One year of representative days (52 weeks * 24 hours = ~1 week per month)
    n_days = 7
    hours_per_day = 24
    n.set_snapshots(range(n_days * hours_per_day))

    # Create multiple buses
    n.add("Bus", "north", x=0, y=1)
    n.add("Bus", "south", x=0, y=0)
    n.add("Bus", "east", x=1, y=0.5)

    # Add transmission lines
    n.add("Line", "north-south", bus0="north", bus1="south", x=0.1, r=0.01, s_nom=1000)
    n.add("Line", "south-east", bus0="south", bus1="east", x=0.08, r=0.01, s_nom=800)

    # Create realistic load patterns for different regions
    rng = np.random.default_rng(42)  # For reproducible results

    def create_load_pattern(base_load, peak_hour=18, seasonal_factor=1.0):
        pattern = []
        for day in range(n_days):
            for hour in range(hours_per_day):
                # Daily pattern: low at night, high in evening
                daily_factor = 0.6 + 0.4 * (1 + np.sin(2 * np.pi * (hour - 6) / 24)) / 2
                # Weekly pattern: higher on weekdays
                weekly_factor = 1.1 if day < 5 else 0.8
                # Some randomness
                random_factor = 1 + 0.1 * rng.normal()

                load = (
                    base_load
                    * daily_factor
                    * weekly_factor
                    * seasonal_factor
                    * random_factor
                )
                pattern.append(max(0, load))
        return pattern

    # Add loads to different buses
    n.add("Load", "load_north", bus="north", p_set=create_load_pattern(800))
    n.add("Load", "load_south", bus="south", p_set=create_load_pattern(1200))
    n.add("Load", "load_east", bus="east", p_set=create_load_pattern(600))

    # Add different types of existing generation

    # 1. Existing coal plant (committable, fixed capacity)
    n.add(
        "Generator",
        "coal_existing",
        bus="south",
        p_nom=400,
        committable=True,
        marginal_cost=35,
        p_min_pu=0.4,
        start_up_cost=8000,
        shut_down_cost=2000,
        min_up_time=8,
        min_down_time=6,
    )

    # 2. Existing gas peaker (committable, fixed capacity)
    n.add(
        "Generator",
        "gas_peaker_existing",
        bus="north",
        p_nom=300,
        committable=True,
        marginal_cost=80,
        p_min_pu=0.3,
        start_up_cost=1000,
        shut_down_cost=500,
        min_up_time=2,
        min_down_time=1,
    )

    # 3. New combined cycle gas plant (committable + extendable)
    n.add(
        "Generator",
        "gas_cc_new",
        bus="south",
        p_nom_extendable=True,
        committable=True,
        marginal_cost=45,
        capital_cost=80000,
        p_nom_max=800,
        p_min_pu=0.5,
        start_up_cost=3000,
        shut_down_cost=1000,
        min_up_time=4,
        min_down_time=4,
    )

    # 4. New gas peaker plants (committable + extendable)
    for i, bus in enumerate(["north", "east"]):
        n.add(
            "Generator",
            f"gas_peaker_new_{bus}",
            bus=bus,
            p_nom_extendable=True,
            committable=True,
            marginal_cost=75,
            capital_cost=50000,
            p_nom_max=400,
            p_min_pu=0.2,
            start_up_cost=800,
            shut_down_cost=400,
            min_up_time=1,
            min_down_time=1,
        )

    # 5. Renewable energy (extendable only, no commitment)
    n.add(
        "Generator",
        "wind_north",
        bus="north",
        p_nom_extendable=True,
        marginal_cost=0,
        capital_cost=120000,
        p_nom_max=600,
        p_max_pu=[
            0.8 + 0.2 * np.sin(2 * np.pi * t / (24 * 7))
            for t in range(len(n.snapshots))
        ],
    )

    n.add(
        "Generator",
        "solar_south",
        bus="south",
        p_nom_extendable=True,
        marginal_cost=0,
        capital_cost=100000,
        p_nom_max=500,
        p_max_pu=[
            max(0, np.sin(np.pi * (t % 24) / 24)) for t in range(len(n.snapshots))
        ],
    )

    # 6. Storage (extendable)
    n.add(
        "StorageUnit",
        "battery_east",
        bus="east",
        p_nom_extendable=True,
        capital_cost=150000,
        p_nom_max=200,
        max_hours=4,
        efficiency_store=0.9,
        efficiency_dispatch=0.9,
        marginal_cost=1,
    )

    # Optimize the system
    status, termination_code = n.optimize(solver_name="highs")

    assert status == "ok", f"Optimization failed with status: {status}"

    # Verify results

    # 1. Power balance should be maintained
    total_load = (n.loads_t.p.sum(axis=1) * 1).sum()  # Sum over time and buses
    total_generation = (n.generators_t.p.sum(axis=1) * 1).sum()
    total_storage_net = 0
    if hasattr(n, "storage_units_t") and not n.storage_units_t.p.empty:
        total_storage_net = (n.storage_units_t.p.sum(axis=1) * 1).sum()

    power_balance_error = abs(total_load - total_generation - total_storage_net)
    assert power_balance_error < 1e-1, f"Power balance error: {power_balance_error}"

    # 2. Check that extendable generators got reasonable capacities
    extendable_gens = n.generators.loc[n.generators.p_nom_extendable].index
    for gen in extendable_gens:
        p_nom_opt = n.generators.p_nom_opt.loc[gen]
        assert p_nom_opt >= 0, f"Negative capacity for {gen}: {p_nom_opt}"

        # Should not exceed maximum
        if np.isfinite(n.generators.p_nom_max.loc[gen]):
            assert p_nom_opt <= n.generators.p_nom_max.loc[gen] + 1e-6

    # 3. Check unit commitment constraints for committable generators
    committable_gens = n.generators.loc[n.generators.committable].index
    for gen in committable_gens:
        if gen in n.generators_t.status.columns:
            status_vals = n.generators_t.status[gen]
            dispatch_vals = n.generators_t.p[gen]

            if gen in extendable_gens:
                p_nom_opt = n.generators.p_nom_opt.loc[gen]
            else:
                p_nom_opt = n.generators.p_nom.loc[gen]

            p_min_pu = n.generators.p_min_pu.loc[gen]
            min_power = p_min_pu * p_nom_opt

            for t in range(len(status_vals)):
                if status_vals.iloc[t] > 0.5:  # Online
                    assert dispatch_vals.iloc[t] >= min_power - 1e-6, (
                        f"Generator {gen} violates minimum power at time {t}"
                    )
                else:  # Offline
                    assert dispatch_vals.iloc[t] <= 1e-6, (
                        f"Generator {gen} has power when offline at time {t}"
                    )

    # 4. Economic reasonableness checks
    objective_value = n.objective
    assert objective_value > 0, "Objective value should be positive"

    # Print summary
    print(f"Optimization successful! Objective: {objective_value:.0f}")
    print("Installed capacities:")
    for gen in n.generators.index:
        if n.generators.p_nom_extendable.loc[gen]:
            cap = n.generators.p_nom_opt.loc[gen]
            print(f"  {gen}: {cap:.1f} MW")
        else:
            cap = n.generators.p_nom.loc[gen]
            print(f"  {gen}: {cap:.1f} MW (fixed)")

    return n


def test_renewable_integration_with_flexible_backup():
    """Test high renewable penetration with flexible backup generation."""
    n = pypsa.Network()
    n.set_snapshots(range(72))  # 3 days

    n.add("Bus", "main")

    # Variable renewable load
    load_pattern = []
    for t in range(72):
        hour_of_day = t % 24
        # Higher load during day, lower at night
        daily_pattern = 0.7 + 0.3 * max(0, np.sin(np.pi * (hour_of_day - 6) / 12))
        # Some variability
        noise = 1 + 0.1 * np.sin(2 * np.pi * t / 17)  # Some variability
        load_pattern.append(1000 * daily_pattern * noise)

    n.add("Load", "load", bus="main", p_set=load_pattern)

    # High renewable generation with variability
    wind_pattern = []
    solar_pattern = []
    for t in range(72):
        hour_of_day = t % 24

        # Wind: more variable, can be high at night
        wind_factor = 0.3 + 0.7 * (
            0.5 + 0.5 * np.sin(2 * np.pi * t / 27)
        )  # Longer cycle
        wind_pattern.append(wind_factor)

        # Solar: only during day
        if 6 <= hour_of_day <= 18:
            solar_factor = np.sin(np.pi * (hour_of_day - 6) / 12)
        else:
            solar_factor = 0
        solar_pattern.append(solar_factor)

    # Large renewable capacity (lower capital costs to make them competitive)
    n.add(
        "Generator",
        "wind_large",
        bus="main",
        p_nom_extendable=True,
        marginal_cost=0,
        capital_cost=40000,  # Much lower to compete with gas
        p_nom_max=1500,
        p_max_pu=wind_pattern,
    )

    n.add(
        "Generator",
        "solar_large",
        bus="main",
        p_nom_extendable=True,
        marginal_cost=0,
        capital_cost=35000,  # Much lower to compete with gas
        p_nom_max=1200,
        p_max_pu=solar_pattern,
    )

    # Flexible backup generation (committable + extendable)
    n.add(
        "Generator",
        "gas_flexible",
        bus="main",
        p_nom_extendable=True,
        committable=True,
        marginal_cost=60,
        capital_cost=70000,
        p_nom_max=800,
        p_min_pu=0.2,  # Very flexible
        start_up_cost=500,  # Low startup cost for flexibility
        shut_down_cost=200,
    )

    # Fast-responding backup (high startup cost but very responsive)
    n.add(
        "Generator",
        "gas_fast",
        bus="main",
        p_nom_extendable=True,
        committable=True,
        marginal_cost=80,
        capital_cost=60000,
        p_nom_max=600,
        p_min_pu=0.1,  # Very low minimum
        start_up_cost=200,  # Very low startup cost
        shut_down_cost=100,
        min_up_time=1,  # Very responsive
        min_down_time=1,
    )

    # Energy storage
    n.add(
        "StorageUnit",
        "battery_large",
        bus="main",
        p_nom_extendable=True,
        capital_cost=180000,
        p_nom_max=400,
        max_hours=6,
        efficiency_store=0.9,
        efficiency_dispatch=0.9,
        marginal_cost=0.5,
    )

    status, termination_code = n.optimize(solver_name="highs")
    assert status == "ok"

    # Debug: check what capacities were built
    print("Built capacities:")
    for gen in n.generators.index:
        if n.generators.p_nom_extendable.loc[gen]:
            cap = n.generators.p_nom_opt.loc[gen]
            print(f"  {gen}: {cap:.1f} MW")

    # Check renewable penetration
    renewable_generation = (
        n.generators_t.p["wind_large"].sum() + n.generators_t.p["solar_large"].sum()
    )
    total_generation = n.generators_t.p.sum().sum()

    if total_generation > 0:
        renewable_share = renewable_generation / total_generation
    else:
        renewable_share = 0

    print(f"Renewable share: {renewable_share:.1%}")

    # Check that some renewables were built (may be low due to storage costs)
    wind_capacity = n.generators.p_nom_opt.loc["wind_large"]
    solar_capacity = n.generators.p_nom_opt.loc["solar_large"]

    assert wind_capacity > 0 or solar_capacity > 0, (
        f"Expected some renewable capacity, got wind: {wind_capacity}, solar: {solar_capacity}"
    )

    # Check that backup generation is used appropriately
    backup_generators = ["gas_flexible", "gas_fast"]
    total_online_time = 0
    for gen in backup_generators:
        if gen in n.generators_t.status.columns:
            online_fraction = n.generators_t.status[gen].mean()
            total_online_time += online_fraction
            print(f"{gen} online fraction: {online_fraction:.1%}")

    # At least one backup should not be running constantly if renewables are working
    # (though individual units might run often due to minimum load constraints)
    print(f"Total backup online time: {total_online_time:.2f}")
    assert renewable_share > 0.2, (
        f"Expected renewable share > 20%, got {renewable_share:.1%}"
    )

    return n


if __name__ == "__main__":
    print("Testing power system expansion planning...")
    test_power_system_expansion_planning()

    print("\nTesting renewable integration...")
    test_renewable_integration_with_flexible_backup()

    print("\nAll real-world scenario tests passed!")
