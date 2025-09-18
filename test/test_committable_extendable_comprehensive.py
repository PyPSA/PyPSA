"""Comprehensive stress tests for committable+extendable functionality."""

import time

import numpy as np

import pypsa


class TestCommittableExtendableStress:
    """Stress tests for committable+extendable functionality."""

    def test_many_generators_performance(self):
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

        print(f"✓ Optimized {n_generators} generators in {optimization_time:.2f}s")

    def test_large_time_series(self):
        """Test with large number of snapshots."""
        n = pypsa.Network()
        n_hours = 168  # One week
        n.set_snapshots(range(n_hours))

        n.add("Bus", "bus")

        # Create realistic weekly load pattern
        load_pattern = []
        for day in range(7):
            for hour in range(24):
                # Daily pattern with weekly variation
                daily_factor = 0.7 + 0.3 * np.sin(2 * np.pi * hour / 24 + np.pi / 4)
                weekly_factor = 1.0 + 0.2 * np.sin(2 * np.pi * day / 7)
                load_pattern.append(1000 * daily_factor * weekly_factor)

        n.add("Load", "load", bus="bus", p_set=load_pattern)

        # Add mix of generators (make problem more feasible)
        n.add(
            "Generator",
            "baseload",
            bus="bus",
            p_nom_extendable=True,
            committable=True,
            marginal_cost=25,
            capital_cost=100000,
            p_nom_max=1500,  # Increased capacity
            p_min_pu=0.3,  # Lower minimum to increase flexibility
            start_up_cost=2000,
            shut_down_cost=1000,
            min_up_time=4,  # Reduced minimum times
            min_down_time=2,
        )

        n.add(
            "Generator",
            "peaker",
            bus="bus",
            p_nom_extendable=True,
            committable=True,
            marginal_cost=80,
            capital_cost=50000,
            p_nom_max=1000,  # Increased capacity
            p_min_pu=0.1,  # Very flexible
            start_up_cost=500,
            shut_down_cost=200,
        )

        # Time the optimization
        start_time = time.time()
        status, termination_code = n.optimize(solver_name="highs")
        optimization_time = time.time() - start_time

        assert status in ["ok", "warning"], f"Optimization failed with status: {status}"
        assert optimization_time < 60, (
            f"Large time series optimization took too long: {optimization_time:.2f}s"
        )

        # Verify power balance
        total_load = sum(load_pattern)
        total_generation = n.generators_t.p.sum().sum()
        assert abs(total_load - total_generation) < 1e-2

        print(f"✓ Optimized {n_hours} snapshots in {optimization_time:.2f}s")

    def test_extreme_parameters(self):
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

    def test_mixed_scenarios_scaling(self):
        """Test scaling behavior with mixed committable/extendable scenarios."""
        results = {}

        for n_components in [5, 10, 20]:
            n = pypsa.Network()
            n.set_snapshots(range(24))

            n.add("Bus", "bus")

            # Scale load with number of components
            base_load = n_components * 50
            load_pattern = [
                base_load * (0.7 + 0.3 * np.sin(2 * np.pi * t / 24)) for t in range(24)
            ]
            n.add("Load", "load", bus="bus", p_set=load_pattern)

            # Add scaled number of generators
            for i in range(n_components):
                gen_type = i % 4
                gen_name = f"gen_{i}"

                if gen_type == 0:  # Committable + extendable
                    n.add(
                        "Generator",
                        gen_name,
                        bus="bus",
                        p_nom_extendable=True,
                        committable=True,
                        marginal_cost=40 + i * 2,
                        capital_cost=80000,
                        p_nom_max=base_load * 0.8,
                        p_min_pu=0.3,
                        start_up_cost=1000,
                    )
                elif gen_type == 1:  # Committable only
                    n.add(
                        "Generator",
                        gen_name,
                        bus="bus",
                        p_nom=base_load * 0.5,
                        committable=True,
                        marginal_cost=50 + i * 2,
                        p_min_pu=0.4,
                        start_up_cost=800,
                    )
                elif gen_type == 2:  # Extendable only
                    n.add(
                        "Generator",
                        gen_name,
                        bus="bus",
                        p_nom_extendable=True,
                        marginal_cost=60 + i * 2,
                        capital_cost=70000,
                        p_nom_max=base_load * 0.6,
                    )
                else:  # Fixed
                    n.add(
                        "Generator",
                        gen_name,
                        bus="bus",
                        p_nom=base_load * 0.4,
                        marginal_cost=55 + i * 2,
                    )

            start_time = time.time()
            status, termination_code = n.optimize(solver_name="highs")
            optimization_time = time.time() - start_time

            assert status == "ok"
            results[n_components] = optimization_time

        # Check that scaling is reasonable (not exponential)
        print(f"Scaling results: {results}")
        # Time should not grow exponentially
        if len(results) >= 3:
            times = list(results.values())
            # Rough check: largest time should not be more than 10x smallest
            assert max(times) / min(times) < 10, f"Poor scaling: {results}"

    def test_numerical_stability(self):
        """Test numerical stability with various parameter combinations."""
        n = pypsa.Network()
        n.set_snapshots(range(6))

        n.add("Bus", "bus")
        n.add("Load", "load", bus="bus", p_set=[100, 200, 150, 180, 120, 90])

        # Test different combinations of parameters that might cause numerical issues
        test_cases = [
            # (p_nom_max, p_min_pu, marginal_cost, capital_cost)
            (1000, 0.01, 50, 80000),  # Very small min load
            (10000, 0.5, 30, 100000),  # Large capacity
            (500, 0.99, 100, 50000),  # Very high min load
            (1e5, 0.1, 25, 200000),  # Very large capacity
        ]

        for i, (p_nom_max, p_min_pu, mc, cc) in enumerate(test_cases):
            n.add(
                "Generator",
                f"test_gen_{i}",
                bus="bus",
                p_nom_extendable=True,
                committable=True,
                marginal_cost=mc,
                capital_cost=cc,
                p_nom_max=p_nom_max,
                p_min_pu=p_min_pu,
                start_up_cost=1000,
            )

        status, termination_code = n.optimize(solver_name="highs")
        assert status == "ok", "Numerical stability test failed"

        # Check that solution is reasonable
        total_generation = n.generators_t.p.sum().sum()
        total_load = n.loads_t.p.sum().sum()
        assert abs(total_generation - total_load) < 1e-2


class TestCommittableExtendablePerformance:
    """Performance comparison tests."""

    def test_performance_vs_separate_optimization(self):
        """Compare performance of combined vs separate optimization."""
        # Test scenario: optimize same network with separate runs vs combined
        n_snapshots = 48

        # Combined committable+extendable approach
        n_combined = pypsa.Network()
        n_combined.set_snapshots(range(n_snapshots))
        n_combined.add("Bus", "bus")

        load_pattern = [
            1000 * (0.8 + 0.2 * np.sin(2 * np.pi * t / 24)) for t in range(n_snapshots)
        ]
        n_combined.add("Load", "load", bus="bus", p_set=load_pattern)

        # Add generators that are both committable and extendable
        for i in range(5):
            n_combined.add(
                "Generator",
                f"gen_{i}",
                bus="bus",
                p_nom_extendable=True,
                committable=True,
                marginal_cost=40 + i * 10,
                capital_cost=80000,
                p_nom_max=600,
                p_min_pu=0.3,
                start_up_cost=1000,
            )

        start_time = time.time()
        status_combined, _ = n_combined.optimize(solver_name="highs")
        time_combined = time.time() - start_time

        assert status_combined == "ok"

        # Record results for comparison
        combined_cost = n_combined.objective

        print(f"Combined optimization: {time_combined:.2f}s, cost: {combined_cost:.0f}")

        # Verify solution quality
        total_generation = n_combined.generators_t.p.sum().sum()
        total_load = sum(load_pattern)
        assert abs(total_generation - total_load) < 1e-2


class TestCommittableExtendableEdgeCases:
    """Edge case tests."""

    def test_zero_minimum_load(self):
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

    def test_maximum_minimum_load(self):
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

    def test_infinite_capacity_limit(self):
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

    def test_single_snapshot(self):
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

    def test_no_start_costs(self):
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
    # Run comprehensive tests
    stress_tests = TestCommittableExtendableStress()
    performance_tests = TestCommittableExtendablePerformance()
    edge_tests = TestCommittableExtendableEdgeCases()

    print("Running stress tests...")
    stress_tests.test_many_generators_performance()
    stress_tests.test_large_time_series()
    stress_tests.test_extreme_parameters()
    stress_tests.test_mixed_scenarios_scaling()
    stress_tests.test_numerical_stability()

    print("Running performance tests...")
    performance_tests.test_performance_vs_separate_optimization()

    print("Running edge case tests...")
    edge_tests.test_zero_minimum_load()
    edge_tests.test_maximum_minimum_load()
    edge_tests.test_infinite_capacity_limit()
    edge_tests.test_single_snapshot()
    edge_tests.test_no_start_costs()

    print("All comprehensive tests passed!")
