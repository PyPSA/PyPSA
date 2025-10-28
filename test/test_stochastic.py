# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""
Test stochastic functionality of PyPSA networks.
"""

import warnings
from pathlib import Path

import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as equal
from xarray import DataArray

import pypsa
from pypsa.common import expand_series
from pypsa.components.common import as_components


def test_stoch_example():
    n = pypsa.examples.stochastic_network()

    n.consistency_check(strict="all")


def test_network_properties():
    """Test basic properties of a stochastic network."""
    snapshots = pd.date_range("2024-01-01", periods=5, freq="h")
    n = pypsa.Network(snapshots=snapshots)
    n.add("Bus", "bus1")
    n.add("Load", "load1", bus="bus1", p_set=100)

    # Set up scenarios
    n.set_scenarios({"low": 0.33, "medium": 0.34, "high": 0.33})

    # Check scenarios were properly set up
    assert len(n.scenarios) == 3
    assert "low" in n.scenarios
    assert "medium" in n.scenarios
    assert "high" in n.scenarios
    assert n.has_scenarios

    # Check probabilities sum to 1
    assert abs(n.scenario_weightings["weight"].sum() - 1.0) < 1e-10

    p_set = n.get_switchable_as_dense("Load", "p_set")

    assert p_set.columns.names == ["scenario", "name"]

    # Check data shape for each scenario
    for scenario in n.scenarios:
        assert p_set.loc[:, scenario].shape[0] == 5

    # Check string representation contains scenario information
    assert "Scenarios:" in repr(n)


def test_example_consistency(ac_dc_stochastic):
    n = ac_dc_stochastic
    n.c.lines.static.x = n.c.lines.static.x.where(
        n.c.lines.static.x > 0, 0.0001
    )  # Avoid zero reactance
    n.c.lines.static.r = n.c.lines.static.r.where(
        n.c.lines.static.r > 0, 0.0001
    )  # Avoid zero reactance
    n.consistency_check(strict="all")


def test_component_functions(ac_dc_stochastic):
    assert isinstance(ac_dc_stochastic.branches(), pd.DataFrame)
    assert isinstance(ac_dc_stochastic.passive_branches(), pd.DataFrame)
    assert isinstance(ac_dc_stochastic.controllable_branches(), pd.DataFrame)


def test_calculate_dependent_values(ac_dc_stochastic: pypsa.Network):
    """
    Test the calculation of dependent values in a stochastic network.
    This includes checking that the function runs without errors and that
    the expected attributes are present in the network object.
    """
    n = ac_dc_stochastic
    n.calculate_dependent_values()
    assert n.c.lines.static.x_pu_eff.notnull().all()


def test_determine_network_topology(ac_dc_stochastic: pypsa.Network):
    """
    Test the determination of network topology in a stochastic network.
    This includes checking that the function runs without errors and that
    the expected attributes are present in the network object.
    """
    n = ac_dc_stochastic
    n.determine_network_topology()

    assert not n.c.sub_networks.static.empty
    assert "AC" in n.c.sub_networks.static.carrier.values

    sub = n.c.sub_networks.static.obj.loc["0"]
    assert not sub.components.generators.static.empty
    assert not sub.components.loads.static.empty

    # check slack bus and slack generator assignment via subnetworks and network components
    assert set(
        n.c.generators.static.query("control == 'Slack'").index.unique("name")
    ) == set(n.c.sub_networks.static.obj.map(lambda sub: sub.slack_generator).dropna())
    assert set(
        n.c.buses.static.query("control == 'Slack'").index.unique("name")
    ) == set(n.c.sub_networks.static.obj.map(lambda sub: sub.slack_bus).dropna())


def test_cycles(ac_dc_stochastic: pypsa.Network):
    n = ac_dc_stochastic
    C = n.cycle_matrix()

    assert isinstance(C, pd.DataFrame)
    assert C.notnull().all().all()  # Check for NaN values

    # repeat with apply weights
    n.calculate_dependent_values()
    C = n.cycle_matrix(apply_weights=True)
    assert isinstance(C, pd.DataFrame)
    assert C.notnull().all().all()  # Check for NaN values


def test_model_creation(stochastic_benchmark_network):
    """
    Test stochastic optimization model variable and constraint dimensions.

    Verifies that when creating an optimization model for a stochastic network:

    Variables:
    - Operational variables (e.g., Generator-p) include scenario dimension
    - Investment variables (e.g., Generator-p_nom) exclude scenario dimension

    Constraints:
    - Operational constraints include (scenario, component, snapshot) dimensions
    - Investment constraints include (component, scenario) dimensions
    """
    n = stochastic_benchmark_network
    n.optimize.create_model()

    assert hasattr(n, "model")
    assert n.model is not None

    # Test operational variable Generator-p has scenario dimension
    assert n.model.variables["Generator-p"].dims == (
        "scenario",
        "name",
        "snapshot",
    )

    # Test that Generator-p_nom does not have scenario dimension (investment variable)
    assert n.model.variables["Generator-p_nom"].dims == ("name",)

    # Test operational constraints have scenario dimension

    # Generator-ext-p_nom-lower constraint should have (name, scenario) dimensions
    assert {
        d
        for d in n.model.constraints["Generator-ext-p_nom-lower"].sizes.keys()
        if d != "_term"
    } == {"name", "scenario"}

    # Generator-ext-p-lower constraint should have (scenario, name, snapshot) dimensions
    assert {
        d
        for d in n.model.constraints["Generator-ext-p-lower"].sizes.keys()
        if d != "_term"
    } == {"scenario", "name", "snapshot"}

    # Generator-ext-p-upper constraint should have (scenario, name, snapshot) dimensions
    assert {
        d
        for d in n.model.constraints["Generator-ext-p-upper"].sizes.keys()
        if d != "_term"
    } == {"scenario", "name", "snapshot"}

    # Bus-nodal_balance constraint should have (name, scenario, snapshot) dimensions
    assert {
        d for d in n.model.constraints["Bus-nodal_balance"].sizes.keys() if d != "_term"
    } == {"name", "scenario", "snapshot"}


def test_statistics(ac_dc_stochastic_r):
    """
    Test the statistics of a stochastic network.
    """
    n = ac_dc_stochastic_r
    ds = n.statistics.installed_capacity()
    assert isinstance(ds, pd.Series)
    assert isinstance(ds.index, pd.MultiIndex)
    assert "scenario" in ds.index.names
    assert not ds.empty

    stats = n.statistics()
    assert isinstance(stats, pd.DataFrame)
    assert isinstance(stats.index, pd.MultiIndex)
    assert "scenario" in ds.index.names
    assert not stats.empty

    df = n.statistics.supply(groupby_time=False)
    assert isinstance(df, pd.DataFrame)
    assert isinstance(df.index, pd.MultiIndex)
    assert "scenario" in df.index.names
    assert not df.empty


def test_statistics_plot(ac_dc_stochastic_r):
    """
    Test the statistics plot of a stochastic network.
    """
    n = ac_dc_stochastic_r
    s = n.statistics
    s.installed_capacity.plot.bar()


def test_optimization_with_scenarios(ac_dc_stochastic):
    """
    Test optimization of a stochastic network and compare results with deterministic equivalent.

    This test verifies that:
    - Stochastic optimization completes successfully
    - The objective value matches a deterministic network with identical data
    """
    n = ac_dc_stochastic
    status, _ = n.optimize(solver_name="highs")
    assert status == "ok"

    m = pypsa.examples.ac_dc_meshed()
    m.optimize(solver_name="highs")
    assert abs(m.objective - n.objective) < 1e-2, (
        f"Expected objective {m.objective}, got {n.objective}"
    )
    assert abs(m.objective_constant - n.objective_constant) < 1e-2, (
        f"Expected objective constant {m.objective_constant}, got {n.objective_constant}"
    )


def test_solved_network_simple(stochastic_benchmark_network):
    """
    Solve the stochastic problem and compare results with benchmark data.
    Simple test case with a single bus and multiple generators.
    """
    # Load the benchmark results
    benchmark_path = Path(__file__).parent / "data" / "benchmark-sp"

    if not benchmark_path.exists():
        pytest.skip("Benchmark data not available")

    n_r = pypsa.Network(benchmark_path)

    # Create a new network for the stochastic model
    n = stochastic_benchmark_network

    # GAS_PRICES = {"low": 40, "med": 70, "high": 100}
    n.c.generators.static.loc[("medium", "gas"), "marginal_cost"] = (
        70 / n.c.generators.static.loc[("medium", "gas"), "efficiency"]
    )
    n.c.generators.static.loc[("high", "gas"), "marginal_cost"] = (
        100 / n.c.generators.static.loc[("high", "gas"), "efficiency"]
    )

    status, _ = n.optimize(solver_name="highs")
    assert status == "ok"

    # Compare generator capacities (these are the main result of stochastic planning)
    equal(
        n.c.generators.static.p_nom_opt.loc["low", :],
        n_r.c.generators.static.p_nom_opt,
        decimal=2,
    )

    # Compare objective value
    equal(n.objective, n_r.objective, decimal=2)


def test_solved_network_multiperiod():
    """
    Test combined stochastic + multiperiod optimization.

    Creates a multiperiod network with investment periods and scenarios,
    then verifies that the optimization completes successfully and produces
    expected results for both scenarios and investment periods.
    """

    # Suppress pandas FutureWarning about fillna downcasting for entire test
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=".*Downcasting object dtype arrays.*",
        )

        # Combined multiperiod + stochastic optimization
        n = pypsa.Network(snapshots=range(3))
        n.investment_periods = [2020, 2030]

        n.add("Carrier", "elec")
        n.add("Bus", "bus1", carrier="elec")
        n.add(
            "Generator",
            "gen-2020",
            bus="bus1",
            p_nom_extendable=True,
            capital_cost=100,
            marginal_cost=10,
            build_year=2020,
            lifetime=30,
            carrier="elec",
        )
        n.add(
            "Generator",
            "gen-2030",
            bus="bus1",
            p_nom_extendable=True,
            capital_cost=110,
            marginal_cost=11,
            build_year=2030,
            lifetime=30,
            carrier="elec",
        )
        n.add("Load", "load1", bus="bus1", p_set=100)

        # Now set scenarios
        n.set_scenarios({"high": 0.5, "low": 0.5})

        # Set scenario-specific loads for multiperiod (6 snapshots total: 2 periods × 3 timesteps)
        n.c.loads.dynamic.p_set = pd.DataFrame(
            index=n.snapshots,
            columns=pd.MultiIndex.from_product(
                [n.scenarios, ["load1"]], names=["scenario", "name"]
            ),
        )

        load_high = [120, 144, 132] * len(n.investment_periods)
        load_low = [80, 96, 88] * len(n.investment_periods)

        n.c.loads.dynamic.p_set.loc[:, ("high", "load1")] = load_high
        n.c.loads.dynamic.p_set.loc[:, ("low", "load1")] = load_low

        # This should now work with both multiperiod and stochastic features!
        status, condition = n.optimize(multi_investment_periods=True)
        assert status == "ok"

        # Verify we have results for both scenarios and investment periods
        assert "high" in n.c.generators.dynamic.p.columns.get_level_values("scenario")
        assert "low" in n.c.generators.dynamic.p.columns.get_level_values("scenario")

        # Check basic energy balance for each scenario
        for scenario in ["high", "low"]:
            gen_output = (
                n.c.generators.dynamic.p.loc[:, (scenario, slice(None))].sum().sum()
            )
            load_demand = n.c.loads.dynamic.p_set.loc[:, (scenario, "load1")].sum()
            # Generation should equal load
            assert abs(gen_output - load_demand) < 1e-1

        # Verify that high scenario has higher generation than low scenario
        gen_high = n.c.generators.dynamic.p.loc[:, ("high", slice(None))].sum().sum()
        gen_low = n.c.generators.dynamic.p.loc[:, ("low", slice(None))].sum().sum()
        assert gen_high > gen_low

        # Test multiperiod-specific functionality
        p_nom_opt = n.c.generators.static.p_nom_opt
        assert (
            len(p_nom_opt) == 4
        )  # Should have optimal capacities for both generators × both scenarios

        # Verify we have generators for both scenarios
        scenarios_in_gens = p_nom_opt.index.get_level_values("scenario").unique()
        assert "high" in scenarios_in_gens
        assert "low" in scenarios_in_gens


def test_single_scenario():
    """
    Test that a network with a single scenario works correctly.

    Verifies that:
    - Single-scenario stochastic networks optimize successfully
    - Scenario indexing works correctly with one scenario
    - Solution is identical to a non-stochastic network with same data
    """

    # Suppress pandas FutureWarning about fillna downcasting
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=".*Downcasting object dtype arrays.*",
        )
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=".*Setting an item of incompatible dtype.*",
        )

        # Create a simple network
        n = pypsa.Network(snapshots=range(3))
        n.add("Carrier", "elec")
        n.add("Bus", "bus1", carrier="elec")
        n.add(
            "Generator",
            "gen1",
            bus="bus1",
            p_nom_extendable=True,
            capital_cost=100,
            marginal_cost=10,
            carrier="elec",
        )
        n.add("Load", "load1", bus="bus1", p_set=[100, 120, 110])

        # Solve deterministic problem first
        status_det, _ = n.optimize()
        assert status_det == "ok"
        obj_det = n.objective
        capacity_det = n.c.generators.static.p_nom_opt.loc["gen1"]
        dispatch_det = n.c.generators.dynamic.p.loc[:, "gen1"].sum()

        # Convert to single-scenario stochastic
        n.set_scenarios(["scenario"])

        # Set scenario-specific load data (same as deterministic)
        n.c.loads.dynamic.p_set = pd.DataFrame(
            index=n.snapshots,
            columns=pd.MultiIndex.from_product(
                [n.scenarios, ["load1"]], names=["scenario", "name"]
            ),
        )
        n.c.loads.dynamic.p_set.loc[:, ("scenario", "load1")] = pd.Series(
            [100.0, 120.0, 110.0], dtype=float
        )

        # Solve stochastic problem
        status_stoch, _ = n.optimize()
        assert status_stoch == "ok"

        # Verify structure
        assert len(n.scenarios) == 1
        assert "scenario" in n.scenarios
        assert "scenario" in n.c.generators.dynamic.p.columns.get_level_values(
            "scenario"
        )

        # Compare solutions (should be identical)
        assert abs(n.objective - obj_det) < 1e-6

        stoch_capacity = n.c.generators.static.p_nom_opt.loc[("scenario", "gen1")]
        assert abs(stoch_capacity - capacity_det) < 1e-6

        stoch_dispatch = n.c.generators.dynamic.p.loc[:, ("scenario", "gen1")].sum()
        assert abs(stoch_dispatch - dispatch_det) < 1e-6

        # Energy balance check
        gen_output = (
            n.c.generators.dynamic.p.loc[:, ("scenario", slice(None))].sum().sum()
        )
        load_demand = n.c.loads.dynamic.p_set.loc[:, ("scenario", "load1")].sum()
        assert abs(gen_output - load_demand) < 1e-1


def test_slack_bus_consistency_check():
    """
    Test that the consistency check correctly identifies when different slack buses
    are chosen across scenarios.
    """

    # Suppress pandas FutureWarning about fillna downcasting
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=".*Downcasting object dtype arrays.*",
        )

        # Create a simple stochastic network
        n = pypsa.Network(snapshots=range(3))
        n.add("Carrier", "elec")
        n.add("Bus", "bus1", carrier="elec")
        n.add("Bus", "bus2", carrier="elec")
        n.add(
            "Generator",
            "gen1",
            bus="bus1",
            p_nom_extendable=True,
            capital_cost=100,
            marginal_cost=10,
            carrier="elec",
        )
        n.add(
            "Generator",
            "gen2",
            bus="bus2",
            p_nom_extendable=True,
            capital_cost=100,
            marginal_cost=10,
            carrier="elec",
        )
        n.add("Load", "load1", bus="bus1", p_set=[100, 120, 110])

        n.set_scenarios(["scenario1", "scenario2"])

        # Manually set different slack buses across scenarios to trigger the check
        # This simulates what would happen if different slack buses were chosen
        # during topology determination
        if n.c.buses.static.index.nlevels > 1:
            n.c.buses.static.loc[("scenario1", "bus1"), "control"] = "Slack"
            n.c.buses.static.loc[("scenario1", "bus2"), "control"] = "PQ"
            n.c.buses.static.loc[("scenario2", "bus1"), "control"] = "PQ"
            n.c.buses.static.loc[("scenario2", "bus2"), "control"] = (
                "Slack"  # Different slack bus!
            )

            # Now run the slack bus consistency check and expect it to raise a warning
            from pypsa.consistency import check_stochastic_slack_bus_consistency

            # Test with strict=False (should log warning)
            check_stochastic_slack_bus_consistency(n, strict=False)

            # Test with strict=True (should raise error)
            import pytest

            with pytest.raises(pypsa.consistency.ConsistencyError):
                check_stochastic_slack_bus_consistency(n, strict=True)


def test_slack_bus_consistency_check_passes():
    """
    Test that the consistency check passes when the same slack bus is chosen
    across scenarios.
    """

    # Suppress pandas FutureWarning about fillna downcasting
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=".*Downcasting object dtype arrays.*",
        )

        # Create a simple stochastic network
        n = pypsa.Network(snapshots=range(3))
        n.add("Carrier", "elec")
        n.add("Bus", "bus1", carrier="elec")
        n.add("Bus", "bus2", carrier="elec")
        n.add(
            "Generator",
            "gen1",
            bus="bus1",
            p_nom_extendable=True,
            capital_cost=100,
            marginal_cost=10,
            carrier="elec",
        )
        n.add(
            "Generator",
            "gen2",
            bus="bus2",
            p_nom_extendable=True,
            capital_cost=100,
            marginal_cost=10,
            carrier="elec",
        )
        n.add("Load", "load1", bus="bus1", p_set=[100, 120, 110])

        n.set_scenarios(["scenario1", "scenario2"])

        # Set the same slack bus across scenarios (should pass)
        if n.c.buses.static.index.nlevels > 1:
            n.c.buses.static.loc[("scenario1", "bus1"), "control"] = "Slack"
            n.c.buses.static.loc[("scenario1", "bus2"), "control"] = "PQ"
            n.c.buses.static.loc[("scenario2", "bus1"), "control"] = (
                "Slack"  # Same slack bus
            )
            n.c.buses.static.loc[("scenario2", "bus2"), "control"] = "PQ"

            # Now run the slack bus consistency check - should pass without error
            from pypsa.consistency import check_stochastic_slack_bus_consistency

            # Should not raise any error or warning
            check_stochastic_slack_bus_consistency(n, strict=True)


def test_store_stochastic_optimization_bug():
    """Test that Store component works correctly with stochastic optimization.

    This test reproduces the bug where Store components failed during stochastic
    optimization due to dimension mismatch in the standing efficiency calculation.

    The bug was:
    - `expand_series` returns a DataFrame with shape (snapshots, scenarios)
    - `standing_loss` is a DataArray with shape (snapshots, scenarios, stores)
    - The power operation (1 - standing_loss) ** eh failed with 3D vs 2D mismatch

    The fix:
    - Convert expand_series result to DataArray and try to unstack it
    - This aligns the dimensions properly for the power operation
    """
    n = pypsa.examples.model_energy()

    # Reduce to first day only (8 snapshots) to make test faster
    n.set_snapshots(n.snapshots[:8])

    # Ensure the network has stores (it should)
    assert not n.c.stores.static.empty, "Test network should have stores"

    # The bug occured in operation (1 - standing_loss)**eh due to dimension mismatch
    n.c.stores.static.at["hydrogen storage", "e_nom"] = 1000
    n.c.stores.static.at["hydrogen storage", "e_cyclic"] = False
    n.c.stores.static.at["hydrogen storage", "e_initial"] = 800
    n.c.stores.static.at["hydrogen storage", "standing_loss"] = 0.01

    # Test without scenarios first (should work)
    n_regular = n.copy()
    status_regular, condition_regular = n_regular.optimize()
    assert status_regular == "ok"
    assert condition_regular == "optimal"

    # Test with scenarios (this used to fail)
    n_stochastic = n.copy()
    n_stochastic.set_scenarios(["scenario_a", "scenario_b"])

    # This should not raise an error
    status_stochastic, condition_stochastic = n_stochastic.optimize()
    assert status_stochastic == "ok"
    assert condition_stochastic == "optimal"

    # Verify that the stochastic network has the expected structure
    assert n_stochastic.has_scenarios
    assert len(n_stochastic.scenarios) == 2

    # Verify stores have MultiIndex
    assert isinstance(n_stochastic.c.stores.static.index, pd.MultiIndex)
    assert n_stochastic.c.stores.static.index.names == ["scenario", "name"]

    # Verify optimization results exist
    assert not n_stochastic.c.stores.dynamic.e.empty
    assert not n_stochastic.c.stores.dynamic.p.empty

    # Verify specific energy level at second snapshot
    # it is 800 × (1 - 0.01)³ due to 3h temporal clustering
    second_hour_energy = n_stochastic.c.stores.dynamic.e.loc[
        n_stochastic.snapshots[1], ("scenario_a", "hydrogen storage")
    ]
    assert abs(second_hour_energy - 776.2392) < 0.01, (
        f"Expected hydrogen storage energy ~776.24 at second snapshot, got {second_hour_energy}"
    )


def test_store_stochastic_dimensions():
    """Test that Store component expansion works correctly with stochastic dimensions.

    This test specifically checks the expand_series -> DataArray conversion
    that fixes the dimension mismatch issue.
    """

    n = pypsa.Network()
    n.add("Bus", "bus")
    n.add("Store", "store", bus="bus", e_nom=100, standing_loss=0.01)
    n.add("Load", "load", bus="bus", p_set=50)
    n.add("Generator", "gen", bus="bus", p_nom=100, marginal_cost=20)

    n.set_scenarios(["s1", "s2"])

    c = as_components(n, "Store")
    sns = n.snapshots

    # This should work without errors
    elapsed_h = expand_series(n.snapshot_weightings.stores[sns], c.static.index)
    eh = DataArray(elapsed_h)

    # Test the unstack operation
    if n.has_scenarios:
        eh_final = eh.unstack("dim_1")
    else:
        eh_final = eh

    # This should work without dimension errors
    standing_loss = c.da.standing_loss.sel(snapshot=sns)
    eff_stand = (1 - standing_loss) ** eh_final

    # Verify the result has the expected dimensions
    assert isinstance(eff_stand, DataArray)
    assert "snapshot" in eff_stand.dims

    # The optimization should also work
    status, condition = n.optimize()
    assert status == "ok"
    assert condition == "optimal"


def test_scenario_ordering_bug():
    """Test that scenario ordering is preserved correctly in optimization results.

    This test ensures that when different scenarios have different parameter values,
    the optimization results correspond to the correct scenario labels.
    """
    n = pypsa.examples.ac_dc_meshed()

    # Say we have good and bad wind scenarios
    n.set_scenarios({"wind_lulls": 0.5, "windy": 0.5})
    wind_gens = ["Manchester Wind", "Frankfurt Wind", "Norway Wind"]
    for gen_name in wind_gens:
        n.c.generators.dynamic.p_max_pu.loc[:, ("wind_lulls", gen_name)] *= 0.3

    # Check 1: Wind lulls should have lower wind potential in model input
    expected_lulls_input = (
        n.c.generators.dynamic.p_max_pu.loc[:, ("wind_lulls", wind_gens)].sum().sum()
    )
    expected_windy_input = (
        n.c.generators.dynamic.p_max_pu.loc[:, ("windy", wind_gens)].sum().sum()
    )

    assert expected_lulls_input < expected_windy_input

    # Check 2: Wind lulls scenario should have lower wind generation
    n.optimize()

    actual_lulls_result = (
        n.c.generators.dynamic.p.loc[:, ("wind_lulls", wind_gens)].sum().sum()
    )
    actual_windy_result = (
        n.c.generators.dynamic.p.loc[:, ("windy", wind_gens)].sum().sum()
    )

    assert actual_lulls_result < actual_windy_result, (
        f"Scenario misalignment detected: wind lulls generation ({actual_lulls_result:.2f}) "
        f"should be less than windy generation ({actual_windy_result:.2f})"
    )

    # Check 3: verify that the results match the expected scenario ordering
    # by checking that the DataArray scenario coordinate order is preserved
    da_p_max_pu = n.c.generators.da.p_max_pu.sel(name="Manchester Wind")
    da_scenarios = list(da_p_max_pu.coords["scenario"].values)
    network_scenarios = list(n.scenarios)

    assert da_scenarios == network_scenarios, (
        f"DataArray scenario order {da_scenarios} does not match "
        f"network scenario order {network_scenarios}"
    )


# Multiperiod stochastic fixtures and tests
@pytest.fixture
def n_multiperiod():
    """Basic multiperiod network fixture similar to test_lopf_multiinvest.py"""
    n = pypsa.Network(snapshots=range(10))
    n.investment_periods = [2020, 2030, 2040, 2050]
    n.add("Carrier", "gencarrier")
    n.add("Carrier", "AC")
    n.add("Bus", [1, 2], carrier="AC")

    for i, period in enumerate(n.investment_periods):
        factor = (10 + i) / 10
        n.add(
            "Generator",
            [f"gen1-{period}", f"gen2-{period}"],
            bus=[1, 2],
            lifetime=30,
            build_year=period,
            capital_cost=[100 / factor, 100 * factor],
            marginal_cost=[i + 2, i + 1],
            p_nom_extendable=True,
            carrier="gencarrier",
        )

    for i, period in enumerate(n.investment_periods):
        n.add(
            "Line",
            f"line-{period}",
            bus0=1,
            bus1=2,
            length=1,
            build_year=period,
            lifetime=40,
            capital_cost=30 + i,
            x=0.0001,
            r=0.001,
            s_nom_extendable=True,
            carrier="AC",
        )

    load = range(100, 100 + len(n.snapshots))
    load = pd.DataFrame({"load1": load, "load2": load}, index=n.snapshots)
    n.add(
        "Load",
        ["load1", "load2"],
        bus=[1, 2],
        p_set=load,
    )

    return n


@pytest.fixture
def n_multiperiod_stochastic(n_multiperiod):
    """Convert multiperiod network to stochastic"""
    n = n_multiperiod
    n.set_scenarios({"high": 0.5, "low": 0.5})

    # Set scenario-specific loads
    n.c.loads.dynamic.p_set = pd.DataFrame(
        index=n.snapshots,
        columns=pd.MultiIndex.from_product(
            [n.scenarios, ["load1", "load2"]], names=["scenario", "name"]
        ),
    )

    # High scenario: 20% higher load
    base_load = range(100, 100 + len(n.snapshots))
    for load_name in ["load1", "load2"]:
        n.c.loads.dynamic.p_set.loc[:, ("high", load_name)] = [
            l * 1.2 for l in base_load
        ]
        n.c.loads.dynamic.p_set.loc[:, ("low", load_name)] = [
            l * 0.8 for l in base_load
        ]

    return n


@pytest.fixture
def n_multiperiod_sus(n_multiperiod):
    """Multiperiod network with storage units"""
    n = n_multiperiod
    # Remove some generators to activate storage
    n.remove("Generator", n.c.generators.static.query('bus == "1"').index)
    n.c.generators.static.capital_cost *= 5

    for i, period in enumerate(n.investment_periods):
        factor = (10 + i) / 10
        n.add(
            "StorageUnit",
            f"sto1-{period}",
            bus=1,
            lifetime=30,
            build_year=period,
            capital_cost=10 / factor,
            marginal_cost=i,
            p_nom_extendable=True,
        )
    return n


@pytest.fixture
def n_multiperiod_sus_stochastic(n_multiperiod_sus):
    """Convert multiperiod storage network to stochastic"""
    n = n_multiperiod_sus
    n.set_scenarios({"high": 0.6, "low": 0.4})

    # Set scenario-specific loads
    n.c.loads.dynamic.p_set = pd.DataFrame(
        index=n.snapshots,
        columns=pd.MultiIndex.from_product(
            [n.scenarios, ["load1", "load2"]], names=["scenario", "name"]
        ),
    )

    # Different load patterns for scenarios
    base_load = range(100, 100 + len(n.snapshots))
    for load_name in ["load1", "load2"]:
        n.c.loads.dynamic.p_set.loc[:, ("high", load_name)] = [
            l * 1.3 for l in base_load
        ]
        n.c.loads.dynamic.p_set.loc[:, ("low", load_name)] = [
            l * 0.7 for l in base_load
        ]

    return n


# Small focused tests
def test_multiperiod_stochastic_tiny_default():
    """Test tiny multiperiod stochastic network with default parameters"""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)

        n = pypsa.Network(snapshots=range(2))
        n.investment_periods = [2020, 2030]
        n.add("Carrier", "elec")
        n.add("Bus", 1, carrier="elec")
        n.add(
            "Generator",
            1,
            bus=1,
            p_nom_extendable=True,
            capital_cost=10,
            carrier="elec",
        )
        n.add("Load", 1, bus=1, p_set=100)

        n.set_scenarios({"high": 0.5, "low": 0.5})
        n.c.loads.dynamic.p_set = pd.DataFrame(
            index=n.snapshots,
            columns=pd.MultiIndex.from_product(
                [n.scenarios, ["1"]], names=["scenario", "name"]
            ),
        )
        n.c.loads.dynamic.p_set.loc[:, ("high", "1")] = [
            120,
            120,
            120,
            120,
        ]
        n.c.loads.dynamic.p_set.loc[:, ("low", "1")] = [80, 80, 80, 80]

        status, _ = n.optimize(multi_investment_periods=True)
        assert status == "ok"

        # Check that we have results for both scenarios
        assert "high" in n.c.generators.static.p_nom_opt.index.get_level_values(
            "scenario"
        )
        assert "low" in n.c.generators.static.p_nom_opt.index.get_level_values(
            "scenario"
        )

        # Capacities should be identical across scenarios in stochastic optimization
        high_cap = n.c.generators.static.p_nom_opt.loc[("high", "1")]
        low_cap = n.c.generators.static.p_nom_opt.loc[("low", "1")]
        assert high_cap == low_cap


def test_multiperiod_stochastic_tiny_build_year():
    """Test tiny multiperiod stochastic network with specific build year"""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)

        n = pypsa.Network(snapshots=range(2))
        n.investment_periods = [2020, 2030]
        n.add("Carrier", "elec")
        n.add("Bus", 1, carrier="elec")
        n.add(
            "Generator",
            1,
            bus=1,
            p_nom_extendable=True,
            capital_cost=10,
            build_year=2020,
            carrier="elec",
        )
        n.add("Load", 1, bus=1, p_set=100)

        n.set_scenarios({"scenario": 1.0})  # Single scenario
        n.c.loads.dynamic.p_set = pd.DataFrame(
            index=n.snapshots,
            columns=pd.MultiIndex.from_product(
                [n.scenarios, ["1"]], names=["scenario", "name"]
            ),
        )
        n.c.loads.dynamic.p_set.loc[:, ("scenario", "1")] = [100, 100, 100, 100]

        status, _ = n.optimize(multi_investment_periods=True)
        assert status == "ok"
        assert n.c.generators.static.p_nom_opt.loc[("scenario", "1")] == 100


def test_multiperiod_stochastic_tiny_infeasible():
    """Test infeasible multiperiod stochastic network"""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)

        n = pypsa.Network(snapshots=range(2))
        n.investment_periods = [2020, 2030]
        n.add("Carrier", "elec")
        n.add("Bus", 1, carrier="elec")
        n.add(
            "Generator",
            1,
            bus=1,
            p_nom_extendable=True,
            capital_cost=10,
            build_year=2030,
            carrier="elec",
        )
        n.add("Load", 1, bus=1, p_set=100)

        n.set_scenarios({"scenario": 1.0})
        n.c.loads.dynamic.p_set = pd.DataFrame(
            index=n.snapshots,
            columns=pd.MultiIndex.from_product(
                [n.scenarios, ["1"]], names=["scenario", "name"]
            ),
        )
        n.c.loads.dynamic.p_set.loc[:, ("scenario", "1")] = [100, 100, 100, 100]

        # This should fail because generator only available in 2030 but load exists in 2020
        with pytest.raises(ValueError):
            n.optimize(multi_investment_periods=True)


def test_multiperiod_stochastic_simple_network(n_multiperiod_stochastic):
    """Test simple multiperiod stochastic network optimization"""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)

        n = n_multiperiod_stochastic
        status, condition = n.optimize(multi_investment_periods=True)
        assert status == "ok"
        assert condition == "optimal"

        # Check that generators are only active in their respective periods
        for scenario in n.scenarios:
            # gen1-2050 should not be active in early periods
            assert (
                n.c.generators.dynamic.p.loc[
                    (2020, slice(None)), (scenario, "gen1-2050")
                ]
                == 0
            ).all()
            assert (
                n.c.generators.dynamic.p.loc[
                    (2030, slice(None)), (scenario, "gen1-2050")
                ]
                == 0
            ).all()
            assert (
                n.c.generators.dynamic.p.loc[
                    (2040, slice(None)), (scenario, "gen1-2050")
                ]
                == 0
            ).all()

            # gen1-2020 should not be active in 2050 (lifetime = 30)
            assert (
                n.c.generators.dynamic.p.loc[
                    (2050, slice(None)), (scenario, "gen1-2020")
                ]
                == 0
            ).all()

            # line-2050 should not be active in early periods
            assert (
                n.c.lines.dynamic.p0.loc[(2020, slice(None)), (scenario, "line-2050")]
                == 0
            ).all()
            assert (
                n.c.lines.dynamic.p0.loc[(2030, slice(None)), (scenario, "line-2050")]
                == 0
            ).all()
            assert (
                n.c.lines.dynamic.p0.loc[(2040, slice(None)), (scenario, "line-2050")]
                == 0
            ).all()


def test_multiperiod_stochastic_snapshot_subset(n_multiperiod_stochastic):
    """Test multiperiod stochastic network with snapshot subset"""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)

        n = n_multiperiod_stochastic
        status, condition = n.optimize(n.snapshots[:20], multi_investment_periods=True)
        assert status == "ok"
        assert condition == "optimal"

        # Same checks as above but with subset of snapshots
        for scenario in n.scenarios:
            assert (
                n.c.generators.dynamic.p.loc[
                    (2020, slice(None)), (scenario, "gen1-2050")
                ]
                == 0
            ).all()
            assert (
                n.c.generators.dynamic.p.loc[
                    (2030, slice(None)), (scenario, "gen1-2050")
                ]
                == 0
            ).all()
            assert (
                n.c.generators.dynamic.p.loc[
                    (2040, slice(None)), (scenario, "gen1-2050")
                ]
                == 0
            ).all()


def test_multiperiod_stochastic_storage_units_bug(n_multiperiod_sus_stochastic):
    """Test multiperiod stochastic network with storage units

    This test verifies that storage units work correctly in multiperiod + stochastic
    optimization. Previously, this combination caused a broadcasting error in the
    storage constraint creation due to dimension mismatches between:
    - (snapshots, scenarios, storage_units) from the mask
    - (scenarios, storage_units, snapshots) from the previous_soc_pp variable

    The fix ensures dimension order consistency by transposing the previous_soc_pp
    variable to match the expected dimension order when scenarios are present.
    """

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)

        n = n_multiperiod_sus_stochastic
        status, condition = n.optimize(multi_investment_periods=True)
        assert status == "ok"
        assert condition == "optimal"

        # Check that storage units have been built
        for scenario in n.scenarios:
            total_storage_cap = n.c.storage_units.static.p_nom_opt.loc[
                scenario, :
            ].sum()
            assert total_storage_cap > 0

            # Check that storage is only active in appropriate periods
            for period in n.investment_periods:
                period_snapshots = n.snapshots[
                    n.snapshots.get_level_values("period") == period
                ]

                # Storage units should respect their build years
                for storage_name in n.c.storage_units.static.loc[scenario, :].index:
                    storage = n.c.storage_units.static.loc[(scenario, storage_name)]
                    build_year = storage.build_year
                    lifetime = storage.lifetime

                    if build_year <= period <= build_year + lifetime:
                        # Storage should be available in this period
                        storage_dispatch = n.c.storage_units.dynamic.p.loc[
                            period_snapshots, (scenario, storage_name)
                        ]
                        # At least some periods should have non-zero dispatch (charging or discharging)
                        assert storage_dispatch.abs().sum() >= 0
                    else:
                        storage_dispatch = n.c.storage_units.dynamic.p.loc[
                            period_snapshots, (scenario, storage_name)
                        ]
                        assert (storage_dispatch == 0).all()


def test_multiperiod_stochastic_scenario_differences(n_multiperiod_stochastic):
    """Test that scenarios produce different results"""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)

        n = n_multiperiod_stochastic
        status, condition = n.optimize(multi_investment_periods=True)
        assert status == "ok"
        assert condition == "optimal"

        # Check that high and low scenarios have different generation patterns
        high_total_gen = (
            n.c.generators.dynamic.p.loc[:, ("high", slice(None))].sum().sum()
        )
        low_total_gen = (
            n.c.generators.dynamic.p.loc[:, ("low", slice(None))].sum().sum()
        )

        # High load scenario should have higher generation
        assert high_total_gen > low_total_gen

        # But capacities should be the same (stochastic optimization)
        for gen_name in n.c.generators.static.loc[
            ("high", slice(None)), :
        ].index.get_level_values("name"):
            high_cap = n.c.generators.static.p_nom_opt.loc[("high", gen_name)]
            low_cap = n.c.generators.static.p_nom_opt.loc[("low", gen_name)]
            assert high_cap == low_cap


def test_multiperiod_stochastic_coordinate_alignment():
    """Test coordinate alignment problem for multiperiod + stochastic + storage units.

    This test reproduces the exact scenario that caused the coordinate alignment bug:
    - Multiperiod optimization with investment periods
    - Scenarios (creating MultiIndex snapshots)
    - Storage units (triggering the problematic constraint)
    """
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)

        # Create minimal multiperiod network
        n = pypsa.Network(snapshots=range(4))
        n.investment_periods = [2020, 2030]

        n.add("Carrier", "elec")
        n.add("Bus", "bus1", carrier="elec")

        # Fill network with components
        n.add(
            "Generator",
            "gen1",
            bus="bus1",
            p_nom_extendable=True,
            capital_cost=100,
            marginal_cost=10,
            carrier="elec",
        )
        n.add("Load", "load1", bus="bus1", p_set=100)

        n.add("StorageUnit", "storage1", bus="bus1", p_nom=50, max_hours=4)

        # Set scenarios
        n.set_scenarios(["scenario_a", "scenario_b"])

        # The key test: model creation should succeed without coordinate alignment errors
        try:
            n.optimize.create_model()
            model_created = True
        except Exception as e:
            if "cannot align objects with join='exact'" in str(e):
                pytest.fail(
                    f"Coordinate alignment error in multiperiod stochastic storage: {e}"
                )
            else:
                raise e

        assert model_created, (
            "Multiperiod stochastic model with storage should be created successfully"
        )

        # Verify the problematic constraint was created correctly
        assert "StorageUnit-energy_balance" in n.model.constraints
        storage_constraint = n.model.constraints["StorageUnit-energy_balance"]

        # Should have all expected dimensions
        expected_dims = {"scenario", "name", "snapshot"}
        actual_dims = set(storage_constraint.dims)
        assert expected_dims.issubset(actual_dims), (
            f"Missing dimensions in storage constraint: {expected_dims - actual_dims}"
        )

        # Test that full optimization also works
        status, condition = n.optimize(multi_investment_periods=True)
        assert status == "ok", (
            f"Multiperiod stochastic optimization should succeed, got status: {status}"
        )


def test_storage_unit_energy_balance_scenario_sorting_bug():
    """Test that storage unit energy balance constraints have correct RHS values with scenarios.

    This test specifically catches the bug where scenario-based sorting caused storage units
    to get incorrect RHS values in their energy balance constraints. The bug occurred when:
    1. Network has scenarios (triggering sorting code)
    2. Storage units are present (creating energy balance constraints)
    3. Different storage units have different inflow values

    The sorting caused RHS values to be misaligned, so storage units would get RHS values
    from other storage units alphabetically earlier in the sorted list.
    """
    n = pypsa.Network(snapshots=pd.date_range("2020-01-01", periods=1, freq="h"))

    n.add("Carrier", "electricity")
    n.add("Bus", "bus1", carrier="electricity")

    # Add storage units with different names (to trigger alphabetical sorting)
    # and different inflow values (to detect when values get misaligned)
    n.add(
        "StorageUnit",
        "ZZ_storage",
        bus="bus1",
        p_nom=100,
        inflow=0,
        state_of_charge_initial=0,
        cyclic_state_of_charge=True,
    )
    n.add(
        "StorageUnit",
        "AA_storage",
        bus="bus1",
        p_nom=100,
        inflow=50,
        state_of_charge_initial=0,
        cyclic_state_of_charge=True,
    )

    n.add("Generator", "gen", bus="bus1", p_nom=200, marginal_cost=10)
    n.add("Load", "load", bus="bus1", p_set=100)

    # Test without scenarios first (should work correctly)
    n_no_scenarios = n.copy()
    n_no_scenarios.optimize.create_model()

    constraint_no_scenarios = n_no_scenarios.model.constraints[
        "StorageUnit-energy_balance"
    ]

    # Get RHS values for both storage units
    zz_rhs_no_scenarios = constraint_no_scenarios.loc[
        {"name": "ZZ_storage", "snapshot": n.snapshots[0]}
    ].rhs.item()
    aa_rhs_no_scenarios = constraint_no_scenarios.loc[
        {"name": "AA_storage", "snapshot": n.snapshots[0]}
    ].rhs.item()

    # AA_storage has inflow=50, so its RHS should be -50 (negative inflow)
    # ZZ_storage has inflow=0, so its RHS should be 0
    assert abs(aa_rhs_no_scenarios - (-50.0)) < 1e-6, (
        f"AA_storage should have RHS=-50, got {aa_rhs_no_scenarios}"
    )
    assert abs(zz_rhs_no_scenarios - 0.0) < 1e-6, (
        f"ZZ_storage should have RHS=0, got {zz_rhs_no_scenarios}"
    )

    # Test with scenarios (this is where the bug would occur)
    n_scenarios = n.copy()
    n_scenarios.set_scenarios({"scenario1": 1.0})
    n_scenarios.optimize.create_model()

    constraint_scenarios = n_scenarios.model.constraints["StorageUnit-energy_balance"]

    # Get RHS values for both storage units with scenarios
    zz_rhs_scenarios = constraint_scenarios.loc[
        {"name": "ZZ_storage", "snapshot": n.snapshots[0]}
    ].rhs.item()
    aa_rhs_scenarios = constraint_scenarios.loc[
        {"name": "AA_storage", "snapshot": n.snapshots[0]}
    ].rhs.item()

    # The key test: RHS values should be the same with and without scenarios
    # This would fail before the bug fix because sorting caused value misalignment
    assert abs(aa_rhs_scenarios - aa_rhs_no_scenarios) < 1e-6, (
        f"AA_storage RHS mismatch: without scenarios={aa_rhs_no_scenarios}, "
        f"with scenarios={aa_rhs_scenarios}. This indicates the scenario sorting bug!"
    )
    assert abs(zz_rhs_scenarios - zz_rhs_no_scenarios) < 1e-6, (
        f"ZZ_storage RHS mismatch: without scenarios={zz_rhs_no_scenarios}, "
        f"with scenarios={zz_rhs_scenarios}. This indicates the scenario sorting bug!"
    )

    # Additional verification: each storage unit should still have its expected RHS value
    assert abs(aa_rhs_scenarios - (-50.0)) < 1e-6, (
        f"AA_storage should have RHS=-50 with scenarios, got {aa_rhs_scenarios}"
    )
    assert abs(zz_rhs_scenarios - 0.0) < 1e-6, (
        f"ZZ_storage should have RHS=0 with scenarios, got {zz_rhs_scenarios}"
    )


@pytest.fixture
def n():
    n = pypsa.Network(snapshots=range(10))
    n.investment_periods = [2020, 2030, 2040, 2050]
    n.add("Carrier", "gencarrier")
    n.add("Bus", [1, 2])

    for i, period in enumerate(n.investment_periods):
        factor = (10 + i) / 10
        n.add(
            "Generator",
            [f"gen1-{period}", f"gen2-{period}"],
            bus=[1, 2],
            lifetime=30,
            build_year=period,
            capital_cost=[100 / factor, 100 * factor],
            marginal_cost=[i + 2, i + 1],
            p_nom_extendable=True,
            carrier="gencarrier",
        )

    for i, period in enumerate(n.investment_periods):
        n.add(
            "Line",
            f"line-{period}",
            bus0=1,
            bus1=2,
            length=1,
            build_year=period,
            lifetime=40,
            capital_cost=30 + i,
            x=0.0001,
            s_nom_extendable=True,
        )

    load = range(100, 100 + len(n.snapshots))
    load = pd.DataFrame({"load1": load, "load2": load}, index=n.snapshots)
    n.add(
        "Load",
        ["load1", "load2"],
        bus=[1, 2],
        p_set=load,
    )

    return n


def test_primary_energy_constraint_stochastic(ac_dc_stochastic):
    """Test primary energy constraint works correctly with stochastic networks."""
    n = ac_dc_stochastic
    assert ("low", "co2_limit") in n.c.global_constraints.static.index
    assert ("high", "co2_limit") in n.c.global_constraints.static.index
    n.optimize.create_model()
    assert "GlobalConstraint-co2_limit" in n.model.constraints
    assert "scenario" in n.model.constraints["GlobalConstraint-co2_limit"].dims


def test_operational_limit_constraint_stochastic():
    """Test operational limit constraints work correctly with stochastic networks."""
    n = pypsa.Network(snapshots=range(3))

    n.add("Carrier", "solar")
    n.add("Carrier", "gas")
    n.add("Carrier", "AC")

    n.add("Bus", "bus1", carrier="AC")

    n.add(
        "Generator",
        "solar",
        bus="bus1",
        p_nom=100,
        marginal_cost=0,
        carrier="solar",
    )
    n.add("Generator", "gas", bus="bus1", p_nom=200, marginal_cost=50, carrier="gas")
    n.add("Load", "load1", bus="bus1", p_set=[50, 100, 100])

    n.add(
        "GlobalConstraint",
        "solar_limit",
        type="operational_limit",
        sense="<=",
        constant=150,  # Total solar generation limit across all snapshots
        carrier_attribute="solar",
    )

    n.set_scenarios(["scenario1", "scenario2"])

    # Verify constraints exist in both scenarios
    assert ("scenario1", "solar_limit") in n.c.global_constraints.static.index
    assert ("scenario2", "solar_limit") in n.c.global_constraints.static.index

    # Create model to verify constraints are properly added
    n.optimize.create_model()
    assert "GlobalConstraint-solar_limit" in n.model.constraints
    assert "scenario" in n.model.constraints["GlobalConstraint-solar_limit"].dims


def test_max_growth_constraint_stochastic(n):
    """Test growth limit constraints work correctly with stochastic networks."""
    gen_carrier = n.c.generators.static.carrier.unique()[0]
    n.c.carriers.static.at[gen_carrier, "max_growth"] = 300
    n.set_scenarios({"scenario_1": 0.5, "scenario_2": 0.5})
    n.c.carriers.static.loc[("scenario_1", gen_carrier), "max_growth"] = 218
    kwargs = {"multi_investment_periods": True}
    status, cond = n.optimize(**kwargs)

    # In stochastic optimization, capacity decisions are shared across scenarios
    # The growth limit is constrained with the strictest bound
    capacity_per_period = (
        n.c.generators.static.xs("scenario_1")
        .p_nom_opt.groupby(n.c.generators.static.xs("scenario_1").build_year)
        .sum()
    )
    assert all(capacity_per_period <= 218), (
        f"Capacity per period exceeds limit: {capacity_per_period}"
    )
    assert "Carrier-growth_limit" in n.model.constraints


def test_max_relative_growth_constraint(n):
    """Test growth relative limit constraints work correctly with stochastic networks."""
    gen_carrier = n.c.generators.static.carrier.unique()[0]
    n.c.carriers.static.at[gen_carrier, "max_growth"] = 218
    n.c.carriers.static.at[gen_carrier, "max_relative_growth"] = 3
    n.set_scenarios({"scenario_1": 0.5, "scenario_2": 0.5})
    n.c.carriers.static.loc[("scenario_1", gen_carrier), "max_relative_growth"] = 1.5
    kwargs = {"multi_investment_periods": True}
    status, cond = n.optimize(**kwargs)
    built_per_period = (
        n.c.generators.static.xs("scenario_1")
        .p_nom_opt.groupby(n.c.generators.static.xs("scenario_1").build_year)
        .sum()
    )
    assert all(built_per_period - built_per_period.shift(fill_value=0) * 1.5 <= 218)


@pytest.mark.parametrize("assign", [True, False])
def test_assign_all_duals_stochastic(ac_dc_network, assign):
    """Test that all duals are written back to the network with stochastic scenarios."""
    n = ac_dc_network

    # Set up two scenarios
    n.set_scenarios({"scenario_1": 0.5, "scenario_2": 0.5})

    limit = 30_000

    m = n.optimize.create_model()

    transmission = m.variables["Link-p"]
    m.add_constraints(
        transmission.sum() <= limit, name="GlobalConstraint-generation_limit"
    )
    m.add_constraints(
        transmission.sum(dim="name") <= limit,
        name="GlobalConstraint-generation_limit_dynamic",
    )

    if assign:
        # TODO Add when custom constraints duals are written to extra custom constraint
        with pytest.raises(NotImplementedError):
            # This should raise because we are not assigning duals yet
            n.optimize.solve_model(assign_all_duals=assign)

        # assert ("generation_limit" in n.c.global_constraints.static.index) == assign
        # assert ("mu_generation_limit_dynamic" in n.c.global_constraints.dynamic) == assign
        # if "mu_upper" in n.c.generators.dynamic:
        #     assert not n.c.generators.dynamic.mu_upper.empty, (
        #         "Generator mu_upper should be assigned when assign_all_duals=True"
        #     )
        # if "mu_lower" in n.c.generators.dynamic:
        #     assert not n.c.generators.dynamic.mu_lower.empty, (
        #         "Generator mu_lower should be assigned when assign_all_duals=True"
        #     )

        # if "mu_upper" in n.c.links.dynamic:
        #     assert not n.c.links.dynamic.mu_upper.empty, (
        #         "Link mu_upper should be assigned when assign_all_duals=True"
        #     )
        # if "mu_lower" in n.c.links.dynamic:
        #     assert not n.c.links.dynamic.mu_lower.empty, (
        #         "Link mu_lower should be assigned when assign_all_duals=True"
        #     )

        # # Verify that stochastic dimensions are preserved in dual variables
        # if not n.c.buses.dynamic.marginal_price.empty:
        #     assert isinstance(n.c.buses.dynamic.marginal_price.columns, pd.MultiIndex), (
        #         "Marginal prices should have MultiIndex columns with scenarios"
        #     )
        #     scenarios_in_marginal_price = (
        #         n.c.buses.dynamic.marginal_price.columns.get_level_values("scenario").unique()
        #     )
        #     assert all(s in scenarios_in_marginal_price for s in n.scenarios), (
        #         "All scenarios should be present in marginal prices"
        #     )

    else:
        n.optimize.solve_model(assign_all_duals=assign)
        assert not n.c.buses.dynamic.marginal_price.empty, (
            "Marginal prices should always be assigned"
        )


def test_transmission_volume_expansion_limit_constraint_stochastic():
    """Test transmission volume expansion limit works correctly with scenarios."""
    n = pypsa.Network(snapshots=range(3))

    # Ensure carrier and buses/line exist and line is extendable
    n.add("Carrier", "AC")
    n.add("Bus", ["b1", "b2"], carrier="AC")
    n.add(
        "Line",
        "l1",
        bus0="b1",
        bus1="b2",
        length=1.0,
        x=0.0001,
        r=0.001,
        s_nom_extendable=True,
        carrier="AC",
    )

    n.add("Generator", "g", bus="b1", p_nom=100, marginal_cost=10, carrier="AC")
    n.add("Load", "load1", bus="b2", p_set=[50, 50, 50])

    n.add(
        "GlobalConstraint",
        "tx_vol",
        type="transmission_volume_expansion_limit",
        sense="<=",
        constant=1e6,
        carrier_attribute="AC",
    )

    # Scenarios
    n.set_scenarios(["scenario1", "scenario2"])

    # Verify constraint exists in both scenarios in the input table
    assert ("scenario1", "tx_vol") in n.c.global_constraints.static.index
    assert ("scenario2", "tx_vol") in n.c.global_constraints.static.index

    # Build model and verify a single constraint with scenario dimension
    n.optimize.create_model()
    assert "GlobalConstraint-tx_vol" in n.model.constraints
    assert "scenario" in n.model.constraints["GlobalConstraint-tx_vol"].dims
