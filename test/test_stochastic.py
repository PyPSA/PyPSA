"""
Test stochastic functionality of PyPSA networks.
"""

import os

import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as equal

import pypsa


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
    assert abs(n.scenarios.sum() - 1.0) < 1e-10

    # Check data shape for each scenario
    for scenario in n.scenarios.index:
        assert n.get_switchable_as_dense("Load", "p_set").loc[:, scenario].shape[0] == 5

    # Check string representation contains scenario information
    assert "Scenarios:" in repr(n)


def test_component_functions(ac_dc_meshed_stoch):
    assert isinstance(ac_dc_meshed_stoch.branches(), pd.DataFrame)
    assert isinstance(ac_dc_meshed_stoch.passive_branches(), pd.DataFrame)
    assert isinstance(ac_dc_meshed_stoch.controllable_branches(), pd.DataFrame)


def test_calculate_dependent_values(ac_dc_meshed_stoch: pypsa.Network):
    """
    Test the calculation of dependent values in a stochastic network.
    This includes checking that the function runs without errors and that
    the expected attributes are present in the network object.
    """
    n = ac_dc_meshed_stoch
    n.calculate_dependent_values()
    assert n.lines.x_pu_eff.notnull().all()


def test_cycles(ac_dc_meshed_stoch: pypsa.Network):
    n = ac_dc_meshed_stoch
    C = n.cycles()

    assert isinstance(C, pd.DataFrame)
    assert C.notnull().all().all()  # Check for NaN values

    # repeat with apply weights
    n.calculate_dependent_values()
    C = n.cycles(apply_weights=True)
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

    assert hasattr(n, "model") and n.model is not None

    # Test operational variable Generator-p has scenario dimension
    assert n.model.variables["Generator-p"].dims == (
        "scenario",
        "component",
        "snapshot",
    )

    # Test that Generator-p_nom does not have scenario dimension (investment variable)
    assert n.model.variables["Generator-p_nom"].dims == ("component",)

    # Test operational constraints have scenario dimension

    # Generator-ext-p_nom-lower constraint should have (component, scenario) dimensions
    assert {
        d
        for d in n.model.constraints["Generator-ext-p_nom-lower"].sizes.keys()
        if d != "_term"
    } == {"component", "scenario"}

    # Generator-ext-p-lower constraint should have (scenario, component, snapshot) dimensions
    assert {
        d
        for d in n.model.constraints["Generator-ext-p-lower"].sizes.keys()
        if d != "_term"
    } == {"scenario", "component", "snapshot"}

    # Generator-ext-p-upper constraint should have (scenario, component, snapshot) dimensions
    assert {
        d
        for d in n.model.constraints["Generator-ext-p-upper"].sizes.keys()
        if d != "_term"
    } == {"scenario", "component", "snapshot"}

    # Bus-nodal_balance constraint should have (component, scenario, snapshot) dimensions
    assert {
        d for d in n.model.constraints["Bus-nodal_balance"].sizes.keys() if d != "_term"
    } == {"component", "scenario", "snapshot"}


def test_statistics(ac_dc_meshed_stoch_r):
    """
    Test the statistics of a stochastic network.
    """
    n = ac_dc_meshed_stoch_r
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

    df = n.statistics.supply(aggregate_time=False)
    assert isinstance(df, pd.DataFrame)
    assert isinstance(df.index, pd.MultiIndex)
    assert "scenario" in df.index.names
    assert not df.empty


def test_statistics_plot(ac_dc_meshed_stoch_r):
    """
    Test the statistics plot of a stochastic network.
    """
    n = ac_dc_meshed_stoch_r
    s = n.statistics
    s.installed_capacity.plot.bar()


def test_optimization_simple(ac_dc_meshed_stoch):
    """
    Simple test case for the optimization of a stochastic network.
    """
    n = ac_dc_meshed_stoch
    n.optimize.create_model()
    status, _ = n.optimize(solver_name="highs")
    assert status == "ok"


def test_optimization_advanced(storage_hvdc_network):
    """
    Advanced test case for the optimization of a stochastic network.
    """
    n = storage_hvdc_network
    n.set_scenarios({"low": 0.5, "high": 0.5})
    status, _ = n.optimize(solver_name="highs")
    assert status == "ok"


def test_solved_network_simple(stochastic_benchmark_network):
    """
    Solve the stochastic problem and compare results with benchmark data.
    Simple test case with a single bus and multiple generators.
    """
    # Load the benchmark results
    benchmark_path = os.path.join(os.path.dirname(__file__), "data", "benchmark-sp")

    if not os.path.exists(benchmark_path):
        pytest.skip("Benchmark data not available")

    n_r = pypsa.Network(benchmark_path)

    # Create a new network for the stochastic model
    n = stochastic_benchmark_network

    # GAS_PRICES = {"low": 40, "med": 70, "high": 100}
    n.generators.loc[("medium", "gas"), "marginal_cost"] = (
        70 / n.generators.loc[("medium", "gas"), "efficiency"]
    )
    n.generators.loc[("high", "gas"), "marginal_cost"] = (
        100 / n.generators.loc[("high", "gas"), "efficiency"]
    )

    status, _ = n.optimize(solver_name="highs")
    assert status == "ok"

    # Compare generator capacities (these are the main result of stochastic planning)
    equal(
        n.generators.p_nom_opt.loc["low", :],
        n_r.generators.p_nom_opt,
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

    Key insight: All components must be added BEFORE calling set_scenarios()
    for the scenario indexing to work correctly in multiperiod networks.
    """
    import warnings

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

        # Add all components BEFORE setting scenarios (this is the key fix!)
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

        # Now set scenarios AFTER all components are added
        n.set_scenarios({"high": 0.5, "low": 0.5})

        # Set scenario-specific loads for multiperiod (6 snapshots total: 2 periods × 3 timesteps)
        n.loads_t.p_set = pd.DataFrame(
            index=n.snapshots,
            columns=pd.MultiIndex.from_product(
                [n.scenarios.index, ["load1"]], names=["scenario", "component"]
            ),
        )

        load_high = [120, 144, 132] * len(n.investment_periods)
        load_low = [80, 96, 88] * len(n.investment_periods)

        n.loads_t.p_set.loc[:, ("high", "load1")] = load_high
        n.loads_t.p_set.loc[:, ("low", "load1")] = load_low

        # This should now work with both multiperiod and stochastic features!
        status, condition = n.optimize(multi_investment_periods=True)
        assert status == "ok"

        # Verify we have results for both scenarios and investment periods
        assert "high" in n.generators_t.p.columns.get_level_values("scenario")
        assert "low" in n.generators_t.p.columns.get_level_values("scenario")

        # Check basic energy balance for each scenario
        for scenario in ["high", "low"]:
            gen_output = n.generators_t.p.loc[:, (scenario, slice(None))].sum().sum()
            load_demand = n.loads_t.p_set.loc[:, (scenario, "load1")].sum()
            # Generation should equal load
            assert abs(gen_output - load_demand) < 1e-1

        # Verify that high scenario has higher generation than low scenario
        gen_high = n.generators_t.p.loc[:, ("high", slice(None))].sum().sum()
        gen_low = n.generators_t.p.loc[:, ("low", slice(None))].sum().sum()
        assert gen_high > gen_low

        # Test multiperiod-specific functionality
        p_nom_opt = n.generators.p_nom_opt
        assert (
            len(p_nom_opt) == 4
        )  # Should have optimal capacities for both generators × both scenarios

        # Verify we have generators for both scenarios
        scenarios_in_gens = p_nom_opt.index.get_level_values("scenario").unique()
        assert "high" in scenarios_in_gens
        assert "low" in scenarios_in_gens


def test_objective_scaling_with_weights():
    """
    Placeholder: verify that the objective function is correctly scaled with scenario weights.
    Check that generation costs and investment costs are properly weighted by scenario probabilities.
    """
    pass


def test_weight_sensitivity():
    """
    Placeholder: test that changing scenario weights affects the solution accordingly.
    Compare solutions with different probability distributions across the same scenarios.
    """
    pass
