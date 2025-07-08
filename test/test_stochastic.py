"""
Test stochastic functionality of PyPSA networks.
"""

from pathlib import Path

import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as equal
from xarray import DataArray

import pypsa
from pypsa.common import expand_series
from pypsa.components.common import as_components


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
    benchmark_path = Path(__file__).parent / "data" / "benchmark-sp"

    if not benchmark_path.exists():
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
        n.loads_t.p_set = pd.DataFrame(
            index=n.snapshots,
            columns=pd.MultiIndex.from_product(
                [n.scenarios, ["load1"]], names=["scenario", "name"]
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


def test_single_scenario():
    """
    Test that a network with a single scenario works correctly.

    Verifies that:
    - Single-scenario stochastic networks optimize successfully
    - Scenario indexing works correctly with one scenario
    - Solution is identical to a non-stochastic network with same data
    """
    import warnings

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
        capacity_det = n.generators.p_nom_opt.loc["gen1"]
        dispatch_det = n.generators_t.p.loc[:, "gen1"].sum()

        # Convert to single-scenario stochastic
        n.set_scenarios(["scenario"])

        # Set scenario-specific load data (same as deterministic)
        n.loads_t.p_set = pd.DataFrame(
            index=n.snapshots,
            columns=pd.MultiIndex.from_product(
                [n.scenarios, ["load1"]], names=["scenario", "name"]
            ),
        )
        n.loads_t.p_set.loc[:, ("scenario", "load1")] = pd.Series(
            [100.0, 120.0, 110.0], dtype=float
        )

        # Solve stochastic problem
        status_stoch, _ = n.optimize()
        assert status_stoch == "ok"

        # Verify structure
        assert len(n.scenarios) == 1
        assert "scenario" in n.scenarios
        assert "scenario" in n.generators_t.p.columns.get_level_values("scenario")

        # Compare solutions (should be identical)
        assert abs(n.objective - obj_det) < 1e-6

        stoch_capacity = n.generators.p_nom_opt.loc[("scenario", "gen1")]
        assert abs(stoch_capacity - capacity_det) < 1e-6

        stoch_dispatch = n.generators_t.p.loc[:, ("scenario", "gen1")].sum()
        assert abs(stoch_dispatch - dispatch_det) < 1e-6

        # Energy balance check
        gen_output = n.generators_t.p.loc[:, ("scenario", slice(None))].sum().sum()
        load_demand = n.loads_t.p_set.loc[:, ("scenario", "load1")].sum()
        assert abs(gen_output - load_demand) < 1e-1


def test_slack_bus_consistency_check():
    """
    Test that the consistency check correctly identifies when different slack buses
    are chosen across scenarios.
    """
    import warnings

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
        if n.buses.index.nlevels > 1:
            n.buses.loc[("scenario1", "bus1"), "control"] = "Slack"
            n.buses.loc[("scenario1", "bus2"), "control"] = "PQ"
            n.buses.loc[("scenario2", "bus1"), "control"] = "PQ"
            n.buses.loc[("scenario2", "bus2"), "control"] = (
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
    import warnings

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
        if n.buses.index.nlevels > 1:
            n.buses.loc[("scenario1", "bus1"), "control"] = "Slack"
            n.buses.loc[("scenario1", "bus2"), "control"] = "PQ"
            n.buses.loc[("scenario2", "bus1"), "control"] = "Slack"  # Same slack bus
            n.buses.loc[("scenario2", "bus2"), "control"] = "PQ"

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
    assert not n.stores.empty, "Test network should have stores"

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
    assert isinstance(n_stochastic.stores.index, pd.MultiIndex)
    assert n_stochastic.stores.index.names == ["scenario", "name"]

    # Verify optimization results exist
    assert not n_stochastic.stores_t.e.empty
    assert not n_stochastic.stores_t.p.empty


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
