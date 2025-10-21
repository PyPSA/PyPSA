# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import pandas as pd
import pytest

import pypsa


class TestNetworkScenarioIndex:
    def test_empty_input(self, ac_dc_network):
        """Test that an error is raised when no scenarios are provided."""
        n = ac_dc_network
        with pytest.raises(ValueError, match="You must pass either"):
            n.set_scenarios()

    def test_both_kwargs_and_scenarios(self, ac_dc_network):
        """Test that an error is raised when both kwargs and scenarios are provided."""
        n = ac_dc_network
        with pytest.raises(ValueError, match="You can pass scenarios either via"):
            n.set_scenarios(scenarios=["a", "b"], weights=[1, 2], scenario1=1)

    def test_dict_with_weights(self, ac_dc_network):
        """Test that an error is raised when a dict is provided with weights."""
        n = ac_dc_network
        with pytest.raises(ValueError, match="You can pass scenarios either via"):
            n.set_scenarios(scenarios={"a": 1, "b": 2}, weights=[1, 2])

    def test_series_with_weights(self, ac_dc_network):
        """Test that an error is raised when a Series is provided with weights."""
        n = ac_dc_network
        with pytest.raises(ValueError, match="You can pass scenarios either via"):
            n.set_scenarios(scenarios=pd.Series({"a": 1, "b": 2}), weights=[1, 2])

    def test_mismatched_weights_length(self, ac_dc_network):
        """Test that an error is raised when weights length doesn't match scenarios length."""
        n = ac_dc_network
        with pytest.raises(ValueError, match="You can pass scenarios either via"):
            n.set_scenarios(scenarios=["a", "b", "c"], weights=[1, 2])

    def test_dict_scenarios(self, ac_dc_network):
        """Test setting scenarios from a dict."""
        n = ac_dc_network
        n.set_scenarios(scenarios={"scenario1": 0.3, "scenario2": 0.7})
        expected_index = pd.Index(["scenario1", "scenario2"], name="scenario")
        expected_weights = pd.DataFrame({"weight": [0.3, 0.7]}, index=expected_index)
        pd.testing.assert_index_equal(n.scenarios, expected_index)
        pd.testing.assert_frame_equal(n.scenario_weightings, expected_weights)

    def test_series_scenarios(self, ac_dc_network):
        """Test setting scenarios from a Series."""
        n = ac_dc_network
        series = pd.Series({"scenario1": 0.3, "scenario2": 0.7})
        n.set_scenarios(scenarios=series)
        expected_index = pd.Index(["scenario1", "scenario2"], name="scenario")
        expected_weights = pd.DataFrame({"weight": [0.3, 0.7]}, index=expected_index)
        pd.testing.assert_index_equal(n.scenarios, expected_index)
        pd.testing.assert_frame_equal(n.scenario_weightings, expected_weights)

    def test_sequence_scenarios(self, ac_dc_network):
        """Test setting scenarios from a sequence with weights."""
        n = ac_dc_network
        n.set_scenarios(scenarios=["scenario1", "scenario2"])
        expected_index = pd.Index(["scenario1", "scenario2"], name="scenario")
        expected_weights = pd.DataFrame({"weight": [0.5, 0.5]}, index=expected_index)
        pd.testing.assert_index_equal(n.scenarios, expected_index)
        pd.testing.assert_frame_equal(n.scenario_weightings, expected_weights)

    def test_kwargs_scenarios(self, ac_dc_network):
        """Test setting scenarios from keyword arguments."""
        n = ac_dc_network
        n.set_scenarios(scenario1=0.3, scenario2=0.7)
        expected_index = pd.Index(["scenario1", "scenario2"], name="scenario")
        expected_weights = pd.DataFrame({"weight": [0.3, 0.7]}, index=expected_index)
        pd.testing.assert_index_equal(n.scenarios, expected_index)
        pd.testing.assert_frame_equal(n.scenario_weightings, expected_weights)

    def test_series_name_preserved(self, ac_dc_network):
        """Test that the scenario_weightings column name is set to 'weight'."""
        n = ac_dc_network
        series = pd.Series({"scenario1": 0.3, "scenario2": 0.7}, name="original_name")
        n.set_scenarios(scenarios=series)
        assert n.scenario_weightings.columns[0] == "weight"
        assert n.scenarios.name == "scenario"

    def test_sequence_without_weights(self, ac_dc_network):
        """Test setting scenarios from a sequence without weights."""
        n = ac_dc_network
        n.set_scenarios(scenarios=["scenario1", "scenario2", "scenario3"])

        # When no weights are provided, equal weights (1/n) should be assigned
        expected_index = pd.Index(
            ["scenario1", "scenario2", "scenario3"], name="scenario"
        )
        expected_weights = pd.DataFrame(
            {"weight": [1 / 3, 1 / 3, 1 / 3]}, index=expected_index
        )

        pd.testing.assert_index_equal(n.scenarios, expected_index)
        pd.testing.assert_frame_equal(n.scenario_weightings, expected_weights)

    def test_weights_must_sum_to_one(self, ac_dc_network):
        """Test that an error is raised when scenario weights don't sum to 1."""
        n = ac_dc_network
        # Create a series with weights that don't sum to 1
        scenarios = pd.Series({"scenario1": 0.3, "scenario2": 0.4})  # Sum = 0.7

        with pytest.raises(
            ValueError, match="The sum of the weights in `scenarios` must be equal to 1"
        ):
            n.set_scenarios(scenarios=scenarios)


def test_get_scenario():
    n = pypsa.examples.ac_dc_meshed()
    n.set_scenarios(high=0.1, low=0.9)
    n.c.generators.static.loc[("high", "Manchester Wind"), "p_nom"] = 200

    n_high = n.get_scenario("high")
    n_low = n.get_scenario("low")

    assert n_high.name == "AC-DC-Meshed - Scenario 'high'"

    n_high.name = n.name
    n_low.name = n.name

    ac_dc_meshed = pypsa.examples.ac_dc_meshed()
    assert n_low.equals(ac_dc_meshed, log_mode="strict")
    assert n_low is not ac_dc_meshed

    ac_dc_meshed.c.generators.static.loc[("Manchester Wind"), "p_nom"] = 200
    assert n_high.equals(ac_dc_meshed, log_mode="strict")
    assert n_high is not ac_dc_meshed

    # Test ValueError when network has no scenarios
    n_no_scenarios = pypsa.examples.ac_dc_meshed()
    with pytest.raises(
        ValueError,
        match="This method can only be used on a stochastic network with scenarios",
    ):
        n_no_scenarios.get_scenario("high")

    # Test KeyError when scenario doesn't exist
    with pytest.raises(
        KeyError, match="Scenario 'nonexistent' not found in network scenarios"
    ):
        n.get_scenario("nonexistent")


def test_get_scenario_empty_components_bug_1402():
    """
    Test that get_scenario() and __getitem__ properly reset MultiIndex for empty components.

    See https://github.com/PyPSA/PyPSA/issues/1402:
    When extracting a scenario from a stochastic network, empty components (like Links)
    should have their MultiIndex reset to a simple Index, not retain the scenario level.
    This ensures the extracted network can be optimized without MultiIndex conflicts.
    """
    n = pypsa.Network()
    n.set_snapshots(range(3))
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=100)
    n.add(
        "Generator",
        "gen",
        bus="bus",
        p_nom_extendable=True,
        capital_cost=1000,
        marginal_cost=10,
    )

    n.set_scenarios({"low": 0.5, "high": 0.5})

    # Verify Links component is empty but has MultiIndex after set_scenarios
    assert n.c.links.static.empty
    assert isinstance(n.c.links.static.index, pd.MultiIndex)
    assert n.c.links.static.index.names == ["scenario", "name"]

    # Test get_scenario() method
    n_low = n.get_scenario("low")

    # After extraction, Links should have simple Index, not MultiIndex
    assert n_low.c.links.static.empty
    assert not isinstance(n_low.c.links.static.index, pd.MultiIndex)
    assert n_low.c.links.static.index.name == "name"

    # Verify network can be optimized without errors
    status, _ = n_low.optimize(solver_name="highs", log_to_console=False)
    assert status == "ok"

    # Test __getitem__ method (n['scenario'])
    n_high = n["high"]

    # After extraction via __getitem__, Links should also have simple Index
    assert n_high.c.links.static.empty
    assert not isinstance(n_high.c.links.static.index, pd.MultiIndex)
    assert n_high.c.links.static.index.name == "name"

    # Verify this network can also be optimized
    status, _ = n_high.optimize(solver_name="highs", log_to_console=False)
    assert status == "ok"


def test_get_network_from_collection():
    n = pypsa.examples.ac_dc_meshed()
    n2 = n.copy()
    n2.name = "AC-DC-Meshed Copy"
    n2.c.generators.static.loc[("Manchester Wind"), "p_nom"] = 200

    nc = pypsa.NetworkCollection([n, n2])

    nc1 = nc.get_network("AC-DC-Meshed")
    nc2 = nc.get_network("AC-DC-Meshed Copy")

    # Make sure a collection only holds references to the original networks
    assert nc1 is n
    assert nc2 is n2

    # Test KeyError when collection doesn't exist
    with pytest.raises(
        KeyError, match="Collection 'nonexistent' not found in network collection"
    ):
        nc.get_network("nonexistent")

    # Stochastic network in collection
    n2.set_scenarios(high=0.1, low=0.9)
    n2.c.generators.static.loc[("high", "Manchester Wind"), "p_nom"] = 300
    nc = pypsa.NetworkCollection([n, n2])

    nc2 = nc.get_network("AC-DC-Meshed Copy")

    assert nc2 is n2

    with pytest.raises(
        KeyError, match="Collection 'nonexistent' not found in network collection"
    ):
        nc.get_network("nonexistent")

    # Test ValueError when network is not a collection
    n_no_collection = pypsa.examples.ac_dc_meshed()
    with pytest.raises(
        ValueError,
        match="This method can only be used on a NetworkCollection",
    ):
        n_no_collection.get_network("high")


def test_slice_network():
    n = pypsa.examples.ac_dc_meshed()

    # Test slicing by buses - single bus
    n_single = n.slice_network(buses="Manchester")
    assert len(n_single.buses) == 1
    assert "Manchester" in n_single.c.buses.static.index
    # Check connected components are included
    assert len(n_single.c.generators) > 0
    assert all(n_single.c.generators.static.bus == "Manchester")

    # Test slicing by buses - list of buses
    bus_list = ["Manchester", "Frankfurt"]
    n_subset = n.slice_network(buses=bus_list)
    assert len(n_subset.buses) == 2
    assert set(n_subset.c.buses.static.index) == set(bus_list)
    # Check lines between selected buses
    lines_between = n.c.lines.static[
        n.c.lines.static.bus0.isin(bus_list) & n.c.lines.static.bus1.isin(bus_list)
    ]
    assert len(n_subset.c.lines.static) == len(lines_between)

    # Test slicing by snapshots using slice object
    n_snap = n.slice_network(snapshots=slice(None, 2))
    assert len(n_snap.snapshots) == 2
    assert all(n_snap.snapshots == n.snapshots[:2])
    assert len(n_snap.buses) == len(n.buses)  # All buses preserved

    # Test slicing by both buses and snapshots
    n_both = n.slice_network(buses="Manchester", snapshots=0)
    assert len(n_both.buses) == 1
    assert len(n_both.snapshots) == 1
    assert n_both.snapshots[0] == n.snapshots[0]

    # Test with boolean mask for buses
    bus_mask = n.c.buses.static.v_nom > 300
    n_hv = n.slice_network(buses=bus_mask)
    assert all(n_hv.c.buses.static.v_nom > 300)

    # Test error when neither buses nor snapshots provided
    with pytest.raises(
        ValueError, match="Either `buses` or `snapshots` must be provided"
    ):
        n.slice_network()

    # Test that dynamic data is properly sliced
    n.generators_t.p_set.loc[:, "Manchester Wind"] = range(len(n.snapshots))
    n_snap2 = n.slice_network(snapshots=slice(1, 3))
    assert all(n_snap2.generators_t.p_set.loc[:, "Manchester Wind"] == [1, 2])

    # Test that network attributes are preserved
    n.name = "Test Network"
    n_sliced = n.slice_network(buses="Manchester")
    assert n_sliced.name == "Test Network"

    # Test snapshot weightings are properly sliced with list of indices
    n_snap3 = n.slice_network(snapshots=[0, 2])
    expected_weights = n.snapshot_weightings.iloc[[0, 2]]
    pd.testing.assert_frame_equal(n_snap3.snapshot_weightings, expected_weights)


def test_getitem_index_methods():
    """Test that __getitem__ mirrors all three index methods."""
    # Test network slicing via __getitem__
    n = pypsa.examples.ac_dc_meshed()

    # Test slicing by single bus (mirrors slice_network)
    n_manchester = n["Manchester"]
    n_manchester_slice = n.slice_network(buses="Manchester")
    assert n_manchester.equals(n_manchester_slice, log_mode="strict")

    # Test slicing by multiple buses
    bus_list = ["Manchester", "Frankfurt"]
    n_multi = n[bus_list]
    n_multi_slice = n.slice_network(buses=bus_list)
    assert n_multi.equals(n_multi_slice, log_mode="strict")

    # Slicing by snapshots is currently not supported in __getitem__
    # as it would clash with the other two methods.

    # Test NetworkCollection __getitem__ (mirrors get_network)
    n2 = n.copy()
    n2.name = "Test Network Copy"
    nc = pypsa.NetworkCollection([n, n2])

    nc_getitem = nc["AC-DC-Meshed"]
    nc_get = nc.get_network("AC-DC-Meshed")
    assert nc_getitem is nc_get  # Should return same reference

    # Test stochastic network __getitem__ (mirrors get_scenario)
    n_stoch = pypsa.examples.ac_dc_meshed()
    n_stoch.set_scenarios(high=0.3, low=0.7)
    n_stoch.c.generators.static.loc[("high", "Manchester Wind"), "p_nom"] = 200

    n_high_getitem = n_stoch["high"]
    n_high_get = n_stoch.get_scenario("high")
    assert n_high_getitem.equals(n_high_get, log_mode="strict")

    # Test error cases
    with pytest.raises(KeyError):
        nc["nonexistent"]

    with pytest.raises(KeyError):
        n_stoch["nonexistent"]

    # Test that regular network raises KeyError for non-existent bus
    with pytest.raises(KeyError):
        n["nonexistent_bus"]

    # Test deprecated tuple slicing raises warning and error
    with pytest.warns(
        DeprecationWarning, match="Slicing by \\(buses, snapshots\\) tuples"
    ):
        with pytest.raises(NotImplementedError, match="Tuple slicing is deprecated"):
            n[("Manchester", slice(0, 2))]
