import pandas as pd
import pytest


class TestNetworkScenarioIndex:
    def test_empty_input(self, n):
        """Test that an error is raised when no scenarios are provided."""
        with pytest.raises(ValueError, match="You must pass either"):
            n.set_scenarios()

    def test_both_kwargs_and_scenarios(self, n):
        """Test that an error is raised when both kwargs and scenarios are provided."""
        with pytest.raises(ValueError, match="You can pass scenarios either via"):
            n.set_scenarios(scenarios=["a", "b"], weights=[1, 2], scenario1=1)

    def test_dict_with_weights(self, n):
        """Test that an error is raised when a dict is provided with weights."""
        with pytest.raises(ValueError, match="You can pass scenarios either via"):
            n.set_scenarios(scenarios={"a": 1, "b": 2}, weights=[1, 2])

    def test_series_with_weights(self, n):
        """Test that an error is raised when a Series is provided with weights."""
        with pytest.raises(ValueError, match="You can pass scenarios either via"):
            n.set_scenarios(scenarios=pd.Series({"a": 1, "b": 2}), weights=[1, 2])

    def test_mismatched_weights_length(self, n):
        """Test that an error is raised when weights length doesn't match scenarios length."""
        with pytest.raises(ValueError, match="You can pass scenarios either via"):
            n.set_scenarios(scenarios=["a", "b", "c"], weights=[1, 2])

    def test_dict_scenarios(self, n):
        """Test setting scenarios from a dict."""
        n.set_scenarios(scenarios={"scenario1": 0.3, "scenario2": 0.7})
        expected_index = pd.Index(["scenario1", "scenario2"], name="scenario")
        expected_weights = pd.DataFrame({"weight": [0.3, 0.7]}, index=expected_index)
        pd.testing.assert_index_equal(n.scenarios, expected_index)
        pd.testing.assert_frame_equal(n.scenario_weightings, expected_weights)

    def test_series_scenarios(self, n):
        """Test setting scenarios from a Series."""
        series = pd.Series({"scenario1": 0.3, "scenario2": 0.7})
        n.set_scenarios(scenarios=series)
        expected_index = pd.Index(["scenario1", "scenario2"], name="scenario")
        expected_weights = pd.DataFrame({"weight": [0.3, 0.7]}, index=expected_index)
        pd.testing.assert_index_equal(n.scenarios, expected_index)
        pd.testing.assert_frame_equal(n.scenario_weightings, expected_weights)

    def test_sequence_scenarios(self, n):
        """Test setting scenarios from a sequence with weights."""
        n.set_scenarios(scenarios=["scenario1", "scenario2"])
        expected_index = pd.Index(["scenario1", "scenario2"], name="scenario")
        expected_weights = pd.DataFrame({"weight": [0.5, 0.5]}, index=expected_index)
        pd.testing.assert_index_equal(n.scenarios, expected_index)
        pd.testing.assert_frame_equal(n.scenario_weightings, expected_weights)

    def test_kwargs_scenarios(self, n):
        """Test setting scenarios from keyword arguments."""
        n.set_scenarios(scenario1=0.3, scenario2=0.7)
        expected_index = pd.Index(["scenario1", "scenario2"], name="scenario")
        expected_weights = pd.DataFrame({"weight": [0.3, 0.7]}, index=expected_index)
        pd.testing.assert_index_equal(n.scenarios, expected_index)
        pd.testing.assert_frame_equal(n.scenario_weightings, expected_weights)

    def test_series_name_preserved(self, n):
        """Test that the scenario_weightings column name is set to 'weight'."""
        series = pd.Series({"scenario1": 0.3, "scenario2": 0.7}, name="original_name")
        n.set_scenarios(scenarios=series)
        assert n.scenario_weightings.columns[0] == "weight"
        assert n.scenarios.name == "scenario"

    def test_sequence_without_weights(self, n):
        """Test setting scenarios from a sequence without weights."""
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

    def test_weights_must_sum_to_one(self, n):
        """Test that an error is raised when scenario weights don't sum to 1."""
        # Create a series with weights that don't sum to 1
        scenarios = pd.Series({"scenario1": 0.3, "scenario2": 0.4})  # Sum = 0.7

        with pytest.raises(
            ValueError, match="The sum of the weights in `scenarios` must be equal to 1"
        ):
            n.set_scenarios(scenarios=scenarios)

    def test_standard_types_not_broadcasted_by_scenario(self, n):
        """Test that standard types (LineType, TransformerType) are not broadcasted by scenario."""

        # Get initial line types index (should not have scenarios)
        initial_line_types_index = n.line_types.index.copy()
        initial_bus_count = len(n.buses)

        # Set scenarios
        n.set_scenarios(scenarios=["scenario1", "scenario2"])

        # After setting scenarios, line types should still have the original index
        pd.testing.assert_index_equal(n.line_types.index, initial_line_types_index)

        # But other components like buses should be broadcasted
        assert len(n.buses) == initial_bus_count * 2
        # Check that buses have scenario index
        assert n.buses.index.names[0] == "scenario"
