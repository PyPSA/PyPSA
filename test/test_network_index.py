import pandas as pd
import pytest


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
