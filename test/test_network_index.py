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
        expected = pd.Series({"scenario1": 0.3, "scenario2": 0.7}, name="scenario")
        pd.testing.assert_series_equal(n.scenarios, expected)

    def test_series_scenarios(self, n):
        """Test setting scenarios from a Series."""
        series = pd.Series({"scenario1": 0.3, "scenario2": 0.7})
        n.set_scenarios(scenarios=series)
        expected = pd.Series({"scenario1": 0.3, "scenario2": 0.7}, name="scenario")
        pd.testing.assert_series_equal(n.scenarios, expected)

    def test_sequence_scenarios(self, n):
        """Test setting scenarios from a sequence with weights."""
        n.set_scenarios(scenarios=["scenario1", "scenario2"])
        expected = pd.Series(
            [0.5, 0.5], index=["scenario1", "scenario2"], name="scenario"
        )
        pd.testing.assert_series_equal(n.scenarios, expected)

    def test_kwargs_scenarios(self, n):
        """Test setting scenarios from keyword arguments."""
        n.set_scenarios(scenario1=0.3, scenario2=0.7)
        expected = pd.Series({"scenario1": 0.3, "scenario2": 0.7}, name="scenario")
        pd.testing.assert_series_equal(n.scenarios, expected)

    def test_series_name_preserved(self, n):
        """Test that the series name is set to 'scenario'."""
        series = pd.Series({"scenario1": 0.3, "scenario2": 0.7}, name="original_name")
        n.set_scenarios(scenarios=series)
        assert n.scenarios.name == "scenario"

    def test_sequence_without_weights(self, n):
        """Test setting scenarios from a sequence without weights."""
        n.set_scenarios(scenarios=["scenario1", "scenario2", "scenario3"])

        # When no weights are provided, equal weights (1/n) should be assigned
        expected = pd.Series(
            [1 / 3, 1 / 3, 1 / 3],
            index=["scenario1", "scenario2", "scenario3"],
            name="scenario",
        )

        pd.testing.assert_series_equal(n.scenarios, expected)

    def test_weights_must_sum_to_one(self, n):
        """Test that an error is raised when scenario weights don't sum to 1."""
        # Create a series with weights that don't sum to 1
        scenarios = pd.Series({"scenario1": 0.3, "scenario2": 0.4})  # Sum = 0.7

        with pytest.raises(
            ValueError, match="The sum of the weights in `scenarios` must be equal to 1"
        ):
            n.set_scenarios(scenarios=scenarios)
