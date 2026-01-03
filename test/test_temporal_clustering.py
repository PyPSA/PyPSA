# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
import pytest

import pypsa
from pypsa.clustering.temporal import (
    TemporalClustering,
    downsample,
    from_snapshot_map,
    resample,
)


@pytest.fixture
def simple_network():
    """Create a simple network with hourly resolution for one week."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2020-01-01", periods=168, freq="h"))

    n.add("Bus", "bus0")
    n.add("Bus", "bus1")

    n.add(
        "Generator",
        "gen0",
        bus="bus0",
        p_nom=100,
        marginal_cost=10,
    )

    n.add(
        "Generator",
        "gen1",
        bus="bus1",
        p_nom=50,
        marginal_cost=20,
        p_max_pu=np.random.default_rng(42).uniform(0.5, 1.0, 168),
    )

    n.add("Load", "load0", bus="bus0", p_set=50)
    n.add("Load", "load1", bus="bus1", p_set=30)

    n.add("Line", "line01", bus0="bus0", bus1="bus1", s_nom=100, x=0.1, r=0.01)

    return n


@pytest.fixture
def yearly_network():
    """Create a network with full year hourly resolution."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2020-01-01", periods=8760, freq="h"))

    n.add("Bus", "bus0")
    n.add(
        "Generator",
        "gen0",
        bus="bus0",
        p_nom=100,
        p_max_pu=np.random.default_rng(42).uniform(0.3, 1.0, 8760),
    )
    n.add("Load", "load0", bus="bus0", p_set=50)

    return n


@pytest.fixture
def multiperiod_network():
    """Create a network with investment periods."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2020-01-01", periods=24, freq="h"))
    n.set_investment_periods([2020, 2030])

    n.add("Bus", "bus0")
    n.add("Generator", "gen0", bus="bus0", p_nom=100)
    n.add("Load", "load0", bus="bus0", p_set=50)

    return n


class TestResample:
    def test_resample_3h(self, simple_network):
        n = simple_network
        result = resample(n, "3h")

        assert isinstance(result, TemporalClustering)
        assert len(result.n.snapshots) == 168 // 3
        assert np.isclose(
            result.n.snapshot_weightings["objective"].sum(),
            n.snapshot_weightings["objective"].sum(),
        )

    def test_resample_preserves_total_hours(self, simple_network):
        n = simple_network
        original_hours = n.snapshot_weightings["objective"].sum()

        result = resample(n, "6h")
        new_hours = result.n.snapshot_weightings["objective"].sum()

        assert np.isclose(original_hours, new_hours)

    def test_resample_daily(self, simple_network):
        n = simple_network
        result = resample(n, "24h")

        assert len(result.n.snapshots) == 7

    def test_resample_time_series_aggregated(self, simple_network):
        n = simple_network
        result = resample(n, "3h")

        assert "p_max_pu" in result.n.c.generators.dynamic
        assert len(result.n.c.generators.dynamic["p_max_pu"]) == len(result.n.snapshots)

    def test_resample_snapshot_map(self, simple_network):
        n = simple_network
        result = resample(n, "3h")

        assert len(result.snapshot_map) == len(n.snapshots)
        assert set(result.snapshot_map.values).issubset(set(result.n.snapshots))

    def test_resample_accessor(self, simple_network):
        n = simple_network
        m = n.cluster.temporal.resample("3h")

        assert isinstance(m, pypsa.Network)
        assert len(m.snapshots) == 168 // 3


class TestDownsample:
    def test_downsample_stride_4(self, simple_network):
        n = simple_network
        result = downsample(n, 4)

        assert isinstance(result, TemporalClustering)
        assert len(result.n.snapshots) == 168 // 4

    def test_downsample_preserves_total_hours(self, simple_network):
        n = simple_network
        original_hours = n.snapshot_weightings["objective"].sum()

        result = downsample(n, 4)
        new_hours = result.n.snapshot_weightings["objective"].sum()

        assert np.isclose(original_hours, new_hours)

    def test_downsample_weightings_scaled(self, simple_network):
        n = simple_network
        stride = 4
        result = downsample(n, stride)

        expected_weight = n.snapshot_weightings["objective"].iloc[0] * stride
        assert np.isclose(
            result.n.snapshot_weightings["objective"].iloc[0], expected_weight
        )

    def test_downsample_time_series_selected(self, simple_network):
        n = simple_network
        result = downsample(n, 4)

        original_values = n.c.generators.dynamic["p_max_pu"]["gen1"].iloc[::4]
        new_values = result.n.c.generators.dynamic["p_max_pu"]["gen1"]
        assert np.allclose(original_values.values, new_values.values)

    def test_downsample_invalid_stride(self, simple_network):
        n = simple_network
        with pytest.raises(ValueError, match="stride must be >= 1"):
            downsample(n, 0)

    def test_downsample_accessor(self, simple_network):
        n = simple_network
        m = n.cluster.temporal.downsample(4)

        assert isinstance(m, pypsa.Network)
        assert len(m.snapshots) == 168 // 4


class TestFromSnapshotMap:
    def test_from_snapshot_map_basic(self, simple_network):
        n = simple_network
        snapshot_map = pd.Series(
            np.repeat(n.snapshots[::24], 24)[: len(n.snapshots)], index=n.snapshots
        )

        result = from_snapshot_map(n, snapshot_map)

        assert isinstance(result, TemporalClustering)
        assert len(result.n.snapshots) == 7

    def test_from_snapshot_map_preserves_hours(self, simple_network):
        n = simple_network
        original_hours = n.snapshot_weightings["objective"].sum()

        snapshot_map = pd.Series(
            np.repeat(n.snapshots[::24], 24)[: len(n.snapshots)], index=n.snapshots
        )
        result = from_snapshot_map(n, snapshot_map)

        assert np.isclose(
            result.n.snapshot_weightings["objective"].sum(), original_hours
        )

    def test_from_snapshot_map_invalid_index(self, simple_network):
        n = simple_network
        bad_index = pd.date_range("2019-01-01", periods=168, freq="h")
        snapshot_map = pd.Series(n.snapshots[0], index=bad_index)

        with pytest.raises(ValueError, match="snapshot_map index must match"):
            from_snapshot_map(n, snapshot_map)

    def test_from_snapshot_map_accessor(self, simple_network):
        n = simple_network
        snapshot_map = pd.Series(
            np.repeat(n.snapshots[::24], 24)[: len(n.snapshots)], index=n.snapshots
        )

        m = n.cluster.temporal.from_snapshot_map(snapshot_map)

        assert isinstance(m, pypsa.Network)
        assert len(m.snapshots) == 7


class TestNyears:
    def test_nyears_hourly_week(self, simple_network):
        n = simple_network
        expected = 168 / 8760
        assert np.isclose(n.nyears, expected)

    def test_nyears_full_year(self, yearly_network):
        n = yearly_network
        assert np.isclose(n.nyears, 1.0)

    def test_nyears_preserved_after_resample(self, simple_network):
        n = simple_network
        original_nyears = n.nyears

        result = resample(n, "3h")
        assert np.isclose(result.n.nyears, original_nyears)

    def test_nyears_preserved_after_downsample(self, simple_network):
        n = simple_network
        original_nyears = n.nyears

        result = downsample(n, 4)
        assert np.isclose(result.n.nyears, original_nyears)


class TestLeapDay:
    def test_drop_leap_day(self, yearly_network):
        n = yearly_network
        result = resample(n, "24h", drop_leap_day=True)

        leap_day = pd.Timestamp("2020-02-29")
        assert leap_day not in result.n.snapshots

        original_hours = n.snapshot_weightings["objective"].sum()
        assert np.isclose(
            result.n.snapshot_weightings["objective"].sum(), original_hours
        )


class TestAggregationRules:
    def test_custom_aggregation_rules(self, simple_network):
        n = simple_network
        custom_rules = {"p_max_pu": "max"}

        result = resample(n, "24h", aggregation_rules=custom_rules)

        assert "p_max_pu" in result.n.c.generators.dynamic


class TestAccessorFullResults:
    def test_get_resample_result(self, simple_network):
        n = simple_network
        result = n.cluster.temporal.get_resample_result("3h")

        assert isinstance(result, TemporalClustering)
        assert isinstance(result.n, pypsa.Network)
        assert isinstance(result.snapshot_map, pd.Series)

    def test_get_downsample_result(self, simple_network):
        n = simple_network
        result = n.cluster.temporal.get_downsample_result(4)

        assert isinstance(result, TemporalClustering)


class TestMultiperiod:
    def test_resample_multiperiod(self, multiperiod_network):
        n = multiperiod_network
        result = resample(n, "3h")

        assert result.n.has_periods
        assert len(result.n.periods) == 2
        assert np.isclose(
            result.n.snapshot_weightings["objective"].sum(),
            n.snapshot_weightings["objective"].sum(),
        )


class TestSegment:
    def test_segment_accessor_exists(self, simple_network):
        n = simple_network
        assert hasattr(n.cluster.temporal, "segment")


class TestEdgeCases:
    def test_empty_dynamic_data(self):
        n = pypsa.Network()
        n.set_snapshots(pd.date_range("2020-01-01", periods=24, freq="h"))
        n.add("Bus", "bus0")
        n.add("Generator", "gen0", bus="bus0", p_nom=100)

        result = resample(n, "3h")
        assert len(result.n.snapshots) == 8

    def test_single_snapshot_stride(self, simple_network):
        n = simple_network
        result = downsample(n, 1)

        assert len(result.n.snapshots) == len(n.snapshots)

    def test_large_stride(self, simple_network):
        n = simple_network
        result = downsample(n, 168)

        assert len(result.n.snapshots) == 1
        assert np.isclose(result.n.snapshot_weightings["objective"].iloc[0], 168)
