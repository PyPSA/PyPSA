# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import warnings

import numpy as np
import pandas as pd
import pytest

import pypsa
from pypsa.clustering.temporal import (
    TemporalClustering,
    downsample,
    from_snapshot_map,
    resample,
    segment,
)


@pytest.fixture
def simple_network():
    """Create a simple network with hourly resolution for one week."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2021-01-01", periods=168, freq="h"))

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
    """Create a network with full year hourly resolution (leap year)."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2020-01-01", periods=8784, freq="h"))

    n.add("Bus", "bus0")
    n.add(
        "Generator",
        "gen0",
        bus="bus0",
        p_nom=100,
        p_max_pu=np.random.default_rng(42).uniform(0.3, 1.0, 8784),
    )
    n.add("Load", "load0", bus="bus0", p_set=50)

    return n


@pytest.fixture
def multiperiod_network():
    """Create a network with investment periods."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2021-01-01", periods=24, freq="h"))
    n.set_investment_periods([2021, 2030])

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
        assert np.allclose(
            result.n.c.generators.dynamic["p_max_pu"],
            n.c.generators.dynamic["p_max_pu"].resample("3h").mean(),
        )

    def test_resample_snapshot_map(self, simple_network):
        n = simple_network
        result = resample(n, "3h")

        assert len(result.snapshot_map) == len(n.snapshots)
        assert result.snapshot_map.isin(result.n.snapshots).all()

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
        # Leap year has 8784 hours (366 days), so nyears is slightly > 1.0
        assert np.isclose(n.nyears, 8784 / 8760)

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
        n.set_snapshots(pd.date_range("2021-01-01", periods=24, freq="h"))
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

    def test_downsample_multiperiod(self, multiperiod_network):
        n = multiperiod_network
        original_hours = n.snapshot_weightings["objective"].sum()

        result = downsample(n, 4)

        # 24 snapshots per period / 4 = 6 per period, 2 periods = 12 total
        assert len(result.n.snapshots) == 12
        assert result.n.has_periods
        assert len(result.n.periods) == 2
        assert np.isclose(
            result.n.snapshot_weightings["objective"].sum(), original_hours
        )

    def test_downsample_multiperiod_snapshot_map(self, multiperiod_network):
        n = multiperiod_network
        result = downsample(n, 4)

        assert len(result.snapshot_map) == len(n.snapshots)
        assert all(s in result.n.snapshots for s in result.snapshot_map.values)

    def test_downsample_multiperiod_non_divisible_stride(self, multiperiod_network):
        n = multiperiod_network
        original_hours = n.snapshot_weightings["objective"].sum()

        result = downsample(n, 5)  # 24 % 5 = 4 remainder

        # ceil(24/5) = 5 snapshots per period, 2 periods = 10 total
        assert len(result.n.snapshots) == 10
        assert np.isclose(
            result.n.snapshot_weightings["objective"].sum(), original_hours
        )

    def test_downsample_multiperiod_large_stride(self, multiperiod_network):
        n = multiperiod_network
        original_hours = n.snapshot_weightings["objective"].sum()

        result = downsample(n, 24)  # Each period reduced to 1 snapshot

        assert len(result.n.snapshots) == 2  # One per period
        assert np.isclose(
            result.n.snapshot_weightings["objective"].sum(), original_hours
        )

    def test_from_snapshot_map_dataframe_input(self, simple_network):
        n = simple_network
        snapshot_map_series = pd.Series(
            np.repeat(n.snapshots[::24], 24)[: len(n.snapshots)], index=n.snapshots
        )
        snapshot_map_df = pd.DataFrame({"map": snapshot_map_series})

        result = from_snapshot_map(n, snapshot_map_df)

        assert isinstance(result, TemporalClustering)
        assert len(result.n.snapshots) == 7

    def test_downsample_stride_not_divisible(self, simple_network):
        n = simple_network
        original_hours = n.snapshot_weightings["objective"].sum()

        result = downsample(n, 5)

        assert len(result.n.snapshots) == (168 + 5 - 1) // 5  # ceiling division
        assert np.isclose(
            result.n.snapshot_weightings["objective"].sum(), original_hours
        )

    def test_resample_frequency_larger_than_snapshots(self, simple_network):
        n = simple_network
        original_hours = n.snapshot_weightings["objective"].sum()

        result = resample(n, "500h")

        assert len(result.n.snapshots) == 1
        assert np.isclose(
            result.n.snapshot_weightings["objective"].sum(), original_hours
        )


class TestAggregationRulesDetailed:
    def test_e_min_pu_uses_max(self, simple_network):
        n = simple_network
        n.add(
            "StorageUnit",
            "storage0",
            bus="bus0",
            p_nom=10,
            e_min_pu=np.random.default_rng(42).uniform(0.1, 0.3, 168),
        )

        result = resample(n, "24h")

        assert "e_min_pu" in result.n.c.storage_units.dynamic

    def test_e_max_pu_uses_min(self, simple_network):
        n = simple_network
        n.add(
            "StorageUnit",
            "storage0",
            bus="bus0",
            p_nom=10,
            e_max_pu=np.random.default_rng(42).uniform(0.7, 1.0, 168),
        )

        result = resample(n, "24h")

        assert "e_max_pu" in result.n.c.storage_units.dynamic

    def test_e_min_pu_clipped_below_e_max_pu(self, simple_network):
        n = simple_network
        rng = np.random.default_rng(0)
        e_max_pu = rng.uniform(0.3, 0.7, 168)
        e_min_pu = np.minimum(rng.uniform(0.3, 0.7, 168), e_max_pu)
        n.add(
            "Store",
            "store0",
            bus="bus0",
            e_nom=100,
            e_min_pu=e_min_pu,
            e_max_pu=e_max_pu,
        )

        result = resample(n, "24h")

        clustered_min = result.n.c.stores.dynamic["e_min_pu"]["store0"]
        clustered_max = result.n.c.stores.dynamic["e_max_pu"]["store0"]
        assert (clustered_min <= clustered_max).all()


class TestSegmentErrors:
    def test_segment_invalid_num_segments(self, simple_network):

        n = simple_network
        with pytest.raises(ValueError, match="num_segments must be >= 1"):
            segment(n, 0)

    def test_segment_map_too_many_segments(self, simple_network):
        pytest.importorskip("tsam")
        with pytest.raises(
            ValueError, match=r"cannot exceed number of snapshots \(168\)"
        ):
            simple_network.cluster.temporal.get_segment_map(169)

    def test_segment_map_requires_tsam_3(self, simple_network, monkeypatch):
        pytest.importorskip("tsam")
        monkeypatch.setattr("pypsa.clustering.temporal.version", lambda _: "2.3.9")
        with pytest.raises(ImportError, match=r"tsam >= 3 is required.*2\.3\.9"):
            simple_network.cluster.temporal.get_segment_map(10)

    def test_segment_map_no_dynamic_data_with_periods(self, multiperiod_network):
        pytest.importorskip("tsam")
        with pytest.raises(ValueError, match="No time-varying data"):
            multiperiod_network.cluster.temporal.get_segment_map(5)

    def test_segment_no_dynamic_data(self):
        pytest.importorskip("tsam")
        n = pypsa.Network()
        n.set_snapshots(pd.date_range("2021-01-01", periods=24, freq="h"))
        n.add("Bus", "bus0")
        n.add("Generator", "gen0", bus="bus0", p_nom=100)

        with pytest.raises(ValueError, match="No time-varying data"):
            segment(n, 5)


class TestSegmentFunctionality:
    def test_segment_basic(self, simple_network):
        pytest.importorskip("tsam")
        n = simple_network

        result = segment(n, 10)

        assert isinstance(result, TemporalClustering)
        assert len(result.n.snapshots) == 10
        assert np.isclose(
            result.n.snapshot_weightings["objective"].sum(),
            n.snapshot_weightings["objective"].sum(),
        )

    def test_segment_accessor(self, simple_network):
        pytest.importorskip("tsam")
        n = simple_network

        m = n.cluster.temporal.segment(10)

        assert isinstance(m, pypsa.Network)
        assert len(m.snapshots) == 10

    def test_segment_p_min_pu_not_above_p_max_pu(self, simple_network):
        pytest.importorskip("tsam")
        n = simple_network

        rng = np.random.default_rng(0)
        base = rng.uniform(0.3, 0.95, 168)
        p_max_pu = base.copy()
        p_min_pu = base.copy()
        p_max_pu[0] = 1.0
        p_min_pu[:2] = [0.1, 0.05]
        n.add(
            "Generator",
            "gen_eq",
            bus="bus0",
            p_nom=100,
            p_min_pu=p_min_pu,
            p_max_pu=p_max_pu,
        )

        m = segment(n, 30).n

        clustered_min = m.c.generators.dynamic["p_min_pu"]["gen_eq"]
        clustered_max = m.c.generators.dynamic["p_max_pu"]["gen_eq"]
        assert (clustered_min <= clustered_max).all()


def _run_starts(mapping: pd.Series) -> np.ndarray:
    values = mapping.to_numpy()
    return np.r_[True, values[1:] != values[:-1]]


class TestSegmentMap:
    @pytest.mark.parametrize("num_segments", [1, 10, 168])
    def test_segment_map_runs(self, simple_network, num_segments):
        pytest.importorskip("tsam")
        mapping = simple_network.cluster.temporal.get_segment_map(num_segments)
        starts = _run_starts(mapping)

        assert mapping.index.equals(simple_network.snapshots)
        assert starts.sum() == mapping.nunique() == num_segments
        assert (mapping[starts] == mapping.index[starts]).all()

    @pytest.mark.parametrize("varying", [True, False])
    @pytest.mark.parametrize("num_segments", [1, 5, 24])
    def test_segment_map_within_periods(
        self, multiperiod_network, num_segments, varying
    ):
        pytest.importorskip("tsam")
        n = multiperiod_network
        rng = np.random.default_rng(42)
        p_max_pu = rng.uniform(size=48) if varying else np.full(48, 0.5)
        n.add("Generator", "gen_vary", bus="bus0", p_max_pu=p_max_pu)
        mapping = n.cluster.temporal.get_segment_map(num_segments)
        periods = mapping.index.get_level_values("period")

        assert mapping.index.equals(n.snapshots)
        assert (mapping.map(lambda s: s[0]) == periods).all()
        assert (mapping.groupby(periods).nunique() == num_segments).all()

    @pytest.mark.parametrize(
        ("p_max_pu", "drop_duplicates"),
        [("duplicate", True), ("constant", True), ("constant", False)],
    )
    def test_segment_map_ignores_redundant_series(
        self, simple_network, p_max_pu, drop_duplicates
    ):
        pytest.importorskip("tsam")
        n = simple_network
        expected = n.cluster.temporal.get_segment_map(
            20, drop_duplicates=drop_duplicates
        )
        series = {
            "duplicate": n.c.generators.dynamic["p_max_pu"]["gen1"].to_numpy(),
            "constant": np.full(168, 0.7),
        }[p_max_pu]
        for name in ["copy0", "copy1"]:
            n.add("Generator", name, bus="bus1", p_max_pu=series)

        pd.testing.assert_series_equal(
            n.cluster.temporal.get_segment_map(20, drop_duplicates=drop_duplicates),
            expected,
        )


def _timestep(label: pd.Timestamp | tuple) -> pd.Timestamp:
    return label[1] if isinstance(label, tuple) else label


def _leap_week_network(periods: bool, hours: int = 96) -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2020-02-27", periods=hours, freq="h"))
    if periods:
        n.set_investment_periods([2020, 2030])
    n.add("Bus", "bus0")
    rng = np.random.default_rng(0)
    n.add("Generator", "gen0", bus="bus0", p_max_pu=rng.uniform(size=len(n.snapshots)))
    return n


def _with_varying_generator(n: pypsa.Network) -> pypsa.Network:
    rng = np.random.default_rng(1)
    n.add(
        "Generator", "gen_vary", bus="bus0", p_max_pu=rng.uniform(size=len(n.snapshots))
    )
    return n


class TestSnapshotMaps:
    @pytest.mark.parametrize("network", ["simple_network", "multiperiod_network"])
    @pytest.mark.parametrize("method", ["resample", "downsample", "segment"])
    def test_method_applies_map(self, network, method, request):
        if method == "segment":
            pytest.importorskip("tsam")
        n = _with_varying_generator(request.getfixturevalue(network))
        t = n.cluster.temporal
        first = {attr: "first" for c in n.components for attr in c.dynamic}
        mapping, clustered, rules = {
            "resample": lambda: (t.get_resample_map("3h"), t.resample("3h"), None),
            "downsample": lambda: (t.get_downsample_map(5), t.downsample(5), first),
            "segment": lambda: (t.get_segment_map(6), t.segment(6), None),
        }[method]()

        assert clustered.equals(t.from_snapshot_map(mapping, aggregation_rules=rules))

    @pytest.mark.parametrize("network", ["simple_network", "multiperiod_network"])
    @pytest.mark.parametrize(("method", "size"), [("resample", 3), ("downsample", 5)])
    def test_block_maps_stay_within_periods(self, network, method, size, request):
        n = request.getfixturevalue(network)
        mapping = {
            "resample": lambda: n.cluster.temporal.get_resample_map(f"{size}h"),
            "downsample": lambda: n.cluster.temporal.get_downsample_map(size),
        }[method]()
        groups = (
            mapping.index.get_level_values("period")
            if n.has_periods
            else np.zeros(len(mapping))
        )
        per_group = len(n.snapshots) // len(np.unique(groups))

        assert mapping.index.equals(n.snapshots)
        assert (
            mapping[_run_starts(mapping)] == mapping.index[_run_starts(mapping)]
        ).all()
        assert (mapping.groupby(groups).nunique() == -(-per_group // size)).all()
        if n.has_periods:
            assert (mapping.map(lambda s: s[0]) == groups).all()

    @pytest.mark.parametrize("periods", [False, True])
    @pytest.mark.parametrize("offset", ["5h", "24h"])
    def test_drop_leap_day_moves_feb_29_to_march_1(self, periods, offset):
        n = _leap_week_network(periods)
        kept = resample(n, offset)
        result = resample(n, offset, drop_leap_day=True)
        snapshots = result.n.snapshots
        timesteps = pd.DatetimeIndex([_timestep(s) for s in snapshots])
        mapped = pd.DatetimeIndex([_timestep(s) for s in result.snapshot_map])

        assert not ((timesteps.month == 2) & (timesteps.day == 29)).any()
        assert result.snapshot_map.isin(snapshots).all()
        assert not ((mapped.month == 2) & (mapped.day == 29)).any()
        assert np.isclose(
            result.n.snapshot_weightings["objective"].sum(),
            n.snapshot_weightings["objective"].sum(),
        )
        pd.testing.assert_frame_equal(
            result.n.generators_t.p_max_pu, kept.n.generators_t.p_max_pu.loc[snapshots]
        )

    @pytest.mark.parametrize("stride", [2, 3])
    def test_downsample_segmented_network_keeps_hours(self, simple_network, stride):
        pytest.importorskip("tsam")
        m = segment(simple_network, 40).n
        result = downsample(m, stride)

        assert np.isclose(
            result.n.snapshot_weightings["objective"].sum(),
            simple_network.snapshot_weightings["objective"].sum(),
        )

    @pytest.mark.parametrize("periods", [False, True])
    @pytest.mark.parametrize("offset", ["5h", "24h"])
    @pytest.mark.parametrize("method", ["resample", "get_resample_map"])
    def test_drop_leap_day_requires_march_1(self, periods, offset, method):
        t = _leap_week_network(periods, hours=72).cluster.temporal
        cluster = {"resample": t.resample, "get_resample_map": t.get_resample_map}
        with pytest.raises(ValueError, match="requires a snapshot on March 1"):
            cluster[method](offset, drop_leap_day=True)

    @pytest.mark.parametrize(
        ("cluster", "kept"),
        [
            (lambda t: t.downsample(4), 0),
            (lambda t: t.resample("4h", aggregation_rules={"p_max_pu": "first"}), 1),
            (
                lambda t: t.from_snapshot_map(
                    t.get_downsample_map(4), aggregation_rules={"p_max_pu": "first"}
                ),
                1,
            ),
        ],
    )
    def test_first_value_with_missing_start(self, simple_network, cluster, kept):
        n = simple_network
        p_max_pu = np.random.default_rng(2).uniform(size=168)
        p_max_pu[0] = np.nan
        n.add("Generator", "gen_nan", bus="bus0", p_max_pu=p_max_pu)
        m = cluster(n.cluster.temporal)

        np.testing.assert_equal(
            m.generators_t.p_max_pu["gen_nan"].iloc[0], p_max_pu[kept]
        )

    @pytest.mark.parametrize(
        "method", ["resample", "downsample", "segment", "from_snapshot_map"]
    )
    def test_solved_network_warning_points_at_caller(self, simple_network, method):
        if method == "segment":
            pytest.importorskip("tsam")
        n = simple_network
        n.optimize(solver_name="highs")
        n.model.solver_model = None
        t = n.cluster.temporal
        mapping = t.get_downsample_map(4)
        cluster = {
            "resample": lambda: t.resample("3h"),
            "downsample": lambda: t.downsample(4),
            "segment": lambda: t.segment(6),
            "from_snapshot_map": lambda: t.from_snapshot_map(mapping),
        }[method]
        with pytest.warns(UserWarning, match="solved network") as record:
            cluster()

        assert {w.filename for w in record if "solved network" in str(w.message)} == {
            __file__
        }


class TestDeprecations:
    @pytest.mark.parametrize(
        "method",
        [
            "get_resample_result",
            "get_downsample_result",
            "get_segment_result",
            "get_from_snapshot_map_result",
            "get_resample_result_leap_day",
        ],
    )
    def test_result_methods_warn_and_match_main_path(self, simple_network, method):
        if method == "get_segment_result":
            pytest.importorskip("tsam")
        leap = method.endswith("_leap_day")
        n = _leap_week_network(False) if leap else simple_network
        t = n.cluster.temporal
        mapping, clustered, deprecated = {
            "get_resample_result": lambda: (
                t.get_resample_map("3h"),
                t.resample("3h"),
                lambda: t.get_resample_result("3h"),
            ),
            "get_resample_result_leap_day": lambda: (
                t.get_resample_map("24h", drop_leap_day=True),
                t.resample("24h", drop_leap_day=True),
                lambda: t.get_resample_result("24h", drop_leap_day=True),
            ),
            "get_downsample_result": lambda: (
                t.get_downsample_map(4),
                t.downsample(4),
                lambda: t.get_downsample_result(4),
            ),
            "get_segment_result": lambda: (
                t.get_segment_map(10),
                t.segment(10),
                lambda: t.get_segment_result(10),
            ),
            "get_from_snapshot_map_result": lambda: (
                t.get_downsample_map(4),
                t.from_snapshot_map(t.get_downsample_map(4)),
                lambda: t.get_from_snapshot_map_result(t.get_downsample_map(4)),
            ),
        }[method]()
        name = method.removesuffix("_leap_day")
        with pytest.warns(DeprecationWarning, match=f"{name} is deprecated"):
            result = deprecated()

        assert isinstance(result, TemporalClustering)
        pd.testing.assert_series_equal(result.snapshot_map, mapping)
        assert result.n.equals(clustered)

    @pytest.mark.parametrize(
        "call",
        [
            lambda n: n.cluster.temporal.segment(6, solver="highs"),
            lambda n: n.cluster.temporal.get_segment_map(6, solver="highs"),
            lambda n: segment(n, 6, solver="highs"),
            lambda n: n.cluster.temporal.get_segment_result(6, solver="highs"),
        ],
        ids=["segment", "get_segment_map", "module_segment", "get_segment_result"],
    )
    def test_solver_is_deprecated(self, simple_network, call):
        pytest.importorskip("tsam")
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            call(simple_network)

        assert any("`solver` is deprecated" in str(w.message) for w in record)


class TestStochasticNotSupported:
    """Temporal clustering should raise error for stochastic networks."""

    @pytest.fixture
    def stochastic_network(self):
        """Create a simple stochastic network."""
        n = pypsa.Network()
        n.set_snapshots(pd.date_range("2021-01-01", periods=24, freq="h"))
        n.set_scenarios(low=0.5, high=0.5)

        n.add("Bus", "bus0")
        n.add("Generator", "gen0", bus="bus0", p_nom=100)
        n.add("Load", "load0", bus="bus0", p_set=50)

        return n

    def test_resample_stochastic_raises(self, stochastic_network):
        with pytest.raises(NotImplementedError, match="stochastic networks"):
            resample(stochastic_network, "3h")

    def test_downsample_stochastic_raises(self, stochastic_network):
        with pytest.raises(NotImplementedError, match="stochastic networks"):
            downsample(stochastic_network, 4)

    def test_segment_stochastic_raises(self, stochastic_network):
        pytest.importorskip("tsam")

        with pytest.raises(NotImplementedError, match="stochastic networks"):
            segment(stochastic_network, 10)

    def test_from_snapshot_map_stochastic_raises(self, stochastic_network):
        snapshot_map = pd.Series(
            stochastic_network.snapshots[0], index=stochastic_network.snapshots
        )
        with pytest.raises(NotImplementedError, match="stochastic networks"):
            from_snapshot_map(stochastic_network, snapshot_map)
