# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT
"""Tests for the `apply_outage_schedule` helper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pypsa
from pypsa import apply_outage_schedule, build_factor_series_per_asset


def _build_network(n_snaps: int = 24) -> pypsa.Network:
    """Minimal network: one Generator and one asymmetric Link."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2025-01-01", periods=n_snaps, freq="h"))
    n.add("Bus", "NL")
    n.add("Bus", "NO")
    n.add(
        "Generator",
        "Wind",
        bus="NL",
        p_nom=100.0,
        carrier="wind",
        marginal_cost=0.0,
    )
    n.add(
        "Link",
        "NorNed",
        bus0="NL",
        bus1="NO",
        p_nom=700.0,
        p_min_pu=-0.957,  # asymmetric: 700 MW NL→NO, 670 MW NO→NL
        p_max_pu=1.0,
        marginal_cost=0.0,
    )
    return n


def _outages(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
            "asset_id",
            "component",
            "start",
            "end",
            "p_max_pu",
        ],
    )


class TestApplyOutageSchedule:
    def test_empty_outages_is_noop(self):
        n = _build_network()
        out = apply_outage_schedule(n, _outages([]))
        assert out == {"Generator": 0, "Link": 0}
        assert n.generators_t.p_max_pu.empty
        assert n.links_t.p_max_pu.empty

    def test_missing_columns_raises(self):
        n = _build_network()
        bad = pd.DataFrame({"asset_id": ["Wind"], "component": ["Generator"]})
        with pytest.raises(ValueError, match="missing required columns"):
            apply_outage_schedule(n, bad)

    def test_unsupported_component_raises(self):
        n = _build_network()
        df = _outages(
            [
                {
                    "asset_id": "Wind",
                    "component": "StorageUnit",
                    "start": pd.Timestamp("2025-01-01 06:00", tz="UTC"),
                    "end": pd.Timestamp("2025-01-01 12:00", tz="UTC"),
                    "p_max_pu": 0.0,
                }
            ]
        )
        with pytest.raises(ValueError, match="unsupported component"):
            apply_outage_schedule(n, df)

    def test_generator_full_outage_writes_zeros(self):
        n = _build_network()
        df = _outages(
            [
                {
                    "asset_id": "Wind",
                    "component": "Generator",
                    "start": pd.Timestamp("2025-01-01 06:00", tz="UTC"),
                    "end": pd.Timestamp("2025-01-01 11:00", tz="UTC"),
                    "p_max_pu": 0.0,
                }
            ]
        )
        counts = apply_outage_schedule(n, df)
        assert counts == {"Generator": 1, "Link": 0}
        s = n.generators_t.p_max_pu["Wind"]
        assert (s.iloc[6:12] == 0.0).all()
        assert (s.iloc[:6] == 1.0).all()
        assert (s.iloc[12:] == 1.0).all()

    def test_generator_outage_composes_with_existing_cf(self):
        n = _build_network()
        # Existing CF profile: 0.30 everywhere
        n.generators_t.p_max_pu["Wind"] = pd.Series(0.30, index=n.snapshots)
        df = _outages(
            [
                {
                    "asset_id": "Wind",
                    "component": "Generator",
                    "start": pd.Timestamp("2025-01-01 06:00", tz="UTC"),
                    "end": pd.Timestamp("2025-01-01 11:00", tz="UTC"),
                    "p_max_pu": 0.50,
                }
            ]
        )
        apply_outage_schedule(n, df)
        s = n.generators_t.p_max_pu["Wind"]
        assert np.allclose(s.iloc[6:12], 0.15)  # 0.30 * 0.50
        assert np.allclose(s.iloc[:6], 0.30)
        assert np.allclose(s.iloc[12:], 0.30)

    def test_link_full_outage_zeroes_both_directions(self):
        n = _build_network()
        df = _outages(
            [
                {
                    "asset_id": "NorNed",
                    "component": "Link",
                    "start": pd.Timestamp("2025-01-01 06:00", tz="UTC"),
                    "end": pd.Timestamp("2025-01-01 11:00", tz="UTC"),
                    "p_max_pu": 0.0,
                }
            ]
        )
        apply_outage_schedule(n, df)
        pmax = n.links_t.p_max_pu["NorNed"]
        pmin = n.links_t.p_min_pu["NorNed"]
        assert (pmax.iloc[6:12] == 0.0).all()
        assert (pmin.iloc[6:12] == 0.0).all()  # reverse also zero
        # Outside outage: reverse stays at static p_min_pu = -0.957
        assert (pmin.iloc[:6] == -0.957).all()
        assert (pmax.iloc[:6] == 1.0).all()

    def test_link_partial_outage_clamps_reverse_to_factor(self):
        """50% derate on NorNed: forward 0.5, reverse -0.5 (not -0.957)."""
        n = _build_network()
        df = _outages(
            [
                {
                    "asset_id": "NorNed",
                    "component": "Link",
                    "start": pd.Timestamp("2025-01-01 06:00", tz="UTC"),
                    "end": pd.Timestamp("2025-01-01 11:00", tz="UTC"),
                    "p_max_pu": 0.5,
                }
            ]
        )
        apply_outage_schedule(n, df)
        pmax = n.links_t.p_max_pu["NorNed"]
        pmin = n.links_t.p_min_pu["NorNed"]
        assert (pmax.iloc[6:12] == 0.5).all()
        assert np.allclose(pmin.iloc[6:12], -0.5)

    def test_overlapping_rows_min_wins(self):
        n = _build_network()
        df = _outages(
            [
                {
                    "asset_id": "Wind",
                    "component": "Generator",
                    "start": pd.Timestamp("2025-01-01 06:00", tz="UTC"),
                    "end": pd.Timestamp("2025-01-01 18:00", tz="UTC"),
                    "p_max_pu": 0.50,
                },
                {
                    "asset_id": "Wind",
                    "component": "Generator",
                    "start": pd.Timestamp("2025-01-01 10:00", tz="UTC"),
                    "end": pd.Timestamp("2025-01-01 14:00", tz="UTC"),
                    "p_max_pu": 0.0,
                },
            ]
        )
        apply_outage_schedule(n, df)
        s = n.generators_t.p_max_pu["Wind"]
        assert (s.iloc[6:10] == 0.50).all()
        assert (s.iloc[10:15] == 0.0).all()
        assert (s.iloc[15:19] == 0.50).all()
        assert (s.iloc[19:] == 1.0).all()

    def test_unknown_asset_warns_by_default(self, caplog):
        n = _build_network()
        df = _outages(
            [
                {
                    "asset_id": "DoesNotExist",
                    "component": "Generator",
                    "start": pd.Timestamp("2025-01-01 06:00", tz="UTC"),
                    "end": pd.Timestamp("2025-01-01 11:00", tz="UTC"),
                    "p_max_pu": 0.0,
                }
            ]
        )
        counts = apply_outage_schedule(n, df)
        assert counts == {"Generator": 0, "Link": 0}
        assert any("not in network" in r.message for r in caplog.records)

    def test_unknown_asset_raises_when_requested(self):
        n = _build_network()
        df = _outages(
            [
                {
                    "asset_id": "DoesNotExist",
                    "component": "Link",
                    "start": pd.Timestamp("2025-01-01 06:00", tz="UTC"),
                    "end": pd.Timestamp("2025-01-01 11:00", tz="UTC"),
                    "p_max_pu": 0.0,
                }
            ]
        )
        with pytest.raises(KeyError, match="not in network"):
            apply_outage_schedule(n, df, on_unknown="raise")

    def test_unknown_asset_silent_when_ignore(self, caplog):
        n = _build_network()
        df = _outages(
            [
                {
                    "asset_id": "DoesNotExist",
                    "component": "Link",
                    "start": pd.Timestamp("2025-01-01 06:00", tz="UTC"),
                    "end": pd.Timestamp("2025-01-01 11:00", tz="UTC"),
                    "p_max_pu": 0.0,
                }
            ]
        )
        with caplog.at_level("WARNING"):
            apply_outage_schedule(n, df, on_unknown="ignore")
        assert not any("not in network" in r.message for r in caplog.records)

    def test_tz_aware_outage_on_tz_naive_snapshots(self):
        """Snapshots tz-naive, outage tz-aware UTC — comparison must still hit."""
        n = _build_network()
        assert n.snapshots.tz is None
        df = _outages(
            [
                {
                    "asset_id": "Wind",
                    "component": "Generator",
                    "start": pd.Timestamp("2025-01-01 06:00", tz="UTC"),
                    "end": pd.Timestamp("2025-01-01 11:00", tz="UTC"),
                    "p_max_pu": 0.25,
                }
            ]
        )
        apply_outage_schedule(n, df)
        s = n.generators_t.p_max_pu["Wind"]
        assert (s.iloc[6:12] == 0.25).all()

    def test_returns_counts_per_component(self):
        n = _build_network()
        df = _outages(
            [
                {
                    "asset_id": "Wind",
                    "component": "Generator",
                    "start": pd.Timestamp("2025-01-01 00:00", tz="UTC"),
                    "end": pd.Timestamp("2025-01-01 02:00", tz="UTC"),
                    "p_max_pu": 0.5,
                },
                {
                    "asset_id": "NorNed",
                    "component": "Link",
                    "start": pd.Timestamp("2025-01-01 00:00", tz="UTC"),
                    "end": pd.Timestamp("2025-01-01 02:00", tz="UTC"),
                    "p_max_pu": 0.5,
                },
            ]
        )
        counts = apply_outage_schedule(n, df)
        assert counts == {"Generator": 1, "Link": 1}


class TestBuildFactorSeriesPerAsset:
    def test_empty(self):
        snaps = pd.date_range("2025-01-01", periods=24, freq="h")
        empty = pd.DataFrame(
            columns=[
                "asset_id",
                "component",
                "start",
                "end",
                "p_max_pu",
            ]
        )
        assert build_factor_series_per_asset(empty, snaps) == {}

    def test_filters_to_unity_only(self):
        """Assets whose factor never dips below 1.0 are not returned."""
        snaps = pd.date_range("2025-01-01", periods=24, freq="h")
        df = pd.DataFrame(
            [
                {
                    "asset_id": "A",
                    "component": "Generator",
                    "start": pd.Timestamp("2025-01-01 06:00", tz="UTC"),
                    "end": pd.Timestamp("2025-01-01 11:00", tz="UTC"),
                    "p_max_pu": 1.0,
                },  # no-op
                {
                    "asset_id": "B",
                    "component": "Generator",
                    "start": pd.Timestamp("2025-01-01 06:00", tz="UTC"),
                    "end": pd.Timestamp("2025-01-01 11:00", tz="UTC"),
                    "p_max_pu": 0.5,
                },
            ]
        )
        out = build_factor_series_per_asset(df, snaps)
        assert set(out) == {"B"}

    def test_component_filter(self):
        snaps = pd.date_range("2025-01-01", periods=24, freq="h")
        df = pd.DataFrame(
            [
                {
                    "asset_id": "G",
                    "component": "Generator",
                    "start": pd.Timestamp("2025-01-01 06:00", tz="UTC"),
                    "end": pd.Timestamp("2025-01-01 11:00", tz="UTC"),
                    "p_max_pu": 0.5,
                },
                {
                    "asset_id": "L",
                    "component": "Link",
                    "start": pd.Timestamp("2025-01-01 06:00", tz="UTC"),
                    "end": pd.Timestamp("2025-01-01 11:00", tz="UTC"),
                    "p_max_pu": 0.5,
                },
            ]
        )
        gens = build_factor_series_per_asset(df, snaps, component="Generator")
        links = build_factor_series_per_asset(df, snaps, component="Link")
        assert set(gens) == {"G"}
        assert set(links) == {"L"}
