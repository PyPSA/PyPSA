# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for ``pypsa.utils.apply_outage_schedule``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pypsa
from pypsa.utils.outage_schedule import (
    _build_factor_series,
    apply_outage_schedule,
    build_factor_series_per_asset,
)


def _two_bus_network(
    snapshots: pd.DatetimeIndex | None = None,
    *,
    p_min_pu_link: float = -1.0,
) -> pypsa.Network:
    if snapshots is None:
        snapshots = pd.date_range("2025-01-01", periods=48, freq="h")
    n = pypsa.Network()
    n.set_snapshots(snapshots)
    n.add("Bus", "b0")
    n.add("Bus", "b1")
    n.add("Generator", "g0", bus="b0", p_nom=100.0)
    n.add("Generator", "g1", bus="b1", p_nom=50.0)
    n.add("Link", "l0", bus0="b0", bus1="b1", p_nom=200.0, p_min_pu=p_min_pu_link)
    return n


def _outage_row(
    asset_id: str,
    component: str,
    start: str,
    end: str,
    p_max_pu: float,
) -> dict[str, object]:
    return {
        "asset_id": asset_id,
        "component": component,
        "start": pd.Timestamp(start),
        "end": pd.Timestamp(end),
        "p_max_pu": p_max_pu,
    }


def test_re_exported_from_top_level() -> None:
    """`pypsa.apply_outage_schedule` is the public entry point."""
    assert pypsa.apply_outage_schedule is apply_outage_schedule
    assert "apply_outage_schedule" in pypsa.utils.__all__


def test_generator_full_outage_zeros_window() -> None:
    """A `p_max_pu=0` outage zeroes only the masked snapshots."""
    n = _two_bus_network()
    outages = pd.DataFrame(
        [
            _outage_row("g0", "Generator", "2025-01-01 06:00", "2025-01-01 12:00", 0.0),
        ]
    )

    counts = apply_outage_schedule(n, outages)

    assert counts == {"Generator": 1, "Link": 0}
    series = n.generators_t.p_max_pu["g0"]
    assert (series.loc["2025-01-01 06:00":"2025-01-01 12:00"] == 0.0).all()
    assert (series.loc["2025-01-01 13:00":] == 1.0).all()


def test_generator_partial_derate_composes_multiplicatively() -> None:
    """Outage factor multiplies existing dynamic `p_max_pu` (capacity factor)."""
    n = _two_bus_network()
    # Pre-existing CF profile: ramp 0.0 -> 0.5 -> 1.0 -> 0.5 -> ...
    profile = pd.Series(
        np.tile([0.0, 0.5, 1.0, 0.5], len(n.snapshots) // 4),
        index=n.snapshots,
    )
    n.generators_t.p_max_pu["g0"] = profile

    outages = pd.DataFrame(
        [
            _outage_row("g0", "Generator", "2025-01-01 02:00", "2025-01-01 05:00", 0.5),
        ]
    )
    apply_outage_schedule(n, outages)

    series = n.generators_t.p_max_pu["g0"]
    # In-window: existing * 0.5
    assert series.loc["2025-01-01 02:00"] == pytest.approx(
        profile.loc["2025-01-01 02:00"] * 0.5
    )
    assert series.loc["2025-01-01 04:00"] == pytest.approx(
        profile.loc["2025-01-01 04:00"] * 0.5
    )
    # Out-of-window: untouched
    assert series.loc["2025-01-01 06:00"] == pytest.approx(
        profile.loc["2025-01-01 06:00"]
    )
    assert series.loc["2025-01-01 01:00"] == pytest.approx(
        profile.loc["2025-01-01 01:00"]
    )


def test_overlapping_outages_collapse_to_minimum() -> None:
    """Two overlapping windows on the same asset take the worse derate."""
    n = _two_bus_network()
    outages = pd.DataFrame(
        [
            _outage_row(
                "g0", "Generator", "2025-01-01 06:00", "2025-01-01 12:00", 0.50
            ),
            _outage_row(
                "g0", "Generator", "2025-01-01 09:00", "2025-01-01 15:00", 0.20
            ),
        ]
    )
    apply_outage_schedule(n, outages)

    series = n.generators_t.p_max_pu["g0"]
    # 06-08: only the 0.50 window
    assert series.loc["2025-01-01 06:00":"2025-01-01 08:00"].eq(0.50).all()
    # 09-12: overlap -> min(0.50, 0.20) = 0.20
    assert series.loc["2025-01-01 09:00":"2025-01-01 12:00"].eq(0.20).all()
    # 13-15: only the 0.20 window
    assert series.loc["2025-01-01 13:00":"2025-01-01 15:00"].eq(0.20).all()


def test_link_outage_clamps_reverse_to_static_p_min_pu() -> None:
    """Link p_min_pu is clamped to the static lower bound (e.g. asymmetric HVDC)."""
    # Real-world example: NorNed has static p_min_pu = -0.957
    n = _two_bus_network(p_min_pu_link=-0.957)
    outages = pd.DataFrame(
        [
            _outage_row("l0", "Link", "2025-01-01 00:00", "2025-01-01 23:00", 0.30),
        ]
    )
    counts = apply_outage_schedule(n, outages)

    assert counts == {"Generator": 0, "Link": 1}
    # Forward: derated to 0.30
    pmax = n.links_t.p_max_pu["l0"]
    assert pmax.loc["2025-01-01 12:00"] == pytest.approx(0.30)
    # Reverse: -0.30 would be more permissive than -0.957, so it's the binding cap
    pmin = n.links_t.p_min_pu["l0"]
    assert pmin.loc["2025-01-01 12:00"] == pytest.approx(-0.30)
    # Out-of-window: untouched (pmin series only spans outage rows)
    assert pmax.loc["2025-01-02 00:00"] == pytest.approx(1.0)


def test_link_reverse_does_not_exceed_static_floor() -> None:
    """Reverse availability never gets *worse* than the static p_min_pu."""
    n = _two_bus_network(p_min_pu_link=-0.5)
    outages = pd.DataFrame(
        [
            # Outage factor 0.9 -> reverse would be -0.9, clamped to -0.5
            _outage_row("l0", "Link", "2025-01-01 00:00", "2025-01-01 12:00", 0.9),
        ]
    )
    apply_outage_schedule(n, outages)

    pmin = n.links_t.p_min_pu["l0"].loc["2025-01-01 06:00"]
    assert pmin == pytest.approx(-0.5)


def test_unknown_asset_raises_when_requested() -> None:
    """`on_unknown='raise'` surfaces typos in `asset_id` immediately."""
    n = _two_bus_network()
    outages = pd.DataFrame(
        [
            _outage_row(
                "g_missing", "Generator", "2025-01-01 06:00", "2025-01-01 12:00", 0.0
            ),
        ]
    )
    with pytest.raises(KeyError, match="not in network"):
        apply_outage_schedule(n, outages, on_unknown="raise")


def test_unknown_asset_ignored_when_requested() -> None:
    """`on_unknown='ignore'` keeps the network untouched, no warning."""
    n = _two_bus_network()
    outages = pd.DataFrame(
        [
            _outage_row(
                "g_missing", "Generator", "2025-01-01 06:00", "2025-01-01 12:00", 0.0
            ),
            _outage_row("g0", "Generator", "2025-01-01 06:00", "2025-01-01 12:00", 0.0),
        ]
    )
    counts = apply_outage_schedule(n, outages, on_unknown="ignore")
    assert counts["Generator"] == 1


def test_missing_required_columns_raises() -> None:
    """Schema validation rejects DataFrames missing any of the required columns."""
    n = _two_bus_network()
    bad = pd.DataFrame(
        {
            "asset_id": ["g0"],
            "component": ["Generator"],
            # missing: start, end, p_max_pu
        }
    )
    with pytest.raises(ValueError, match="missing required columns"):
        apply_outage_schedule(n, bad)


def test_unsupported_component_raises() -> None:
    """Only Generator and Link are supported (e.g. Lines/Buses are out of scope)."""
    n = _two_bus_network()
    outages = pd.DataFrame(
        [
            _outage_row(
                "anything", "Store", "2025-01-01 06:00", "2025-01-01 12:00", 0.0
            ),
        ]
    )
    with pytest.raises(ValueError, match="unsupported component"):
        apply_outage_schedule(n, outages)


def test_p_max_pu_outside_unit_interval_raises() -> None:
    """`p_max_pu > 1` or `< 0` is rejected with a clear error."""
    n = _two_bus_network()
    bad = pd.DataFrame(
        [
            _outage_row("g0", "Generator", "2025-01-01 06:00", "2025-01-01 12:00", 1.5),
        ]
    )
    with pytest.raises(ValueError, match=r"p_max_pu must be in \[0, 1\]"):
        apply_outage_schedule(n, bad)


def test_non_datetime_snapshots_raises() -> None:
    """Snapshots must be a DatetimeIndex; the default `Index(['now'])` is rejected."""
    n = pypsa.Network()
    n.add("Bus", "b0")
    n.add("Generator", "g0", bus="b0", p_nom=100)
    outages = pd.DataFrame(
        [
            _outage_row("g0", "Generator", "2025-01-01 06:00", "2025-01-01 12:00", 0.0),
        ]
    )
    with pytest.raises(TypeError, match="DatetimeIndex"):
        apply_outage_schedule(n, outages)


def test_empty_outages_is_noop() -> None:
    """Empty DataFrame returns zero counts and leaves the network untouched."""
    n = _two_bus_network()
    counts = apply_outage_schedule(
        n,
        pd.DataFrame(
            columns=list(_build_factor_series.__defaults__ or [])
            + ["asset_id", "component", "start", "end", "p_max_pu"]
        ),
    )
    assert counts == {"Generator": 0, "Link": 0}
    # No dynamic columns added
    assert "g0" not in n.generators_t.p_max_pu.columns
    assert "l0" not in n.links_t.p_max_pu.columns


def test_tz_aware_outage_on_tz_naive_snapshots_aligns_correctly() -> None:
    """A tz-aware outage window applies correctly to tz-naive snapshots (UTC convention)."""
    n = _two_bus_network()
    outages = pd.DataFrame(
        [
            {
                "asset_id": "g0",
                "component": "Generator",
                "start": pd.Timestamp("2025-01-01 06:00", tz="UTC"),
                "end": pd.Timestamp("2025-01-01 12:00", tz="UTC"),
                "p_max_pu": 0.0,
            },
        ]
    )
    apply_outage_schedule(n, outages)
    series = n.generators_t.p_max_pu["g0"]
    assert (series.loc["2025-01-01 06:00":"2025-01-01 12:00"] == 0.0).all()


def test_build_factor_series_per_asset_skips_unaffected_assets() -> None:
    """Per-asset factor builder only returns assets whose factor dips below 1."""
    snapshots = pd.date_range("2025-01-01", periods=24, freq="h")
    outages = pd.DataFrame(
        [
            _outage_row("g0", "Generator", "2025-01-01 06:00", "2025-01-01 12:00", 0.5),
            # Window entirely outside snapshots
            _outage_row("g1", "Generator", "2024-12-30 00:00", "2024-12-30 06:00", 0.0),
        ]
    )
    result = build_factor_series_per_asset(outages, snapshots, component="Generator")
    assert set(result.keys()) == {"g0"}


def test_partial_overlap_with_snapshots_only_writes_intersection() -> None:
    """Outage extending past snapshot range writes only the overlapping portion."""
    snapshots = pd.date_range("2025-01-01", periods=12, freq="h")
    n = _two_bus_network(snapshots=snapshots)
    outages = pd.DataFrame(
        [
            _outage_row("g0", "Generator", "2025-01-01 06:00", "2025-01-02 18:00", 0.0),
        ]
    )
    apply_outage_schedule(n, outages)
    series = n.generators_t.p_max_pu["g0"]
    assert (series.loc["2025-01-01 06:00":"2025-01-01 11:00"] == 0.0).all()
    assert series.shape[0] == 12
