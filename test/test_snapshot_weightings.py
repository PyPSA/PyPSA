# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import pandas as pd
import pytest

import pypsa


@pytest.fixture
def time_index():
    start_time = pd.Timestamp("2023-01-01T00:00:00")
    end_time = pd.Timestamp("2023-01-31T00:00:00")
    freq = "7D"
    return pd.date_range(start=start_time, end=end_time, freq=freq)


def test_snapshot_weightings_with_timedelta(time_index):
    n = pypsa.Network()

    hours_per_step = (
        time_index.to_series()
        .diff(periods=1)
        .shift(-1)  # move index forward
        .ffill()  # fill the last value (assume same as the one before)
        .apply(lambda x: x.total_seconds() / 3600)
    )

    df_true = pd.DataFrame(dict.fromkeys(n.snapshot_weightings.columns, hours_per_step))
    df_true.index.name = "snapshot"

    n.set_snapshots(time_index, weightings_from_timedelta=True)
    df_actual = n.snapshot_weightings

    pd.testing.assert_frame_equal(df_true, df_actual)


def test_default_snapshot_weightings(time_index):
    n = pypsa.Network()

    weightings = pd.Series(2.0, index=time_index)
    df_true = pd.DataFrame(dict.fromkeys(n.snapshot_weightings.columns, weightings))
    df_true.index.name = "snapshot"

    n.set_snapshots(time_index, default_snapshot_weightings=2.0)
    df_actual = n.snapshot_weightings

    pd.testing.assert_frame_equal(df_true, df_actual)


@pytest.mark.parametrize("calendar", ["noleap", "360_day"])
def test_snapshot_weightings_with_timedelta_cftime(calendar):
    """See GH#1090 — CFTimeIndex must be treated as datetime-like."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("cftime")

    sns = xr.date_range(
        "2015-01-01", periods=4, freq="2h", use_cftime=True, calendar=calendar
    )
    n = pypsa.Network()
    n.set_snapshots(sns, weightings_from_timedelta=True)

    # Calendar metadata must survive set_snapshots.
    from xarray import CFTimeIndex

    assert isinstance(n.snapshots, CFTimeIndex)
    assert n.snapshots.calendar == calendar

    # 2h freq → 2.0 hours per step.
    assert (n.snapshot_weightings["objective"] == 2.0).all()
