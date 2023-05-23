# -*- coding: utf-8 -*-
import pandas as pd

import pypsa


def test_snapshot_weightings_with_timedelta():
    n = pypsa.Network()

    start_time = pd.Timestamp("2023-01-01T00:00:00")
    end_time = pd.Timestamp("2023-01-31T00:00:00")
    freq = "7D"
    time_index = pd.date_range(start=start_time, end=end_time, freq=freq)

    hours_per_step = (
        time_index.to_series()
        .diff(periods=1)
        .shift(-1)  # move index forward
        .ffill()  # fill the last value (assume same as the one before)
        .apply(lambda x: x.total_seconds() / 3600)
    )

    df_true = pd.DataFrame({c: hours_per_step for c in n.snapshot_weightings.columns})

    n.set_snapshots(time_index, weightings_from_timedelta=True)
    df_actual = n.snapshot_weightings

    pd.testing.assert_frame_equal(df_true, df_actual)
