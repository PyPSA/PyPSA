# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import pandas as pd

import pypsa
from pypsa.optimization.common import _set_dynamic_data


def test_set_dynamic_data_preserves_and_extends_columns():
    """Dynamic data of components added after a previous assignment must not be lost.

    See https://github.com/PyPSA/PyPSA/issues/1723.
    """
    n = pypsa.Network()
    n.set_snapshots(range(3))
    n.add("Bus", "b")
    n.add("Load", ["old", "new"], bus="b")
    n.c.loads.dynamic.p = pd.DataFrame({"old": 1.0}, index=n.snapshots)

    df = pd.DataFrame({"old": 3.0, "new": 2.0}, index=n.snapshots)
    _set_dynamic_data(n, "Load", "p", df)

    pd.testing.assert_frame_equal(
        n.c.loads.dynamic.p, df, check_names=False, check_column_type=False
    )


def test_set_dynamic_data_updates_window_only():
    n = pypsa.Network()
    n.set_snapshots(range(4))
    n.add("Bus", "b")
    n.add("Load", ["a", "b"], bus="b")
    n.c.loads.dynamic.p = pd.DataFrame(1.0, index=n.snapshots, columns=["a", "b"])

    df = pd.DataFrame({"a": [2.0, float("nan")]}, index=n.snapshots[1:3])
    _set_dynamic_data(n, "Load", "p", df)

    expected = pd.DataFrame({"a": [1.0, 2.0, 1.0, 1.0], "b": 1.0}, index=n.snapshots)
    pd.testing.assert_frame_equal(
        n.c.loads.dynamic.p, expected, check_names=False, check_column_type=False
    )
