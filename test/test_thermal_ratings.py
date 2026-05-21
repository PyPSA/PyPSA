# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for pypsa.utils.thermal_ratings."""

import numpy as np
import pandas as pd
import pytest

import pypsa
from pypsa.utils import apply_seasonal_line_ratings


def _make_network(periods: int = 8760) -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2025-01-01", periods=periods, freq="h"))
    n.add("Bus", ["a", "b"])
    n.add(
        "Line",
        "a-b",
        bus0="a",
        bus1="b",
        x=0.1,
        r=0.01,
        s_nom=1000,
    )
    return n


def test_apply_seasonal_basic_summer_derating():
    """summer < winter → s_nom = winter; summer factor = summer/winter."""
    n = _make_network()
    ratings = pd.DataFrame({"summer": [800], "winter": [1000]}, index=["a-b"])
    apply_seasonal_line_ratings(n, ratings)

    assert float(n.lines.at["a-b", "s_nom"]) == 1000.0

    s = n.lines_t.s_max_pu["a-b"]
    summer_hours = s[s.index.month.isin([4, 5, 6, 7, 8, 9])]
    winter_hours = s[~s.index.month.isin([4, 5, 6, 7, 8, 9])]
    assert np.allclose(summer_hours, 0.8)
    assert np.allclose(winter_hours, 1.0)


def test_apply_seasonal_winter_lower_than_summer():
    """winter < summer is allowed (data-driven); envelope still works."""
    n = _make_network()
    ratings = pd.DataFrame({"summer": [1000], "winter": [800]}, index=["a-b"])
    apply_seasonal_line_ratings(n, ratings)

    # s_nom = max(summer, winter) = 1000, factor = min/max = 0.8 in summer.
    assert float(n.lines.at["a-b", "s_nom"]) == 1000.0
    s = n.lines_t.s_max_pu["a-b"]
    assert np.allclose(s[s.index.month.isin([4, 5, 6, 7, 8, 9])], 0.8)


def test_apply_seasonal_equal_ratings_no_broadcast():
    """When summer == winter, factor = 1.0 throughout."""
    n = _make_network()
    ratings = pd.DataFrame({"summer": [1500], "winter": [1500]}, index=["a-b"])
    apply_seasonal_line_ratings(n, ratings)

    assert float(n.lines.at["a-b", "s_nom"]) == 1500.0
    assert np.allclose(n.lines_t.s_max_pu["a-b"], 1.0)


def test_apply_seasonal_southern_hemisphere():
    """summer_months override flips the broadcast."""
    n = _make_network()
    ratings = pd.DataFrame({"summer": [800], "winter": [1000]}, index=["a-b"])
    apply_seasonal_line_ratings(
        n, ratings, summer_months=(10, 11, 12, 1, 2, 3)
    )

    s = n.lines_t.s_max_pu["a-b"]
    nh_summer = s[s.index.month.isin([4, 5, 6, 7, 8, 9])]
    sh_summer = s[s.index.month.isin([10, 11, 12, 1, 2, 3])]
    assert np.allclose(nh_summer, 1.0)
    assert np.allclose(sh_summer, 0.8)


def test_apply_seasonal_compose_multiplies_with_existing_s_max_pu():
    """compose=True preserves a pre-existing N-1 margin via multiplication."""
    n = _make_network()
    # Pre-existing N-1 margin = 0.9 (constant).
    n.lines.at["a-b", "s_max_pu"] = 0.9
    ratings = pd.DataFrame({"summer": [800], "winter": [1000]}, index=["a-b"])
    apply_seasonal_line_ratings(n, ratings, compose=True)

    s = n.lines_t.s_max_pu["a-b"]
    # Summer: 0.9 × 0.8 = 0.72; Winter: 0.9 × 1.0 = 0.9.
    assert np.allclose(s[s.index.month.isin([4, 5, 6, 7, 8, 9])], 0.72)
    assert np.allclose(s[~s.index.month.isin([4, 5, 6, 7, 8, 9])], 0.9)


def test_apply_seasonal_compose_false_overwrites():
    """compose=False overwrites; pre-existing s_max_pu is ignored."""
    n = _make_network()
    n.lines.at["a-b", "s_max_pu"] = 0.9
    ratings = pd.DataFrame({"summer": [800], "winter": [1000]}, index=["a-b"])
    apply_seasonal_line_ratings(n, ratings, compose=False)

    s = n.lines_t.s_max_pu["a-b"]
    assert np.allclose(s[s.index.month.isin([4, 5, 6, 7, 8, 9])], 0.8)
    assert np.allclose(s[~s.index.month.isin([4, 5, 6, 7, 8, 9])], 1.0)


def test_apply_seasonal_compose_with_existing_dynamic_series():
    """Pre-existing per-snapshot s_max_pu composes element-wise."""
    n = _make_network()
    # Half-and-half pre-existing dynamic margin.
    pre = pd.Series(
        np.where(n.snapshots.hour < 12, 0.9, 0.8), index=n.snapshots
    )
    n.lines_t.s_max_pu["a-b"] = pre
    ratings = pd.DataFrame({"summer": [800], "winter": [1000]}, index=["a-b"])
    apply_seasonal_line_ratings(n, ratings, compose=True)

    s = n.lines_t.s_max_pu["a-b"]
    summer_mask = s.index.month.isin([4, 5, 6, 7, 8, 9])

    # Spot-check: morning summer = 0.9 × 0.8 = 0.72
    morning_summer = s[(s.index.hour < 12) & summer_mask]
    afternoon_winter = s[(s.index.hour >= 12) & ~summer_mask]
    assert np.allclose(morning_summer, 0.72)
    assert np.allclose(afternoon_winter, 0.8)


def test_apply_seasonal_rejects_non_datetime_snapshots():
    n = pypsa.Network()
    n.set_snapshots(range(24))  # integer index, not datetime
    n.add("Bus", ["a", "b"])
    n.add("Line", "a-b", bus0="a", bus1="b", x=0.1, s_nom=1000)
    ratings = pd.DataFrame({"summer": [800], "winter": [1000]}, index=["a-b"])
    with pytest.raises(TypeError, match="DatetimeIndex"):
        apply_seasonal_line_ratings(n, ratings)


def test_apply_seasonal_rejects_missing_columns():
    n = _make_network()
    bad = pd.DataFrame({"summer": [800]}, index=["a-b"])  # no 'winter'
    with pytest.raises(KeyError, match="winter"):
        apply_seasonal_line_ratings(n, bad)


def test_apply_seasonal_rejects_unknown_lines():
    n = _make_network()
    bad = pd.DataFrame({"summer": [800], "winter": [1000]}, index=["nope"])
    with pytest.raises(KeyError, match="not in n.lines.index"):
        apply_seasonal_line_ratings(n, bad)


def test_apply_seasonal_rejects_nonpositive_ratings():
    n = _make_network()
    bad = pd.DataFrame({"summer": [0], "winter": [1000]}, index=["a-b"])
    with pytest.raises(ValueError, match="positive"):
        apply_seasonal_line_ratings(n, bad)


def test_apply_seasonal_rejects_empty_summer_months():
    n = _make_network()
    ratings = pd.DataFrame({"summer": [800], "winter": [1000]}, index=["a-b"])
    with pytest.raises(ValueError, match="non-empty"):
        apply_seasonal_line_ratings(n, ratings, summer_months=())


def test_apply_seasonal_multi_line_partial_subset():
    """Lines not in `ratings` are untouched."""
    n = _make_network()
    n.add("Bus", "c")
    n.add("Line", "b-c", bus0="b", bus1="c", x=0.1, s_nom=500)
    ratings = pd.DataFrame({"summer": [800], "winter": [1000]}, index=["a-b"])
    apply_seasonal_line_ratings(n, ratings)

    assert float(n.lines.at["a-b", "s_nom"]) == 1000.0
    # Untouched line keeps its original static s_nom; no dynamic entry created.
    assert float(n.lines.at["b-c", "s_nom"]) == 500.0
    assert "b-c" not in n.lines_t.s_max_pu.columns


def test_apply_seasonal_solve_runs():
    """End-to-end: a network with seasonal ratings still solves.

    Depends on the Arrow-string fix from GH #1690 (or a pandas config with
    ``future.infer_string=False``). Until #1690 lands, force the workaround
    locally so this test is self-contained.
    """
    prev = pd.get_option("future.infer_string")
    pd.set_option("future.infer_string", False)
    try:
        n = _make_network(periods=48)  # 2-day horizon
        n.add("Generator", "g", bus="a", p_nom=500, marginal_cost=10)
        n.add("Load", "ld", bus="b", p_set=400)
        ratings = pd.DataFrame(
            {"summer": [800], "winter": [1000]}, index=["a-b"]
        )
        apply_seasonal_line_ratings(n, ratings)
        status, condition = n.optimize()
        assert status == "ok"
        assert condition == "optimal"
    finally:
        pd.set_option("future.infer_string", prev)
