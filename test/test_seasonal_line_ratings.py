# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for Lines.apply_seasonal_rating."""

import numpy as np
import pandas as pd
import pytest

import pypsa

SUMMER_MONTHS = (4, 5, 6, 7, 8, 9)
SH_SUMMER_MONTHS = (10, 11, 12, 1, 2, 3)


def _build_network(periods: int = 8760) -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2025-01-01", periods=periods, freq="h"))
    n.add("Bus", ["a", "b"])
    n.add("Line", "a-b", bus0="a", bus1="b", x=0.1, r=0.01, s_nom=1000)
    return n


@pytest.fixture
def network() -> pypsa.Network:
    """Fresh network per test, for cases that mutate ``s_max_pu`` in place."""
    return _build_network()


@pytest.fixture(scope="module")
def base_network() -> pypsa.Network:
    """Shared network for validation tests that raise before any mutation."""
    return _build_network()


@pytest.mark.parametrize(
    ("summer", "winter", "summer_months", "summer_pu", "winter_pu"),
    [
        pytest.param(800, 1000, SUMMER_MONTHS, 0.8, 1.0, id="summer-lower"),
        pytest.param(1000, 800, SUMMER_MONTHS, 1.0, 0.8, id="winter-lower"),
        pytest.param(900, 900, SUMMER_MONTHS, 0.9, 0.9, id="equal-ratings"),
        pytest.param(1200, 1000, SUMMER_MONTHS, 1.2, 1.0, id="rating-above-s_nom"),
        pytest.param(800, 1000, SH_SUMMER_MONTHS, 0.8, 1.0, id="southern-hemisphere"),
    ],
)
def test_seasonal_scaling(network, summer, winter, summer_months, summer_pu, winter_pu):
    """Each season scales s_max_pu by rating / s_nom; s_nom stays put."""
    ratings = pd.DataFrame({"summer": [summer], "winter": [winter]}, index=["a-b"])
    network.c.lines.apply_seasonal_rating(ratings, summer_months=summer_months)

    # s_nom is never mutated in place.
    assert float(network.lines.at["a-b", "s_nom"]) == 1000.0

    s = network.lines_t.s_max_pu["a-b"]
    in_summer = s.index.month.isin(summer_months)
    assert np.allclose(s[in_summer], summer_pu)
    assert np.allclose(s[~in_summer], winter_pu)


@pytest.mark.parametrize(
    ("compose", "summer_pu", "winter_pu"),
    [
        pytest.param(True, 0.72, 0.9, id="compose-preserves-n1-margin"),
        pytest.param(False, 0.8, 1.0, id="overwrite-ignores-margin"),
    ],
)
def test_seasonal_compose_with_static_margin(network, compose, summer_pu, winter_pu):
    """compose=True multiplies a pre-existing static N-1 margin; False overwrites."""
    network.lines.at["a-b", "s_max_pu"] = 0.9
    ratings = pd.DataFrame({"summer": [800], "winter": [1000]}, index=["a-b"])
    network.c.lines.apply_seasonal_rating(ratings, compose=compose)

    s = network.lines_t.s_max_pu["a-b"]
    in_summer = s.index.month.isin(SUMMER_MONTHS)
    assert np.allclose(s[in_summer], summer_pu)
    assert np.allclose(s[~in_summer], winter_pu)


def test_seasonal_compose_with_existing_dynamic_series(network):
    """A pre-existing per-snapshot s_max_pu composes element-wise."""
    pre = pd.Series(
        np.where(network.snapshots.hour < 12, 0.9, 0.8), index=network.snapshots
    )
    network.lines_t.s_max_pu["a-b"] = pre
    ratings = pd.DataFrame({"summer": [800], "winter": [1000]}, index=["a-b"])
    network.c.lines.apply_seasonal_rating(ratings, compose=True)

    s = network.lines_t.s_max_pu["a-b"]
    in_summer = s.index.month.isin(SUMMER_MONTHS)
    morning_summer = s[(s.index.hour < 12) & in_summer]
    afternoon_winter = s[(s.index.hour >= 12) & ~in_summer]
    assert np.allclose(morning_summer, 0.9 * 0.8)  # 0.72
    assert np.allclose(afternoon_winter, 0.8 * 1.0)  # 0.80


def test_seasonal_partial_subset_leaves_others_untouched(network):
    """Lines absent from `ratings` keep their s_nom and gain no s_max_pu column."""
    network.add("Bus", "c")
    network.add("Line", "b-c", bus0="b", bus1="c", x=0.1, s_nom=500)
    ratings = pd.DataFrame({"summer": [800], "winter": [1000]}, index=["a-b"])
    network.c.lines.apply_seasonal_rating(ratings)

    assert float(network.lines.at["a-b", "s_nom"]) == 1000.0
    assert float(network.lines.at["b-c", "s_nom"]) == 500.0
    assert "b-c" not in network.lines_t.s_max_pu.columns


@pytest.mark.parametrize(
    ("ratings", "kwargs", "exc", "match"),
    [
        pytest.param(
            pd.DataFrame({"summer": [800]}, index=["a-b"]),
            {},
            KeyError,
            "winter",
            id="missing-column",
        ),
        pytest.param(
            pd.DataFrame({"summer": [800], "winter": [1000]}, index=["nope"]),
            {},
            KeyError,
            "not in n.lines.index",
            id="unknown-line",
        ),
        pytest.param(
            pd.DataFrame({"summer": [0], "winter": [1000]}, index=["a-b"]),
            {},
            ValueError,
            "positive",
            id="nonpositive-rating",
        ),
        pytest.param(
            pd.DataFrame({"summer": [800], "winter": [1000]}, index=["a-b"]),
            {"summer_months": ()},
            ValueError,
            "non-empty",
            id="empty-summer-months",
        ),
    ],
)
def test_seasonal_rejects_invalid_input(base_network, ratings, kwargs, exc, match):
    """Input validation raises before touching the (shared) network."""
    with pytest.raises(exc, match=match):
        base_network.c.lines.apply_seasonal_rating(ratings, **kwargs)


def test_seasonal_rejects_non_datetime_snapshots():
    n = pypsa.Network()
    n.set_snapshots(range(24))
    n.add("Bus", ["a", "b"])
    n.add("Line", "a-b", bus0="a", bus1="b", x=0.1, s_nom=1000)
    ratings = pd.DataFrame({"summer": [800], "winter": [1000]}, index=["a-b"])
    with pytest.raises(TypeError, match="DatetimeIndex"):
        n.c.lines.apply_seasonal_rating(ratings)


def test_seasonal_rejects_nonpositive_s_nom(network):
    """Scaling by rating / s_nom requires a strictly positive s_nom."""
    network.lines.at["a-b", "s_nom"] = 0.0
    ratings = pd.DataFrame({"summer": [800], "winter": [1000]}, index=["a-b"])
    with pytest.raises(ValueError, match="s_nom"):
        network.c.lines.apply_seasonal_rating(ratings)


def test_seasonal_solve_runs():
    """End-to-end: a network with seasonal ratings still solves."""
    n = _build_network(periods=48)
    n.add("Generator", "g", bus="a", p_nom=500, marginal_cost=10)
    n.add("Load", "ld", bus="b", p_set=400)
    ratings = pd.DataFrame({"summer": [800], "winter": [1000]}, index=["a-b"])
    n.c.lines.apply_seasonal_rating(ratings)
    status, condition = n.optimize()
    assert status == "ok"
    assert condition == "optimal"
