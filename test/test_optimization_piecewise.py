# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Unit tests for helpers in ``pypsa.optimization.piecewise``."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pypsa.constants import PIECEWISE_ATTRS
from pypsa.optimization.piecewise import (
    _normalize_breakpoints,
    get_piecewise_names,
)


@pytest.fixture
def gen_marginal_cost_attrs() -> pd.Series:
    return PIECEWISE_ATTRS.query(
        "component == 'Generator' and y == 'marginal_cost'"
    ).squeeze()


def _piecewise_df(
    curves: dict[str, list[tuple[float, float]]],
    x_attr: str = "p_pu",
    y_attr: str = "marginal_cost",
) -> pd.DataFrame:
    """Build a (name, attribute)-columned DataFrame from {name: [(x, y), ...]}."""
    frames = {
        n: pd.DataFrame(rows, columns=[x_attr, y_attr]) for n, rows in curves.items()
    }
    return pd.concat(frames, axis=1, names=["name", "attribute"]).rename_axis(
        index="breakpoint"
    )


class TestNormalizeBreakpoints:
    def test_sorts_unsorted_rows(self, gen_marginal_cost_attrs: pd.Series) -> None:
        df = _piecewise_df({"gen": [(1.0, 40.0), (0.0, 10.0), (0.5, 20.0)]})
        result = _normalize_breakpoints(df, gen_marginal_cost_attrs)
        assert result["gen"]["p_pu"].tolist() == [0.0, 0.5, 1.0]
        assert result["gen"]["marginal_cost"].tolist() == [10.0, 20.0, 40.0]
        assert result.index.name == "breakpoint"

    def test_ragged_curves_aligned_with_trailing_nan(
        self, gen_marginal_cost_attrs: pd.Series
    ) -> None:
        df = _piecewise_df(
            {
                "gen0": [(0.0, 10.0), (0.5, 20.0), (1.0, 40.0)],
                "gen1": [(0.0, 5.0), (1.0, 25.0)],
            }
        )
        result = _normalize_breakpoints(df, gen_marginal_cost_attrs)
        assert len(result) == 3
        assert result["gen0"]["p_pu"].tolist() == [0.0, 0.5, 1.0]
        assert result["gen1"]["p_pu"].iloc[:2].tolist() == [0.0, 1.0]
        assert np.isnan(result["gen1"]["p_pu"].iloc[2])

    def test_idempotent(self, gen_marginal_cost_attrs: pd.Series) -> None:
        df = _piecewise_df({"gen": [(1.0, 40.0), (0.0, 10.0), (0.5, 20.0)]})
        once = _normalize_breakpoints(df, gen_marginal_cost_attrs)
        twice = _normalize_breakpoints(once, gen_marginal_cost_attrs)
        pd.testing.assert_frame_equal(once, twice)

    @pytest.mark.parametrize(
        ("curves", "match"),
        [
            pytest.param(
                {"gen": [(0.0, 10.0), (float("nan"), float("nan")), (1.0, 40.0)]},
                "non-trailing missing breakpoint",
                id="interior-nan-row",
            ),
            pytest.param(
                {"gen": [(0.0, 10.0), (0.5, float("nan")), (1.0, 40.0)]},
                "incomplete breakpoint data",
                id="missing-y",
            ),
            pytest.param(
                {"gen": [(0.0, 10.0), (float("nan"), 20.0), (1.0, 40.0)]},
                "incomplete breakpoint data",
                id="missing-x",
            ),
        ],
    )
    def test_invalid_breakpoints_raise(
        self,
        gen_marginal_cost_attrs: pd.Series,
        curves: dict[str, list[tuple[float, float]]],
        match: str,
    ) -> None:
        df = _piecewise_df(curves)
        with pytest.raises(ValueError, match=match):
            _normalize_breakpoints(df, gen_marginal_cost_attrs)


class TestGetPiecewiseNames:
    def test_missing_attribute_returns_empty(self) -> None:
        c = SimpleNamespace(piecewise={})
        result = get_piecewise_names(c, "marginal_cost", pd.Index(["gen"], name="name"))
        assert result.empty
        assert result.name == "name"

    def test_all_nan_columns_dropped(self) -> None:
        df = _piecewise_df(
            {
                "gen0": [(0.0, 10.0), (1.0, 40.0)],
                "gen1": [(float("nan"), float("nan")), (float("nan"), float("nan"))],
            }
        )
        c = SimpleNamespace(piecewise={"marginal_cost": df})
        active = pd.Index(["gen0", "gen1"], name="name")
        result = get_piecewise_names(c, "marginal_cost", active)
        assert result.tolist() == ["gen0"]

    def test_intersects_with_active_names(self) -> None:
        df = _piecewise_df(
            {
                "gen0": [(0.0, 10.0), (1.0, 40.0)],
                "gen1": [(0.0, 5.0), (1.0, 25.0)],
                "gen2": [(0.0, 3.0), (1.0, 15.0)],
            }
        )
        c = SimpleNamespace(piecewise={"marginal_cost": df})
        active = pd.Index(["gen0", "gen2"], name="name")
        result = get_piecewise_names(c, "marginal_cost", active)
        assert sorted(result.tolist()) == ["gen0", "gen2"]

    @pytest.mark.parametrize(
        ("piecewise", "active"),
        [
            pytest.param({}, pd.Index([], name="name"), id="empty"),
            pytest.param(
                {"marginal_cost": _piecewise_df({"gen": [(0.0, 10.0), (1.0, 40.0)]})},
                pd.Index(["gen"], name="name"),
                id="populated",
            ),
        ],
    )
    def test_result_is_named_name_index(
        self, piecewise: dict, active: pd.Index
    ) -> None:
        c = SimpleNamespace(piecewise=piecewise)
        result = get_piecewise_names(c, "marginal_cost", active)
        assert result.name == "name"
