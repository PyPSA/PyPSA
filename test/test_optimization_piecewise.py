# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Unit tests for helpers in ``pypsa.optimization.piecewise``."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from pypsa.optimization.piecewise import get_piecewise_names


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


class TestGetPiecewiseNames:
    def test_missing_attribute_returns_empty(self) -> None:
        c = SimpleNamespace(
            piecewise={},
        )
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
        c = SimpleNamespace(
            piecewise={"marginal_cost": df},
        )
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
        c = SimpleNamespace(
            piecewise={"marginal_cost": df},
        )
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
        c = SimpleNamespace(
            piecewise=piecewise,
        )
        result = get_piecewise_names(c, "marginal_cost", active)
        assert result.name == "name"
