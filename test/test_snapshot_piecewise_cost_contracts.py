# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Public snapshot piecewise cost contracts.

Failure list:
1. Snapshot-varying slopes or availability-scaled positions give wrong costs.
2. Static and dynamic generators cannot coexist in either insertion order.
3. A selected solve window uses curves from other snapshots.
4. Duplicate or missing snapshots silently select ambiguous curves.
5. Interior NaN or unpaired padding creates a false segment.
6. The existing per-unit terminal coordinate of one is lost.
7. NetCDF drops the snapshot dimension or changes a restored solve.
8. Static-only capital cost accidentally acquires a time-varying axis.
9. Investment-period snapshots are silently flattened by the new row schema.
10. Snapshot and component labels are lost during slope integration.

Analytic dispatch, price and objective protect the public contract. The
immutable upstream does not accept this row schema. Existing tests cover
static curves only; this file needs no production seam or mocks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

import pypsa

if TYPE_CHECKING:
    from pathlib import Path


def dynamic_curve() -> pd.DataFrame:
    nan = np.nan
    index = pd.MultiIndex.from_product(
        [[0, 1, 2], range(5)], names=["snapshot", "breakpoint"]
    )
    return pd.DataFrame(
        {
            "p_pu": [0, 0.9, 0.95, 1, nan, 0, 0.45, 0.475, 0.5, 1, 0, 1, nan, nan, nan],
            "marginal_cost": [
                0,
                50,
                56,
                60,
                nan,
                0,
                55,
                61,
                65,
                65,
                0,
                47,
                nan,
                nan,
                nan,
            ],
        },
        index=index,
    )


def network_with_curve(curve: pd.DataFrame | None = None) -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots(pd.Index([0, 1, 2], name="snapshot"))
    n.add("Bus", "bus")
    n.add(
        "Generator",
        "unit",
        bus="bus",
        p_nom=100,
        p_max_pu=[1, 0.5, 0],
        marginal_cost=dynamic_curve() if curve is None else curve,
    )
    n.add("Generator", "backup", bus="bus", p_nom=100, marginal_cost=1000)
    n.add("Load", "load", bus="bus", p_set=[97, 48.5, 10])
    return n


def assert_solve(n: pypsa.Network, snapshots: pd.Index | None = None) -> None:
    selected = n.snapshots if snapshots is None else snapshots
    status, condition = n.optimize(
        snapshots=selected,
        solver_name="highs",
        solver_options={"threads": 1},
        include_objective_constant=False,
    )
    assert (status, condition) == ("ok", "optimal")
    expected_costs = pd.Series([4900, 2692.5, 10000], index=[0, 1, 2])
    assert n.objective == pytest.approx(expected_costs.loc[selected].sum())
    np.testing.assert_allclose(
        n.generators_t.p.loc[selected, "unit"],
        pd.Series([97, 48.5, 0], index=[0, 1, 2]).loc[selected],
    )
    np.testing.assert_allclose(
        n.buses_t.marginal_price.loc[selected, "bus"],
        pd.Series([60, 65, 1000], index=[0, 1, 2]).loc[selected],
    )
    n.model.solver = None


@pytest.mark.parametrize("multi_columns", [False, True])
def test_dynamic_dataframe_preserves_snapshot_storage(multi_columns: bool) -> None:
    curve = dynamic_curve()
    if multi_columns:
        curve = pd.concat({"unit": curve}, axis=1).rename_axis(
            columns=["name", "attribute"]
        )
    n = network_with_curve(curve)
    expected = pd.concat({"unit": dynamic_curve()}, axis=1).rename_axis(
        columns=["name", "attribute"]
    )
    pd.testing.assert_frame_equal(n.c.generators.piecewise["marginal_cost"], expected)


@pytest.mark.parametrize("window", [None, [1]])
def test_dynamic_native_solve_matches_analytic_dispatch_price(
    window: list[int] | None,
) -> None:
    n = network_with_curve()
    assert_solve(n, None if window is None else pd.Index(window, name="snapshot"))


def test_decimal_snapshot_slopes_keep_rounded_tranche_costs() -> None:
    """Base 50.126 rounds by tranche, then a daily delta 5.017 rounds again."""
    curve = dynamic_curve()
    curve.loc[0, "marginal_cost"] = [0, 50.13, 56.14, 60.15, np.nan]
    curve.loc[1, "marginal_cost"] = [0, 55.15, 61.16, 65.17, 65.17]
    curve.loc[(2, 1), "marginal_cost"] = 47.11
    n = network_with_curve(curve)
    status, condition = n.optimize(
        solver_name="highs",
        solver_options={"threads": 1},
        include_objective_constant=False,
    )
    assert (status, condition) == ("ok", "optimal")
    expected = (
        90 * 50.13 + 5 * 56.14 + 2 * 60.15 + 45 * 55.15 + 2.5 * 61.16 + 65.17 + 10000
    )
    assert n.objective == pytest.approx(expected)
    np.testing.assert_allclose(n.buses_t.marginal_price["bus"], [60.15, 65.17, 1000])
    n.model.solver = None


@pytest.mark.parametrize("static_first", [False, True])
def test_mixed_static_dynamic_costs_broadcast_in_either_order(
    static_first: bool,
) -> None:
    n = pypsa.Network()
    n.set_snapshots(pd.Index([0, 1, 2], name="snapshot"))
    n.add("Bus", "bus")
    static = pd.DataFrame({"p_pu": [0, 1], "marginal_cost": [0, 1000]})

    def add_static() -> None:
        n.add("Generator", "backup", bus="bus", p_nom=100, marginal_cost=static)

    def add_dynamic() -> None:
        n.add(
            "Generator",
            "unit",
            bus="bus",
            p_nom=100,
            p_max_pu=[1, 0.5, 0],
            marginal_cost=dynamic_curve(),
        )

    if static_first:
        add_static()
        add_dynamic()
    else:
        add_dynamic()
        add_static()
    n.add("Load", "load", bus="bus", p_set=[97, 48.5, 10])
    assert_solve(n)


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        ("duplicate", "duplicate snapshot/breakpoint"),
        ("missing", "missing network snapshots"),
        ("interior-nan", "non-trailing missing breakpoint"),
        ("unpaired", "incomplete breakpoint data"),
    ],
)
def test_invalid_dynamic_curve_rejected(invalid: str, message: str) -> None:
    curve = dynamic_curve()
    if invalid == "duplicate":
        curve = pd.concat([curve, curve.iloc[[0]]])
    elif invalid == "missing":
        curve = curve.drop(index=1, level="snapshot")
    elif invalid == "interior-nan":
        curve.loc[(0, 1), :] = np.nan
    else:
        curve.loc[(0, 4), "p_pu"] = 0.99
    with pytest.raises(ValueError, match=message):
        network_with_curve(curve).optimize.create_model(
            include_objective_constant=False
        )


def test_dynamic_per_unit_curve_still_requires_terminal_one() -> None:
    curve = dynamic_curve()
    curve.loc[(1, 4), :] = np.nan
    with pytest.raises(ValueError, match="must end at p_pu=1"):
        network_with_curve(curve).optimize.create_model(
            include_objective_constant=False
        )


def test_dynamic_capital_cost_is_rejected_as_static_only() -> None:
    n = pypsa.Network()
    n.set_snapshots(pd.Index([0, 1, 2], name="snapshot"))
    n.add("Bus", "bus")
    curve = dynamic_curve().rename(
        columns={"p_pu": "p_nom", "marginal_cost": "capital_cost"}
    )
    with pytest.raises(ValueError, match="(?i)(capital_cost|static|varying)"):
        n.add("Generator", "unit", bus="bus", p_nom_extendable=True, capital_cost=curve)


def test_dynamic_curve_rejects_investment_period_snapshots_explicitly() -> None:
    n = pypsa.Network()
    n.set_snapshots(
        pd.MultiIndex.from_product([[2025], [0, 1, 2]], names=["period", "timestep"])
    )
    n.add("Bus", "bus")
    with pytest.raises(ValueError, match="(?i)(investment|multiindex|snapshot)"):
        n.add("Generator", "unit", bus="bus", p_nom=100, marginal_cost=dynamic_curve())


def test_dynamic_netcdf_roundtrip_preserves_curve_and_solve(tmp_path: Path) -> None:
    n = network_with_curve()
    path = tmp_path / "snapshot-cost.nc"
    n.export_to_netcdf(path)
    restored = pypsa.Network(path)
    pd.testing.assert_frame_equal(
        restored.c.generators.piecewise["marginal_cost"],
        n.c.generators.piecewise["marginal_cost"],
    )
    assert_solve(restored)


def test_labeled_dynamic_curves_preserve_dispatch_after_netcdf(tmp_path: Path) -> None:
    # PyPSA requires naive snapshots; convert UTC explicitly at that boundary.
    snapshots = (
        pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
        .tz_localize(None)
        .as_unit("ns")
    )
    curve = dynamic_curve()
    curve.index = pd.MultiIndex.from_product(
        [snapshots, [2, 4, 7, 9, 12]], names=["snapshot", "breakpoint"]
    )
    name = "unit: [2] / 01"
    n = pypsa.Network()
    n.set_snapshots(snapshots)
    n.add("Bus", "bus")
    n.add(
        "Generator", "backup '00'", bus="bus", p_nom=100, marginal_cost={0: 0, 1: 1000}
    )
    n.add(
        "Generator",
        name,
        bus="bus",
        p_nom=100,
        p_max_pu=[1, 0.5, 0],
        marginal_cost=curve,
    )
    n.add("Load", "load", bus="bus", p_set=[97, 48.5, 10])
    path = tmp_path / "labeled-cost.nc"
    n.export_to_netcdf(path)
    restored = pypsa.Network(path)
    status, condition = restored.optimize(
        solver_name="highs",
        solver_options={"threads": 1},
        include_objective_constant=False,
    )
    assert (status, condition) == ("ok", "optimal")
    assert restored.objective == pytest.approx(17592.5)
    pd.testing.assert_index_equal(restored.generators_t.p.index, n.snapshots)
    np.testing.assert_allclose(restored.generators_t.p[name], [97, 48.5, 0])
    np.testing.assert_allclose(restored.generators_t.p["backup '00'"], [0, 0, 10])
    np.testing.assert_allclose(restored.buses_t.marginal_price["bus"], [60, 65, 1000])
    restored.model.solver = None
