# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Endogenous technology learning for PyPSA optimisation problems.

This module adds an *endogenous learning* (learning-by-doing) formulation to the
linopy-based optimisation. The specific investment cost of a carrier decreases as
its cumulative installed capacity grows, following a one-factor learning curve

    c(E) = initial_cost * (E / global_capacity) ** (-alpha)

with the learning exponent ``alpha = log2(1 / (1 - learning_rate))``. The
non-linear cumulative cost (the integral of ``c``) is approximated by a piecewise
linear curve using linopy's :meth:`linopy.Model.add_piecewise_formulation`. The
breakpoints follow Barreto's dynamic-segment scheme (the cumulative cost increase
doubles from one segment to the next), which resolves the steep early part of the
curve with relatively few segments.

Two cost variants are supported:

* ``time_delay=False`` (default): the build in period *a* is priced as the chord
  integral ``TC(E[a]) - TC(E[a-1])`` of the cumulative cost curve.
* ``time_delay=True``: the build in period *a* is priced at the *start-of-period*
  specific cost ``c(E[a-1])``, i.e. learning only benefits future periods. This is
  linearised with per-segment build variables and is the formulation that spreads
  investments out gradually over time.

The feature is activated via ``n.optimize(..., learning=True)`` (or
``create_model(..., learning=True)``) and configured through the carrier
attributes ``learning_rate``, ``initial_cost``, ``global_capacity`` and
``max_capacity``. Assets whose carrier has learning enabled should have a
``capital_cost`` of zero, since their investment cost is provided by the learning
curve instead.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from pypsa.descriptors import nominal_attrs

if TYPE_CHECKING:
    from pypsa import Network

logger = logging.getLogger(__name__)


def get_learning_carriers(n: Network) -> pd.Index:
    """Return the index of carriers for which endogenous learning is enabled."""
    if "learning_rate" not in n.carriers.columns:
        return pd.Index([])
    rate = n.carriers["learning_rate"].fillna(0.0)
    return n.carriers.index[rate > 0]


def _alpha(learning_rate: float) -> float:
    """Learning exponent ``alpha`` corresponding to a given learning rate."""
    return float(np.log2(1.0 / (1.0 - learning_rate)))


def _cumulative_cost(
    e: np.ndarray | float, alpha: float, c0: float, e0: float
) -> np.ndarray | float:
    """Cumulative investment cost (integral of the learning curve) from 0 to ``e``.

    ``TC(E) = c0 * e0**alpha * E**(1 - alpha) / (1 - alpha)``.
    """
    return c0 * e0**alpha * np.asarray(e, dtype=float) ** (1.0 - alpha) / (1.0 - alpha)


def _learning_breakpoints(
    learning_rate: float, c0: float, e0: float, e_max: float, segments: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Barreto-doubling breakpoints of the cumulative cost curve.

    Returns the cumulative-capacity breakpoints ``E``, the cumulative-cost
    breakpoints ``TC`` and the per-segment slope (specific cost) ``slope``.
    """
    alpha = _alpha(learning_rate)
    if alpha >= 1.0:
        msg = (
            "Endogenous learning currently only supports learning rates below 50% "
            f"(got {learning_rate:.3f}, which implies a learning exponent >= 1)."
        )
        raise NotImplementedError(msg)

    tc_low = _cumulative_cost(e0, alpha, c0, e0)
    tc_high = _cumulative_cost(e_max, alpha, c0, e0)
    span = tc_high - tc_low

    # cumulative cost increase doubles from segment to segment (Barreto 2001)
    d_tc = [span * (2**j) / (2**segments - 1) for j in range(segments)]
    tc_break = tc_low + np.cumsum([0.0] + d_tc)

    # invert TC(E) analytically to get the matching capacity breakpoints
    pref = c0 * e0**alpha / (1.0 - alpha)
    e_break = (tc_break / pref) ** (1.0 / (1.0 - alpha))
    e_break[0] = e0  # guard against floating point at the lower bound

    slope = (tc_break[1:] - tc_break[:-1]) / (e_break[1:] - e_break[:-1])
    return e_break, tc_break, slope


def _new_capacity_per_period(n: Network, carrier: str, periods: pd.Index) -> dict:
    """Map each investment period to a linopy expression of newly built capacity.

    The new capacity in period ``a`` is the sum of the nominal capacity variables
    of all extendable assets of ``carrier`` whose ``build_year`` equals ``a``.
    """
    m = n.model
    expr: dict = dict.fromkeys(periods)
    for c, attr in nominal_attrs.items():
        comp = n.components[c]
        static = comp.static
        if static.empty or "carrier" not in static.columns:
            continue
        assets = comp.extendables.intersection(
            static.index[static["carrier"] == carrier]
        )
        if assets.empty:
            continue
        var = m[f"{c}-{attr}"]
        for name in assets:
            build_year = static.at[name, "build_year"]
            if build_year not in expr:
                continue
            term = var.sel(name=name)
            expr[build_year] = (
                term if expr[build_year] is None else expr[build_year] + term
            )
    return expr


def _active_period_weight(n: Network, carrier: str, periods: pd.Index) -> dict:
    """Discounted sum of active investment-period weightings per build year.

    For capacity built in period ``a`` this returns the sum of the objective
    investment-period weightings over all periods in which that capacity is still
    active (i.e. within its lifetime). This is the same weighting that PyPSA's
    standard objective applies to the annualised capital cost of an asset.
    """
    weighting = n.investment_period_weightings["objective"]
    lifetimes: dict = {}
    for c in nominal_attrs:
        static = n.components[c].static
        if static.empty or "carrier" not in static.columns:
            continue
        sub = static[static["carrier"] == carrier]
        for build_year, lt in zip(sub["build_year"], sub["lifetime"], strict=True):
            lifetimes.setdefault(build_year, lt)

    weight: dict = {}
    for a in periods:
        lt = lifetimes.get(a)
        if lt is None:
            weight[a] = 0.0
            continue
        active = [p for p in periods if a <= p < a + lt]
        weight[a] = float(weighting.loc[active].sum())
    return weight


def define_learning(
    n: Network,
    sns: pd.Index,
    segments: int = 5,
    time_delay: bool = False,
) -> None:
    """Add endogenous technology learning to the linopy model.

    Parameters
    ----------
    n : pypsa.Network
        Network whose ``n.model`` has already been built.
    sns : pandas.Index
        Snapshots of the optimisation (unused directly, kept for symmetry with
        other optimisation routines).
    segments : int, default 5
        Number of line segments used for the piecewise linearisation of the
        cumulative cost curve.
    time_delay : bool, default False
        If True, price each period's build at the start-of-period specific cost
        ``c(E[a-1])`` (learning benefits only future periods). If False, use the
        chord integral ``TC(E[a]) - TC(E[a-1])``.

    """
    m = n.model
    learn_i = get_learning_carriers(n)
    if learn_i.empty:
        logger.warning(
            "`learning=True` but no carrier has a positive `learning_rate`; "
            "skipping endogenous learning."
        )
        return

    if not n._multi_invest:
        msg = "Endogenous learning requires `multi_investment_periods=True`."
        raise NotImplementedError(msg)

    periods = pd.Index(n.investment_periods, name="period")
    extra_cost = None  # accumulated learning capex to add to the objective

    for carrier in learn_i:
        params = n.carriers.loc[carrier]
        learning_rate = float(params["learning_rate"])
        c0 = float(params["initial_cost"])
        e0 = float(params["global_capacity"])
        e_max = float(params["max_capacity"])

        if e0 <= 0 or e_max <= e0:
            msg = (
                f"Carrier '{carrier}' needs 0 < global_capacity < max_capacity for "
                f"endogenous learning (got global_capacity={e0}, max_capacity={e_max})."
            )
            raise ValueError(msg)

        e_break, tc_break, slope = _learning_breakpoints(
            learning_rate, c0, e0, e_max, segments
        )
        new_cap = _new_capacity_per_period(n, carrier, periods)
        weight = _active_period_weight(n, carrier, periods)

        # cumulative installed capacity E[a] = e0 + sum_{a'<=a} Q[a']
        E = m.add_variables(
            lower=e0, upper=e_max, coords=[periods], name=f"learning-{carrier}-E"
        )
        for i, a in enumerate(periods):
            lhs = E.sel(period=a)
            if new_cap[a] is not None:
                lhs = lhs - new_cap[a]
            if i == 0:
                m.add_constraints(lhs == e0, name=f"learning-{carrier}-E_track-{a}")
            else:
                m.add_constraints(
                    lhs - E.sel(period=periods[i - 1]) == 0,
                    name=f"learning-{carrier}-E_track-{a}",
                )

        if not time_delay:
            # ----- chord-integral pricing via piecewise linear cumulative cost ---
            TC = m.add_variables(
                lower=float(tc_break[0]),
                coords=[periods],
                name=f"learning-{carrier}-TC",
            )
            m.add_piecewise_formulation(
                (E, list(e_break)),
                (TC, list(tc_break)),
                name=f"learning-{carrier}-pwl",
            )
            for i, a in enumerate(periods):
                if i == 0:
                    inv = TC.sel(period=a) - float(tc_break[0])
                else:
                    inv = TC.sel(period=a) - TC.sel(period=periods[i - 1])
                term = weight[a] * inv
                extra_cost = term if extra_cost is None else extra_cost + term
        else:
            # ----- time-delay pricing at the previous period's specific cost -----
            seg = pd.Index(range(segments), name="seg")
            y = m.add_variables(
                binary=True, coords=[periods, seg], name=f"learning-{carrier}-y"
            )
            lam_l = m.add_variables(
                lower=0.0, coords=[periods, seg], name=f"learning-{carrier}-lamL"
            )
            lam_r = m.add_variables(
                lower=0.0, coords=[periods, seg], name=f"learning-{carrier}-lamR"
            )
            m.add_constraints(y.sum("seg") == 1, name=f"learning-{carrier}-y_sum")
            m.add_constraints(lam_l + lam_r - y == 0, name=f"learning-{carrier}-lam_eq")

            e_left = xr.DataArray(e_break[:-1], coords=[seg], dims=["seg"])
            e_right = xr.DataArray(e_break[1:], coords=[seg], dims=["seg"])
            m.add_constraints(
                (lam_l * e_left + lam_r * e_right).sum("seg") - E == 0,
                name=f"learning-{carrier}-E_locate",
            )

            future = periods[1:]
            x_diff = m.add_variables(
                lower=0.0, coords=[future, seg], name=f"learning-{carrier}-x_diff"
            )
            for i, a in enumerate(future, start=1):
                prev = periods[i - 1]
                build_sum = x_diff.sel(period=a).sum("seg")
                if new_cap[a] is not None:
                    build_sum = build_sum - new_cap[a]
                m.add_constraints(
                    build_sum == 0, name=f"learning-{carrier}-xdiff_sum-{a}"
                )
                m.add_constraints(
                    x_diff.sel(period=a) - e_max * y.sel(period=prev) <= 0,
                    name=f"learning-{carrier}-xdiff_ub-{a}",
                )

            # first period priced at the initial specific cost c0
            if new_cap[periods[0]] is not None:
                term = weight[periods[0]] * c0 * new_cap[periods[0]]
                extra_cost = term if extra_cost is None else extra_cost + term

            slope_weight = xr.DataArray(
                [[slope[j] * weight[a] for j in range(segments)] for a in future],
                coords=[future, seg],
                dims=["period", "seg"],
            )
            term = (x_diff * slope_weight).sum()
            extra_cost = term if extra_cost is None else extra_cost + term

    if extra_cost is not None:
        m.objective = m.objective + extra_cost
