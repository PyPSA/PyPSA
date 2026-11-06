# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Rolling horizon optimization on windows of one full-horizon linopy model."""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
from linopy import Model

from pypsa._options import options
from pypsa.common import as_index

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    import pandas as pd
    from linopy.restrict import WindowModel

    from pypsa import Network

logger = logging.getLogger(__name__)

LINOPY_REQUIREMENT = (
    "`method='restrict'` requires linopy >= 1.0.0, which provides `Model.restrict` "
    "and `WindowModel.solve_fixed_lp`. Until it is released, install linopy from "
    "its master branch."
)
UNSUPPORTED_KWARGS = {"compute_infeasibilities"} | (
    set(inspect.signature(Model.solve).parameters)
    - {"self", "solver_name", "io_api", "log_fn"}
)


def _check_scope(n: Network, multi_investment_periods: bool) -> None:
    """Raise for network features that windows of one model cannot reproduce."""
    found = []
    if multi_investment_periods:
        found.append("multi-period investment")
    if n.has_risk_preference:
        found.append("risk preference (CVaR)")
    if not n.c.global_constraints.static.empty:
        found.append("global constraints")
    if n.c.stores.static.e_cyclic.any():
        found.append("cyclic stores (`e_cyclic`)")
    if n.c.storage_units.static.cyclic_state_of_charge.any():
        found.append("cyclic storage units (`cyclic_state_of_charge`)")
    found.extend(
        f"volume limits of {c.name} (`e_sum_min`, `e_sum_max`)"
        for c in n.components
        if "e_sum_min" in c.static
        and np.isfinite(c.static[["e_sum_min", "e_sum_max"]]).any().any()
    )
    if found:
        msg = (
            "`method='restrict'` does not support " + ", ".join(found) + ". Use "
            "`method='rebuild'` instead."
        )
        raise NotImplementedError(msg)


def _write(
    target: dict[str, xr.DataArray], window: dict[str, xr.DataArray], sns: pd.Index
) -> None:
    """Write the window values of each container into its full-horizon array."""
    for name, values in window.items():
        if "snapshot" in values.dims:
            target[name].loc[{"snapshot": sns}] = values
        else:
            target[name] = values


def _check_window(w: WindowModel, sns: pd.Index, incumbent: bool) -> str:
    """Return the termination condition, raise unless the window is usable.

    An optimal solution is always usable, an `incumbent` of a solve stopped by
    a limit only if allowed, with a warning.
    """
    condition = w.termination_condition
    if condition == "optimal":
        return condition
    if incumbent and w.status == "ok" and w.objective.value is not None:
        logger.warning(
            "Optimization of window [%s:%s] stopped with condition %s, using "
            "its best solution.",
            sns[0],
            sns[-1],
            condition,
        )
        return condition
    msg = (
        f"Optimization of window [{sns[0]}:{sns[-1]}] failed with status "
        f"{w.status} and condition {condition}."
    )
    raise RuntimeError(msg)


def optimize_with_restricted_windows(
    n: Network,
    snapshots: Sequence,
    horizon: int,
    overlap: int,
    *,
    model_kwargs: dict | None = None,
    extra_functionality: Callable | None = None,
    assign_all_duals: bool = False,
    solver_name: str | None = None,
    solver_options: dict | None = None,
    log_to_console: bool | None = None,
    log_fn: str | Path | None = None,
    io_api: str | None = None,
    window_callback: Callable[[Network, WindowModel, pd.Index, int], Any] | None = None,
    **kwargs: Any,
) -> None:
    """Optimize the network in windows of a single full-horizon model.

    The linopy model is built once for all `snapshots`. Each window is cut from
    its matrices with `linopy.Model.restrict` and solved without warm start,
    with all variables of earlier windows fixed to their optimized values. The
    windows are written back to one full-horizon solution, which is assigned to
    the network once at the end.

    Linear windows must be solved to optimality. Mixed-integer windows also
    accept an incumbent of a solve stopped by a limit, with a warning, and
    `n.model.termination_condition` then reports the condition of the last
    such window. Their integer values are rounded before they are fixed for
    later windows, and duals are taken from a re-solve of the window as an LP
    with its integer variables fixed.

    Parameters
    ----------
    n : pypsa.Network
        Network to optimize.
    snapshots : list-like
        Snapshots to optimize.
    horizon : int
        Number of snapshots per window.
    overlap : int
        Number of snapshots shared by two consecutive windows. The later window
        overwrites the shared snapshots.
    model_kwargs : dict, optional
        Keyword arguments for `linopy.Model()`.
    extra_functionality : callable, optional
        Called once as `extra_functionality(n, snapshots)` on the full model.
    assign_all_duals : bool, default False
        Whether to assign all dual values, see `n.optimize.assign_duals`.
    solver_name : str, optional
        Solver with linopy's direct API. Defaults to the module wide option.
    solver_options : dict, optional
        Solver options. Defaults to the module wide option.
    log_to_console : bool, optional
        Whether the solver prints its progress, passed as a solver option.
    log_fn : str or Path, optional
        Solver log file, passed to the solve of every window.
    io_api : str, optional
        Only `"direct"` is supported.
    window_callback : callable, optional
        Called as `window_callback(n, w, sns, i)` with the
        `linopy.restrict.WindowModel` `w` of the `i`-th window over snapshots
        `sns`, once before its solve (`w.status == "initialized"`) and once
        after, before its status is checked. Use it e.g. to log the status,
        write the window with `w.to_file`, compute an IIS with
        `w.compute_infeasibilities` or, before the solve only, leave rows out
        of this window with `w.advance(drop=...)`.
    **kwargs : Any
        Arguments of `n.optimize.create_model`, such as `transmission_losses`.
        Any other argument is passed to the solver as an option.

    Raises
    ------
    ImportError
        If the installed linopy is older than 1.0.0.
    NotImplementedError
        If the network or the arguments are out of scope.
    RuntimeError
        If a window is not solved to optimality, or for mixed-integer windows,
        if the solver returns no solution.

    """
    try:
        from linopy.restrict import RestrictionError, WindowModel  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(LINOPY_REQUIREMENT) from e
    if not hasattr(WindowModel, "solve_fixed_lp"):
        raise ImportError(LINOPY_REQUIREMENT)

    unsupported = sorted(UNSUPPORTED_KWARGS.intersection(kwargs))
    if io_api not in (None, "direct"):
        unsupported.append(f"io_api={io_api!r}")
    if unsupported:
        msg = f"`method='restrict'` solves through linopy's direct API and does not support {unsupported}."
        raise NotImplementedError(msg)

    model_args = set(inspect.signature(n.optimize.create_model).parameters) - {"kwargs"}
    build = {k: kwargs.pop(k) for k in model_args.intersection(kwargs)}
    _check_scope(n, build.get("multi_investment_periods", False))

    solver_name = solver_name or options.params.optimize.solver_name
    solver_options = {
        **(
            options.params.optimize.solver_options
            if solver_options is None
            else solver_options
        ),
        **kwargs,
    }
    if log_to_console is None:
        log_to_console = options.params.optimize.log_to_console
    if log_to_console is not None:
        solver_options["log_to_console"] = log_to_console
    if model_kwargs is None:
        model_kwargs = options.params.optimize.model_kwargs.copy()

    snapshots = as_index(n, snapshots, "snapshots")
    m = n.optimize.create_model(
        snapshots, consistency_check=False, **build, **model_kwargs
    )
    if extra_functionality:
        extra_functionality(n, snapshots)
    discrete = [*m.variables.integers, *m.variables.binaries]

    primal = {
        k: xr.full_like(v.labels, np.nan, dtype=float) for k, v in m.variables.items()
    }
    dual = {
        k: xr.full_like(c.labels, np.nan, dtype=float) for k, c in m.constraints.items()
    }
    starts = range(0, len(snapshots), horizon - overlap)
    w = None
    condition = "optimal"
    for i, start in enumerate(starts):
        sns = snapshots[start : start + horizon]
        logger.info(
            "Optimizing network for snapshot horizon [%s:%s] (%s/%s).",
            sns[0],
            sns[-1],
            i + 1,
            len(starts),
        )
        try:
            w = (
                m.restrict(snapshot=sns, fix=primal)
                if w is None
                else w.advance(snapshot=sns, fix=primal, drop={})
            )
        except RestrictionError as e:
            msg = (
                f"`method='restrict'` cannot cut window [{sns[0]}:{sns[-1]}] from the "
                "model, e.g. because a constraint couples it to later snapshots. Use "
                f"`method='rebuild'` instead. linopy: {e}"
            )
            raise NotImplementedError(msg) from e
        if window_callback:
            window_callback(n, w, sns, i)
        w.solve(solver_name, log_fn=log_fn, **solver_options)
        if window_callback:
            window_callback(n, w, sns, i)
        if (window_condition := _check_window(w, sns, bool(discrete))) != "optimal":
            condition = window_condition
        if discrete:
            w.solve_fixed_lp(log_fn=log_fn)
            _check_window(w, sns, incumbent=False)
        solution = w.solution
        for k in discrete:
            solution[k] = solution[k].round()
        _write(primal, solution, sns)
        _write(dual, w.dual, sns)

    for k, v in primal.items():
        m.variables[k].solution = v
    for k, v in dual.items():
        m.constraints[k].dual = v
    m.status, m.termination_condition = "ok", condition
    m.objective.set_value(float(m.objective.expression.solution))
    n.optimize.assign_solution()
    n.optimize.assign_duals(assign_all_duals)
    n.optimize.post_processing()
