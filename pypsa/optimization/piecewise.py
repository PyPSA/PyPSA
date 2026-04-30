# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Piecewise-constraint helpers for PyPSA optimization models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
import xarray as xr
from linopy import Model, Variable, breakpoints, piecewise
from linopy.constants import BREAKPOINT_DIM

from pypsa.descriptors import nominal_attrs

if TYPE_CHECKING:
    from collections.abc import Iterable
logger = logging.getLogger(__name__)

OPS_T = Literal["<=", ">=", "=="]
OPS: dict[OPS_T, str] = {"<=": "__le__", ">=": "__ge__", "==": "__eq__"}


@dataclass(eq=True, frozen=True)
class PiecewiseOptions:
    """Options for piecewise constraint formulation."""

    component: str
    attribute: str
    operator: OPS_T
    name: str | None = None
    method: str = "auto"
    marginal_attr: bool = False


def define_piecewise(
    m: Model,
    c: Any,
    x_var: Any,
    seg_attr: str,
    aux_var_name: str,
    active_names: pd.Index,
    operator: OPS_T,
    marginal_attr: bool,
    extra_options: Iterable[PiecewiseOptions],
    y_var: Any | None = None,
    method: str = "auto",
) -> Variable | None:
    """Add variable(s) and constraint(s) necessary to define a constraint along a piecewise linear curve.

    Derives the segment DataFrame, x-axis attribute, component names with segment
    data, and the optional p_nom scale factor from the component ``c`` directly.

    Parameters
    ----------
    m : linopy.Model
        Model
    c : pypsa.Components
        Component instance; used to read ``segments``, ``extendables``,
        ``static``, and ``ctype.segments_attrs``.
    x_var : linopy.Variable
        The optimisation variable for the x-axis of the piecewise constraint.
    seg_attr : str
        The piecewise attribute name, e.g. ``"marginal_cost"`` or ``"capital_cost"``.
    aux_var_name : str
        Name for the auxiliary linopy variable (if created) and the prefix for the constraint(s) defining the piecewise cost.
    active_names : pd.Index
        Active component names to consider (e.g. ``c.active_assets`` or ``c.extendables``).
    operator : {"<=", ">=", "=="}
        The operator to use in the piecewise constraint, of the form ``<x-axis var> <operator> <y-axis var>``.
    marginal_attr : bool
        Whether the y-axis breakpoints represent marginal values.
        If True, the integral of the piecewise curve will be used to define y breakpoints.
        If False, the nominal values at each x breakpoint will be used to define y breakpoints.
    y_var : linopy.Variable or None, optional
        The optimisation variable for the y-axis of the piecewise constraint.
        If None, a new auxiliary variable will be created and used instead.
    extra_options : Iterable[PiecewiseOptions]
        Extra options to pass to the piecewise constraint formulation.
        If multiple instance of piecewise options are provided for the same component and attribute,
        then multiple piecewise constraints will be created with each of the specified options.
        If empty, default settings will be used.
    method : str, optional
        The method to use for the piecewise constraint formulation, passed to linopy's add_piecewise_constraints method.
        Default is "auto" (also the linopy default).

    Returns
    -------
    linopy.Variable or None
        The variable representing the piecewise constraint, or None if no piecewise segments were defined.

    """
    seg_names = get_segmented_names(c, seg_attr, active_names)
    if seg_names is None:
        logger.debug(
            "No segments defined for '%s' on component '%s'. Skipping piecewise constraint.",
            seg_attr,
            c.name,
        )
        return None
    x_breakpoints, y_breakpoints = _get_breakpoints(
        c, seg_attr, seg_names, marginal_attr
    )

    if y_var is None:
        y_var_sel = _create_y_var(m, x_var, seg_names, aux_var_name)
    else:
        y_var_sel = y_var.sel(name=seg_names)
    x_var_sel = x_var.sel(name=seg_names)

    for option in extra_options:
        if isinstance(option.name, str):
            valid_names = pd.Index([option.name], name="name")
            aux_var_name_option = f"{aux_var_name}_{option.name}"
        else:
            valid_names = seg_names
            aux_var_name_option = aux_var_name
        _add_piecewise_constraint(
            m,
            x_var_sel,
            y_var_sel,
            x_breakpoints,
            y_breakpoints,
            aux_var_name_option,
            option.method,
            option.operator,
            valid_names,
        )
        seg_names = seg_names.difference(valid_names)
    if not seg_names.empty:
        _add_piecewise_constraint(
            m,
            x_var_sel,
            y_var_sel,
            x_breakpoints,
            y_breakpoints,
            aux_var_name,
            method,
            operator,
            seg_names,
        )
    return y_var_sel


def get_segmented_names(
    c: Any, seg_attr: str, active_names: pd.Index
) -> pd.Index | None:
    """Get component names with a given segmented (piecewise) attribute.

    Parameters
    ----------
    c : Any
        PyPSA component instance with a ``segments`` attribute.
    seg_attr : str
        The segmented attribute to look for, e.g. ``"marginal_cost"`` or ``"capital_cost"``.
    active_names : pd.Index
        The active component names to consider, e.g. ``c.active_assets`` or ``c.extendables``.

    Returns
    -------
    pd.Index | None:
        If there are segments defined for the given attribute, returns a pd.Index of component names that have segments and are in active_names.
        Otherwise, returns None.

    """
    seg_df = c.segments.get(seg_attr)
    if seg_df is None or seg_df.empty:
        return None

    seg_names = seg_df.columns.unique("name").intersection(active_names)
    return seg_names


def _add_piecewise_constraint(
    m: Model,
    x_var: Variable,
    y_var: Variable,
    x_breakpoints: xr.DataArray,
    y_breakpoints: xr.DataArray,
    aux_var_name: str,
    method: str,
    operator: OPS_T,
    valid_names: pd.Index,
) -> None:
    """Create a piecewise constraint for a given set of component names."""
    x_var_sel = x_var.sel(name=valid_names)
    y_var_sel = y_var.sel(name=valid_names)
    x_breakpoints_sel = x_breakpoints.sel(name=valid_names)
    y_breakpoints_sel = y_breakpoints.sel(name=valid_names)
    piecewise_func = piecewise(x_var_sel, x_breakpoints_sel, y_breakpoints_sel)
    m.add_piecewise_constraints(
        getattr(piecewise_func, OPS[operator])(y_var_sel),
        name=aux_var_name,
        method=method,
    )


def _get_breakpoints(
    c: Any, seg_attr: str, seg_names: pd.Index, marginal_attr: bool
) -> tuple[xr.DataArray, xr.DataArray]:
    """Convert segmented data to linopy breakpoints for piecewise constraint."""
    seg_df = c.segments[seg_attr][seg_names]

    x_attr = c.ctype.segments_attrs[seg_attr]

    # seg_attr stores the marginal value (slope) at each breakpoint.
    x_da = _to_da(seg_df, x_attr)
    y_da = _to_da(seg_df, seg_attr)

    # For per-unit x-axes (p_pu, e_pu), scale to absolute values and reject extendables.
    nom_attr = nominal_attrs.get(c.name)
    if nom_attr and x_attr == nom_attr.replace("_nom", "_pu"):
        if not (bad := seg_names.intersection(c.extendables)).empty:
            msg = (
                f"Piecewise '{seg_attr}' segments on a per-unit x-axis are not supported "
                f"for extendable components (fixed {nom_attr} required). "
                f"Extendable components: {bad.tolist()}."
            )
            raise ValueError(msg)
        nom_attr_da = xr.DataArray(c.static.loc[seg_names, nom_attr], dims="name")
        if (
            bad := (nom_attr_da == 0)
            | (nom_attr_da.isnull())
            | (nom_attr_da == float("inf"))
        ).any():
            bad_entries = nom_attr_da.where(bad).to_series().dropna().index.tolist()
            msg = (
                f"Piecewise '{seg_attr}' segments on a per-unit x-axis cannot be scaled to "
                f"absolute values for components with non-positive, non-finite or missing {nom_attr}. "
                f"Problematic components: {bad_entries}."
            )
            raise ValueError(msg)
        x_da = x_da * nom_attr_da
    if marginal_attr:
        y_da = ((y_da * x_da) - (y_da * x_da.shift({BREAKPOINT_DIM: 1}))).cumsum(
            BREAKPOINT_DIM
        )
    else:
        y_da *= x_da

    x_breakpoints = breakpoints(x_da)
    y_breakpoints = breakpoints(y_da)
    return x_breakpoints, y_breakpoints


def _create_y_var(
    m: Model, x_var: Any, seg_names: pd.Index, aux_var_name: str
) -> Variable:
    """Create auxiliary y variable for piecewise constraint."""
    extra_dims = [d for d in x_var.dims if d != "name"]
    coords = [x_var.coords[d].values for d in extra_dims] + [seg_names]
    dims = extra_dims + ["name"]
    y_var = m.add_variables(lower=0, coords=coords, dims=dims, name=aux_var_name)
    return y_var


def _to_da(seg_df: pd.DataFrame, attr: str) -> xr.DataArray:
    """Convert input to DataArray with given coords and dims."""
    da = xr.DataArray(seg_df.xs(attr, level="attribute", axis=1)).rename(
        segment=BREAKPOINT_DIM
    )
    return da
