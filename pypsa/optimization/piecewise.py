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
    pw_attr: str,
    aux_var_name: str,
    active_names: pd.Index,
    operator: OPS_T,
    marginal_attr: bool,
    extra_options: Iterable[PiecewiseOptions],
    invert_attr: bool = False,
    y_var: Any | None = None,
    method: str = "auto",
) -> Variable | None:
    """Add variable(s) and constraint(s) necessary to define a constraint along a piecewise linear curve.

    Derives the piecewise DataFrame, x-axis attribute, component names with piecewise
    data, and the optional p_nom scale factor from the component ``c`` directly.

    Parameters
    ----------
    m : linopy.Model
        Model
    c : pypsa.Components
        Component instance with a ``piecewise`` attribute containing the piecewise data for the specified pw_attr.
    x_var : linopy.Variable
        The optimisation variable for the x-axis of the piecewise constraint.
    pw_attr : str
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
    invert_attr : bool, optional
        Whether to invert the piecewise attribute values before multiplying by the x-axis variable.
        If True, y_attr -> 1 / y_attr.
        Default is False.
    method : str, optional
        The method to use for the piecewise constraint formulation, passed to linopy's add_piecewise_constraints method.
        Default is "auto" (also the linopy default).

    Returns
    -------
    linopy.Variable or None
        The variable representing the piecewise constraint, or None if no piecewise breakpoints were defined.

    """
    pw_names = get_piecewise_names(c, pw_attr, active_names)
    if pw_names is None:
        logger.debug(
            "No piecewise breakpoints defined for '%s' on component '%s'. Skipping piecewise constraint.",
            pw_attr,
            c.name,
        )
        return None
    x_breakpoints, y_breakpoints = _get_breakpoints(
        c, pw_attr, pw_names, marginal_attr, invert_attr
    )

    if y_var is None:
        y_var_sel = _create_y_var(m, x_var, pw_names, aux_var_name)
    else:
        y_var_sel = y_var.sel(name=pw_names)
    x_var_sel = x_var.sel(name=pw_names)

    for option in extra_options:
        if isinstance(option.name, str):
            valid_names = pd.Index([option.name], name="name")
            aux_var_name_option = f"{aux_var_name}_{option.name}"
        else:
            valid_names = pw_names
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
        pw_names = pw_names.difference(valid_names)
    if not pw_names.empty:
        _add_piecewise_constraint(
            m,
            x_var_sel,
            y_var_sel,
            x_breakpoints,
            y_breakpoints,
            aux_var_name,
            method,
            operator,
            pw_names,
        )
    return y_var_sel


def get_piecewise_names(
    c: Any, pw_attr: str, active_names: pd.Index
) -> pd.Index | None:
    """Get component names with a given piecewise attribute.

    Parameters
    ----------
    c : Any
        PyPSA component instance with a ``piecewise`` attribute.
    pw_attr : str
        The piecewise attribute to look for, e.g. ``"marginal_cost"`` or ``"capital_cost"``.
    active_names : pd.Index
        The active component names to consider, e.g. ``c.active_assets`` or ``c.extendables``.

    Returns
    -------
    pd.Index | None:
        If there are piecewise curves defined for the given attribute, returns a pd.Index of component names that have piecewise curves and are in ``active_names``.
        Otherwise, returns None.

    """
    piecewise_df = c.piecewise.get(pw_attr)
    if piecewise_df is None or piecewise_df.empty:
        return None

    pw_names = piecewise_df.columns.unique("name").intersection(active_names)
    return pw_names


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
    c: Any,
    pw_attr: str,
    pw_names: pd.Index,
    marginal_attr: bool,
    invert_attr: bool = False,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Convert piecewise data to linopy breakpoints for piecewise constraint."""
    piecewise_df = c.piecewise[pw_attr][pw_names]

    x_attr = c.ctype.piecewise_attrs[pw_attr]

    # pw_attr stores the marginal value (slope) at each breakpoint.
    x_da = _to_da(piecewise_df, x_attr)
    y_da = _to_da(piecewise_df, pw_attr)
    if invert_attr:
        y_da = 1 / y_da

    # For per-unit x-axes (p_pu, e_pu), scale to absolute values and reject extendables.
    nom_attr = nominal_attrs.get(c.name)
    if nom_attr and x_attr == nom_attr.replace("_nom", "_pu"):
        if not (bad := pw_names.intersection(c.extendables)).empty:
            msg = (
                f"Piecewise '{pw_attr}' breakpoints on a per-unit x-axis are not supported "
                f"for extendable components (fixed {nom_attr} required). "
                f"Extendable components: {bad.tolist()}."
            )
            raise ValueError(msg)
        nom_attr_da = xr.DataArray(c.static.loc[pw_names, nom_attr], dims="name")
        if (
            bad := (nom_attr_da == 0)
            | (nom_attr_da.isnull())
            | (nom_attr_da == float("inf"))
        ).any():
            bad_entries = nom_attr_da.where(bad).to_series().dropna().index.tolist()
            msg = (
                f"Piecewise '{pw_attr}' breakpoints on a per-unit x-axis cannot be scaled to "
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
    m: Model, x_var: Any, pw_names: pd.Index, aux_var_name: str
) -> Variable:
    """Create auxiliary y variable for piecewise constraint."""
    extra_dims = [d for d in x_var.dims if d != "name"]
    coords = [x_var.coords[d].values for d in extra_dims] + [pw_names]
    dims = extra_dims + ["name"]
    y_var = m.add_variables(lower=0, coords=coords, dims=dims, name=aux_var_name)
    return y_var


def _to_da(piecewise_df: pd.DataFrame, attr: str) -> xr.DataArray:
    """Convert input to DataArray with given coords and dims."""
    da = xr.DataArray(piecewise_df.xs(attr, level="attribute", axis=1)).rename(
        breakpoint=BREAKPOINT_DIM
    )
    return da
