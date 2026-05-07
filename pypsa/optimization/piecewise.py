# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Piecewise-constraint helpers for PyPSA optimization models."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd
import xarray as xr
from linopy import Model, Variable, breakpoints
from linopy.constants import BREAKPOINT_DIM, PWL_METHOD, SIGNS, EvolvingAPIWarning

from pypsa.constants import PIECEWISE_ATTRS
from pypsa.descriptors import nominal_attrs

if TYPE_CHECKING:
    from collections.abc import Iterable
logger = logging.getLogger(__name__)


@dataclass(eq=True, frozen=True)
class PiecewiseOptions:
    """Options for piecewise constraint formulation.

    The operator is interpreted as ``y operator f(x)``.
    """

    component: str
    attribute: str
    operator: SIGNS
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
    operator: SIGNS,
    marginal_attr: bool,
    extra_options: Iterable[PiecewiseOptions],
    invert_attr: bool = False,
    y_var: Any | None = None,
    method: PWL_METHOD = "auto",
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
        The operator for the piecewise constraint, interpreted as ``y <operator> f(x)``.
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
        The method to use for the piecewise constraint formulation, passed to linopy's add_piecewise_formulation method.
        Default is "auto" (also the linopy default).

    Returns
    -------
    linopy.Variable or None
        The variable representing the piecewise constraint, or None if no piecewise breakpoints were defined.

    """
    pw_names = get_piecewise_names(c, pw_attr, active_names)
    if pw_names.empty:
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
        y_var = _create_y_var(m, x_var, pw_names, aux_var_name)
    y_var_sel = y_var.sel(name=pw_names)

    for option in [*extra_options, None]:
        if option is None:
            names, aux, opt_method, opt_op = (
                pw_names,
                aux_var_name,
                method,
                operator,
            )
        elif option.name:
            names = pd.Index([option.name], name="name")
            aux = f"{aux_var_name}_{option.name}"
            opt_method, opt_op = option.method, option.operator
        else:
            names, aux = pw_names, aux_var_name
            opt_method, opt_op = option.method, option.operator
        if names.empty:
            continue

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=EvolvingAPIWarning)
            m.add_piecewise_formulation(
                (y_var.sel(name=names), y_breakpoints.sel(name=names), opt_op),
                (x_var.sel(name=names), x_breakpoints.sel(name=names)),
                method=opt_method,
                name=aux,
            )
        pw_names = pw_names.difference(names)
    return y_var_sel


def get_piecewise_names(c: Any, pw_attr: str, active_names: pd.Index) -> pd.Index:
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
    pd.Index:
        If there are piecewise curves defined for the given attribute, returns a pd.Index of component
        names that have piecewise curves and are in ``active_names``.

    """
    piecewise_df = c.piecewise.get(pw_attr)
    if piecewise_df is None or piecewise_df.empty:
        return pd.Index([], name="name")

    return (
        piecewise_df.dropna(how="all", axis=1)
        .columns.unique("name")
        .intersection(active_names)
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
    piecewise_attrs = PIECEWISE_ATTRS.query(
        "component == @c.name and y == @pw_attr"
    ).squeeze()

    # pw_attr stores the marginal value (slope) at each breakpoint.
    piecewise_df = _normalize_segments(piecewise_df, piecewise_attrs)

    x_da = _to_da(piecewise_df, piecewise_attrs.x)
    y_da = _to_da(piecewise_df, pw_attr)
    valid_breakpoints = x_da.notnull() & y_da.notnull()
    if invert_attr:
        y_da = 1 / y_da

    # For per-unit x-axes (p_pu, e_pu), scale to absolute values and reject extendables.
    nom_attr = nominal_attrs.get(c.name)
    if not piecewise_attrs.allow_extendable:
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
    x_da = x_da.where(valid_breakpoints)
    x_breakpoints = breakpoints(x_da)
    if marginal_attr:
        # y_da[i] is the marginal between x[i-1] and x[i]; shift to linopy's slope
        # convention where slopes[i] is the slope between x_points[i] and x_points[i+1].
        slopes = y_da.shift({BREAKPOINT_DIM: -1})
        y_breakpoints = breakpoints(slopes=slopes, x_points=x_da, y0=0.0)
    else:
        y_breakpoints = breakpoints((y_da * x_da).where(valid_breakpoints))
    return x_breakpoints, y_breakpoints


def _normalize_segments(
    piecewise_df: pd.DataFrame, piecewise_attrs: pd.Series[str]
) -> pd.DataFrame:
    """Sort segment rows by x-coordinate and align ragged curves with trailing NaNs."""
    curves = []
    for name in piecewise_df.columns.unique("name"):
        curve = piecewise_df[name].reindex(
            columns=[piecewise_attrs.x, piecewise_attrs.y]
        )
        row_has_data = curve.notna().any(axis=1)
        has_later_data = row_has_data.iloc[::-1].cummax().iloc[::-1]
        non_trailing_missing = ~row_has_data & has_later_data
        if non_trailing_missing.any():
            bad = non_trailing_missing[non_trailing_missing].index.tolist()
            msg = (
                f"Piecewise '{piecewise_attrs.y}' segments for component '{name}' contain "
                f"non-trailing missing breakpoint rows: {bad}."
            )
            raise ValueError(msg)
        incomplete = row_has_data & curve.isna().any(axis=1)
        if incomplete.any():
            bad = incomplete[incomplete].index.tolist()
            msg = (
                f"Piecewise '{piecewise_attrs.y}' segments for component '{name}' have "
                f"incomplete breakpoint data at rows: {bad}."
            )
            raise ValueError(msg)
        curve = curve.loc[row_has_data].sort_values(piecewise_attrs.x, kind="mergesort")
        curve = curve.reset_index(drop=True)
        curve.columns = pd.MultiIndex.from_product(
            [[name], [piecewise_attrs.x, piecewise_attrs.y]],
            names=["name", "attribute"],
        )
        curves.append(curve)

    normalized = pd.concat(curves, axis=1)
    normalized.index.name = "segment"
    return normalized


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
        segment=BREAKPOINT_DIM
    )
    return da
