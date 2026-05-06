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
    seg_attr: str,
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
        c, seg_attr, seg_names, marginal_attr, invert_attr
    )

    if y_var is None:
        y_var = _create_y_var(m, x_var, seg_names, aux_var_name)
    y_var_sel = y_var.sel(name=seg_names)

    for option in [*extra_options, None]:
        if option is None:
            names, aux, opt_method, opt_op = (
                seg_names,
                aux_var_name,
                method,
                operator,
            )
        elif option.name:
            names = pd.Index([option.name], name="name")
            aux = f"{aux_var_name}_{option.name}"
            opt_method, opt_op = option.method, option.operator
        else:
            names, aux = seg_names, aux_var_name
            opt_method, opt_op = option.method, option.operator
        if names.empty:
            continue
        if opt_method == "lp" and opt_op == "==":
            msg = "method 'lp' requires PyPSA operator '<=' or '>='."
            raise ValueError(msg)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=EvolvingAPIWarning)
            m.add_piecewise_formulation(
                (y_var.sel(name=names), y_breakpoints.sel(name=names), opt_op),
                (x_var.sel(name=names), x_breakpoints.sel(name=names)),
                method=opt_method,
                name=aux,
            )
        seg_names = seg_names.difference(names)
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
    valid_names = [name for name in seg_names if seg_df[name].notna().any().any()]
    if not valid_names:
        return None
    return pd.Index(valid_names, name=seg_names.name)


def _get_breakpoints(
    c: Any,
    seg_attr: str,
    seg_names: pd.Index,
    marginal_attr: bool,
    invert_attr: bool = False,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Convert segmented data to linopy breakpoints for piecewise constraint."""
    seg_df = c.segments[seg_attr][seg_names]

    x_attr = c.ctype.segments_attrs[seg_attr]
    seg_df = _normalize_segments(seg_df, x_attr, seg_attr)

    x_da = _to_da(seg_df, x_attr)
    y_da = _to_da(seg_df, seg_attr)
    valid_breakpoints = x_da.notnull() & y_da.notnull()
    if invert_attr:
        y_da = 1 / y_da

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


def _normalize_segments(seg_df: pd.DataFrame, x_attr: str, y_attr: str) -> pd.DataFrame:
    """Sort segment rows by x-coordinate and align ragged curves with trailing NaNs."""
    curves = []
    for name in seg_df.columns.unique("name"):
        curve = seg_df[name].reindex(columns=[x_attr, y_attr])
        row_has_data = curve.notna().any(axis=1)
        has_later_data = row_has_data.iloc[::-1].cummax().iloc[::-1]
        non_trailing_missing = ~row_has_data & has_later_data
        if non_trailing_missing.any():
            bad = non_trailing_missing[non_trailing_missing].index.tolist()
            msg = (
                f"Piecewise '{y_attr}' segments for component '{name}' contain "
                f"non-trailing missing breakpoint rows: {bad}."
            )
            raise ValueError(msg)
        incomplete = row_has_data & curve.isna().any(axis=1)
        if incomplete.any():
            bad = incomplete[incomplete].index.tolist()
            msg = (
                f"Piecewise '{y_attr}' segments for component '{name}' have "
                f"incomplete breakpoint data at rows: {bad}."
            )
            raise ValueError(msg)
        curve = curve.loc[row_has_data].sort_values(x_attr, kind="mergesort")
        curve = curve.reset_index(drop=True)
        curve.columns = pd.MultiIndex.from_product(
            [[name], [x_attr, y_attr]], names=["name", "attribute"]
        )
        curves.append(curve)

    normalized = pd.concat(curves, axis=1)
    normalized.index.name = "segment"
    return normalized


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
