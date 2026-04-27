# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Piecewise-constraint helpers for PyPSA optimization models."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import xarray as xr
from linopy import Model, Variable, breakpoints, piecewise
from linopy.constants import BREAKPOINT_DIM

from pypsa.descriptors import nominal_attrs

logger = logging.getLogger(__name__)


def define_piecewise_cost(
    m: Model,
    c: Any,
    x_var: Any,
    y_attr: str,
    aux_var_name: str,
    active_names: pd.Index,
) -> Variable | None:
    """Add variable(s) and constraint(s) necessary to define a cost along a piecewise linear curve.

    Derives the segment DataFrame, x-axis attribute, component names with segment
    data, and the optional p_nom scale factor from the component ``c`` directly.

    For ``marginal_cost``, components that are extendable are rejected with a
    ``ValueError`` (piecewise marginal cost requires fixed ``p_nom``).

    Parameters
    ----------
    m : linopy.Model
        Model
    c : pypsa.Components
        Component instance; used to read ``segments``, ``extendables``,
        ``static``, and ``ctype.segments_attrs``.
    x_var : linopy.Variable
        The optimisation variable for the x-axis (e.g. dispatch ``p`` or capacity
        ``p_nom``).  Should cover at least ``active_names``.
    y_attr : str
        Y-axis attribute name, e.g. ``"marginal_cost"`` or ``"capital_cost"``.
    aux_var_name : str
        Name for the auxiliary linopy variable and the prefix for the constraint(s) defining the piecewise cost.
    active_names : pd.Index
        Active component names to consider (e.g. ``c.active_assets`` or
        ``c.extendables``).

    """
    piecewise_attrs = _prepare_attrs(
        m,
        c,
        x_var,
        y_attr,
        aux_var_name,
        active_names,
    )
    if piecewise_attrs is None:
        y_var = None
    else:
        x_breakpoints, y_breakpoints, x_var_sel, y_var = piecewise_attrs
        piecewise_func = piecewise(x_var_sel, x_breakpoints, y_breakpoints)
        m.add_piecewise_constraints(piecewise_func <= y_var, name=aux_var_name)
    return y_var


def _prepare_attrs(
    m: Model,
    c: Any,
    x_var: Any,
    y_attr: str,
    aux_var_name: str,
    active_names: pd.Index,
) -> tuple[Any, Any, Any, Any] | None:
    """Helper to prepare the attributes to become inputs to piecewise constraints."""
    seg_df = c.segments.get(y_attr)
    if seg_df is None or seg_df.empty:
        return None

    seg_names = seg_df.columns.unique("name").intersection(active_names)
    if seg_names.empty:
        return None

    x_attr = c.ctype.segments_attrs[y_attr]

    seg_df = seg_df[seg_names]

    # y_attr stores the marginal value (slope) at each breakpoint.
    x_da = _to_da(seg_df, x_attr)
    y_da = _to_da(seg_df, y_attr)

    # For per-unit x-axes (p_pu, e_pu), scale to absolute values and reject extendables.
    nom_attr = nominal_attrs.get(c.name)
    if nom_attr and x_attr == nom_attr.replace("_nom", "_pu"):
        bad = seg_names.intersection(c.extendables)
        if not bad.empty:
            msg = (
                f"Piecewise '{y_attr}' segments on a per-unit x-axis are not supported "
                f"for extendable components (fixed {nom_attr} required). "
                f"Extendable components: {bad.tolist()}."
            )
            raise ValueError(msg)
        x_da = x_da * xr.DataArray(c.static.loc[seg_names, nom_attr], dims="name")
        # We assume that the y-axis data will have also been given _pu, so we scale it too.
        y_da *= x_da

    extra_dims = [d for d in x_var.dims if d != "name"]
    coords = [x_var.coords[d].values for d in extra_dims] + [seg_names]
    dims = extra_dims + ["name"]
    y_var = m.add_variables(lower=0, coords=coords, dims=dims, name=aux_var_name)

    x_var_sel = x_var.sel(name=seg_names)
    x_breakpoints = breakpoints(x_da)
    y_breakpoints = breakpoints(y_da)

    return x_breakpoints, y_breakpoints, x_var_sel, y_var


def _to_da(seg_df: pd.DataFrame, attr: str) -> xr.DataArray:
    """Helper to convert input to DataArray with given coords and dims."""
    da = xr.DataArray(seg_df.xs(attr, level="attribute", axis=1)).rename(
        segment=BREAKPOINT_DIM
    )
    return da
