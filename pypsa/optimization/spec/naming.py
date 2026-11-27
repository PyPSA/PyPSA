# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""The legacy naming of the built model: PyPSA's names and dims of its entries.

It owns the PyPSA name of each spec entry, as `Generator-p`, the renaming of
the spec's dims to PyPSA's, and the split of the global constraints by label.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import xarray as xr
from linopy.constants import BREAKPOINT_DIM
from linopy.constraints import Constraint
from linopy.expressions import LinearExpression
from linopy.variables import Variable

from pypsa.optimization.spec.layers import _unit_dims

if TYPE_CHECKING:
    from collections.abc import Callable

    from linopy import Model
    from mathspec.spec import Spec

    from pypsa.optimization.spec._render import Names


def pypsa_name(name: str, named: dict, mode: str | None = None) -> str:
    """Return the PyPSA name of a spec entry: its `name:`, or its key where it states none.

    An entry PyPSA names by loss mode states one name per mode; `mode` picks it.
    """
    found = named.get(name, name)
    return found if isinstance(found, str) else found.get(mode, name)


def _renamer(s: Spec, named: Names, mode: str | None) -> Callable[[str], str]:
    """Return the PyPSA name of each entry linopy builds from `s`.

    An entry a piecewise curve expands into takes the curve's PyPSA name
    and the suffix the spec's expansion gives it.
    """
    curves = {f"{k}_": pypsa_name(k, named["piecewise"]) for k in s.piecewise}

    def rename(key: str) -> str:
        for prefix, curve in curves.items():
            if key.startswith(prefix):
                return f"{curve}_{key.removeprefix(prefix)}"
        if key in s.constraints:
            return pypsa_name(key, named["constraints"], mode)
        return pypsa_name(key, named["variables"])

    return rename


#: Dims PyPSA names otherwise on one entry.
DIMS = {"Carrier-growth_limit": {"carrier": "Carrier", "period": "periods"}}


def _pypsa_dims(dims: tuple[str, ...], constraint: bool) -> dict[str, str]:
    """Return the renaming of spec dims to PyPSA's: the one unit dim becomes `name`.

    A constraint over buses or GlobalConstraint labels names them `name` too.
    The breakpoints of a curve take linopy's name, which marks an auxiliary entry.
    """
    named = {*_unit_dims(), *(("bus", "global_constraint") if constraint else ())}
    found = [d for d in dims if d in named]
    renamed = {found[0]: "name"} if len(found) == 1 else {}
    return renamed | ({"breakpoint": BREAKPOINT_DIM} if "breakpoint" in dims else {})


#: The fields of a constraint besides its terms, as the build sets them.
FIELDS = ("sign", "rhs", "labels", "scaling")


def _constraint(m: Model, lhs: LinearExpression, con: Constraint) -> Constraint:
    """Return `con` on `lhs`, a relabelling of its lhs, each field laid on the grid of `lhs`."""
    like = lhs.const
    fields = {
        k: like.copy(deep=False, data=getattr(con, k).values).rename(k) for k in FIELDS
    }
    return Constraint(lhs, model=m, name=con.name, attrs=dict(con.attrs), **fields)


#: The global constraints the spec states by sense, which PyPSA names by label.
BY_SENSE = {
    "GlobalConstraint-upper": "<=",
    "GlobalConstraint-lower": ">=",
    "GlobalConstraint-fix": "=",
}


def _split_by_label(m: Model) -> None:
    """Replace each constraint of `BY_SENSE` with one constraint `GlobalConstraint-{label}` per built row label."""
    for name, sign in BY_SENSE.items():
        if name not in m.constraints:
            continue
        con = m.constraints[name]
        m.remove_constraints(name)
        for label in con.indexes["name"]:
            built = con.labels.sel(name=label, drop=True) != -1
            if not built.any():
                continue
            m.add_constraints(
                con.lhs.sel(name=label, drop=True),
                sign,
                con.rhs.sel(name=label, drop=True),
                name=f"GlobalConstraint-{label}",
                mask=built,
                scaling=con.scaling.sel(name=label, drop=True),
            )


def _relabel(
    m: Model,
    *,
    snapshots: pd.Index,
    network_index: pd.Index,
) -> None:
    """Rename every dim of `m` as PyPSA names it, in linopy's order and with linopy's labels.

    A named expression is renamed as a variable is, and kept even without a row.
    Variables and constraints without a single row are removed, as PyPSA does not build them.
    A network without scenarios keeps its one scenario, labelled `BASE_SCENARIO`.
    A unit without a row stays as a masked coordinate.
    """
    levels = (
        {
            name: ("snapshot", network_index.get_level_values(name).to_numpy())
            for name in network_index.names
        }
        if isinstance(network_index, pd.MultiIndex)
        and not isinstance(snapshots, pd.MultiIndex)
        else {}
    )

    def frame[T: (LinearExpression, xr.Dataset)](
        obj: T, dims: tuple[str, ...], renamed: dict[str, str]
    ) -> T:
        if renamed:
            obj = obj.rename(renamed)
        if "snapshot" in dims:
            own = {k: v for k, v in levels.items() if k not in dims}
            obj = obj.assign_coords({"snapshot": snapshots, **own})
        return obj

    for name, var in list(m.variables.items()):
        if not (var.labels.values != -1).any():
            m.variables.remove(name)
            continue
        dims = tuple(var.labels.dims)
        data = frame(var.data, dims, _pypsa_dims(dims, False))
        m.variables.add(Variable(data, m, name, skip_broadcast=True))

    for _, expr in list(m.expressions.items()):
        dims = tuple(expr.coord_dims)
        m.expressions.add(frame(expr, dims, _pypsa_dims(dims, False)))

    constraints = {}
    for name, con in m.constraints.items():
        if not (con.labels.values != -1).any():
            continue
        dims = tuple(con.labels.dims)
        renamed = _pypsa_dims(dims, True) | DIMS.get(name, {})
        lhs = frame(con.lhs, dims, renamed)
        constraints[name] = _constraint(m, lhs, con)
    for name in list(m.constraints):
        m.constraints.remove(name)
    for con in constraints.values():
        m.constraints.add(con)
