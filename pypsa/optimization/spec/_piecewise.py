# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Data prep of the spec: piecewise curves."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from pypsa.optimization.spec.sources import NetworkSources, source

if TYPE_CHECKING:
    from pypsa.components import Components


def _nominal(s: NetworkSources, c: Components, names: pd.Index) -> pd.Series:
    """Read the nominal value that scales a per-unit curve; where it is not finite the curve stays per unit, for an assumption to refuse."""
    nominal = s.static(c).reindex(names)[c._operational_attrs["nom"]]
    return nominal.where(np.isfinite(nominal), 1.0)


def _table(
    s: NetworkSources, c: Components, attr: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the x and y breakpoints of `c.piecewise[attr]`, (breakpoint, unit), x in absolute units."""
    curves = c.piecewise.get(attr)
    if curves is None or curves.empty:
        empty = pd.DataFrame(index=pd.RangeIndex(0, name="breakpoint"))
        return empty, empty
    curves = curves.dropna(how="all", axis=1)
    names = curves.columns.unique("name").intersection(s.names(c))
    schema = c._piecewise_schema(attr)
    x = curves.xs(schema.x, level="attribute", axis=1)[names]
    y = curves.xs(attr, level="attribute", axis=1)[names]
    if not schema.allow_extendable:
        x = x * _nominal(s, c, names)
    valid = x.notna() & y.notna()
    return x.where(valid), y.where(valid)


def _cumulative(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """Sum each segment's width times the value at its upper breakpoint, from zero."""
    return (x.diff() * y).fillna(0).cumsum().where(x.notna())


def _grid(s: NetworkSources, name: str, frame: pd.DataFrame) -> xr.DataArray:
    """Lay a (breakpoint, label) table out over the declared dims of `name`."""
    dim = next(d for d in s.declared(name) if d != "breakpoint")
    frame = frame.reindex(index=s.coords("breakpoint"), columns=s.coords(dim))
    coords = {
        "breakpoint": frame.index.to_numpy(),
        dim: frame.columns.to_numpy(dtype=object),
    }
    return xr.DataArray(frame.to_numpy(dtype=float), coords, ("breakpoint", dim))


@source("breakpoint")
def _breakpoint(s: NetworkSources, name: str) -> pd.Index:
    """Read the breakpoint positions of every curve of the network."""
    index = pd.Index([], dtype=int)
    for c in s.units.values():
        for curves in c.piecewise.values():
            index = index.union(curves.index)
    return pd.Index(index, dtype=int)


@source(
    "${C}_marginal_cost_curve_p",
    "${C}_marginal_cost_curve_cost",
    "${C}_capital_cost_curve_nom",
    "${C}_capital_cost_curve_cost",
)
def _cost_curve(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    """Read a `marginal_cost` or `capital_cost` curve: its x axis, or its cost summed over segments."""
    attr, _, column = name.removeprefix(f"{c.name}_").rpartition("_curve_")
    x, y = _table(s, c, attr)
    return _grid(s, name, _cumulative(x, y) if column == "cost" else x)


@source("Generator_efficiency_curve_p", "Generator_efficiency_curve_primary")
def _primary_curve(s: NetworkSources, name: str) -> xr.DataArray:
    """Read a generator's `efficiency` curve: its output, or the output over the efficiency there."""
    x, y = _table(s, s.n.components.generators, "efficiency")
    return _grid(
        s, name, x if name.endswith("_p") else (x / y).fillna(0).where(x.notna())
    )


@source(
    "Link_efficiency_curve_p",
    "Link_efficiency_curve_delivery",
    "Process_rate_curve_p",
    "Process_rate_curve_transfer",
)
def _port_curve(s: NetworkSources, name: str) -> xr.DataArray:
    """Read the `efficiency<k>` or `rate<k>` curves long by port: the flow, or the flow times the value there."""
    c = s.n.components[name.partition("_")[0]]
    ports = s.ports(c)
    frames = [pd.DataFrame(index=pd.RangeIndex(0, name="breakpoint"))]
    for k, group in ports.groupby("k"):
        x, y = _table(s, c, c._port_coefficient_attr(str(k)))
        frame = x if name.endswith("_p") else (x * y).fillna(0).where(x.notna())
        labels = group.set_index(c._unit_dim)["port"]
        frames.append(frame.rename(columns=labels))
    return _grid(s, name, pd.concat(frames, axis=1))
