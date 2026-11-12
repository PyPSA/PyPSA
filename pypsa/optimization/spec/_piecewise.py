# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Data prep of the spec: piecewise curves."""

from __future__ import annotations

import pandas as pd
import xarray as xr

from pypsa.optimization.spec.sources import (
    NOMINAL,
    UNITS,
    NetworkSources,
    _static,
    dimension,
    single,
)

#: The components that take a piecewise `marginal_cost` curve.
RUNNING = ("Generator", "Link", "Process", "StorageUnit", "Store")
#: The components whose ports take a piecewise curve, with the suffix of its value column.
PORTS = {"Link": "delivery", "Process": "transfer"}


def _fixed_nominal(
    s: NetworkSources, component: str, attr: str, names: pd.Index
) -> pd.Series:
    """Read the fixed nominal value that scales a per-unit curve, refused where it cannot."""
    nom = NOMINAL[component]
    static = _static(s, component).reindex(names)
    extendable = names[static[f"{nom}_extendable"].to_numpy(dtype=bool)]
    if len(extendable):
        msg = (
            f"Piecewise '{attr}' breakpoints on a per-unit x-axis are not supported "
            f"for extendable components (fixed {nom} required). "
            f"Extendable components: {extendable.tolist()}."
        )
        raise ValueError(msg)
    nominal = static[nom]
    bad = (nominal == 0) | nominal.isna() | (nominal == float("inf"))
    if bad.any():
        msg = (
            f"Piecewise '{attr}' breakpoints on a per-unit x-axis cannot be scaled to "
            f"absolute values for components with non-positive, non-finite or missing {nom}. "
            f"Problematic components: {names[bad.to_numpy()].tolist()}."
        )
        raise ValueError(msg)
    return nominal


def _table(
    s: NetworkSources, component: str, attr: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the x and y breakpoints of `c.piecewise[attr]`, (breakpoint, unit), x in absolute units."""
    c = s.n.components[component]
    curves = c.piecewise.get(attr)
    if curves is None or curves.empty:
        empty = pd.DataFrame(index=pd.RangeIndex(0, name="breakpoint"))
        return empty, empty
    curves = curves.dropna(how="all", axis=1)
    names = curves.columns.unique("name").intersection(s.names(component))
    schema = c._piecewise_schema(attr)
    x = curves.xs(schema.x, level="attribute", axis=1)[names]
    y = curves.xs(attr, level="attribute", axis=1)[names]
    if not schema.allow_extendable:
        x = x * _fixed_nominal(s, component, attr, names)
    valid = x.notna() & y.notna()
    return x.where(valid), y.where(valid)


def _cumulative(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """Sum each segment's width times the value at its upper breakpoint, from zero."""
    return (x.diff() * y).fillna(0).cumsum().where(x.notna())


def _grid(s: NetworkSources, name: str, frame: pd.DataFrame) -> xr.DataArray:
    """Lay a (breakpoint, label) table out over the declared dims of `name`."""
    dim = next(d for d in s.declared(name) if d != "breakpoint")
    frame = frame.reindex(index=s.coords("breakpoint"), columns=s.coords(dim))
    da = xr.DataArray(
        frame.to_numpy(dtype=float),
        coords={
            "breakpoint": frame.index.to_numpy(),
            dim: frame.columns.to_numpy(dtype=object),
        },
        dims=("breakpoint", dim),
    )
    return s.frame(name, da)


@dimension("breakpoint")
def _breakpoint(s: NetworkSources) -> pd.Index:
    """Read the breakpoint positions of every curve of the network."""
    index = pd.Index([], dtype=int)
    for c in s.n.components[list(NOMINAL)]:
        for curves in c.piecewise.values():
            index = index.union(curves.index)
    return pd.Index(index, dtype=int)


def _refuse_overnight_cost(s: NetworkSources, component: str, names: pd.Index) -> None:
    """Refuse `overnight_cost` on an extendable unit with a `capital_cost` curve, which is already periodized."""
    static = _static(s, component).reindex(names)
    extendable = static[f"{NOMINAL[component]}_extendable"].to_numpy(dtype=bool)
    bad = names[extendable & static["overnight_cost"].notna().to_numpy()]
    if len(bad):
        msg = (
            f"Components {bad.tolist()} of type {component} define both a piecewise "
            "'capital_cost' curve and 'overnight_cost'. The piecewise "
            "curve must already be periodized; remove 'overnight_cost'."
        )
        raise ValueError(msg)


def _cost_curve(s: NetworkSources, name: str) -> xr.DataArray:
    """Read a `marginal_cost` or `capital_cost` curve: its x axis, or its cost summed over segments."""
    component, _, rest = name.partition("_")
    attr, _, column = rest.rpartition("_curve_")
    x, y = _table(s, component, attr)
    if attr == "capital_cost":
        _refuse_overnight_cost(s, component, x.columns)
    return _grid(s, name, _cumulative(x, y) if column == "cost" else x)


@single("Generator_efficiency_curve_p", "Generator_efficiency_curve_primary")
def _primary_curve(s: NetworkSources, name: str) -> xr.DataArray:
    """Read a generator's `efficiency` curve: its output, or the output over the efficiency there."""
    x, y = _table(s, "Generator", "efficiency")
    return _grid(
        s, name, x if name.endswith("_p") else (x / y).fillna(0).where(x.notna())
    )


def _port_curve(s: NetworkSources, name: str) -> xr.DataArray:
    """Read the `efficiency<k>` or `rate<k>` curves long by port: the flow, or the flow times the value there."""
    component = name.partition("_")[0]
    c = s.n.components[component]
    ports = s.link_ports if component == "Link" else s.process_ports
    frames = [pd.DataFrame(index=pd.RangeIndex(0, name="breakpoint"))]
    for k, group in ports.groupby("k"):
        x, y = _table(s, component, c._port_coefficient_attr(str(k)))
        frame = x if name.endswith("_p") else (x * y).fillna(0).where(x.notna())
        labels = group.set_index(UNITS[component])["port"]
        frames.append(frame.rename(columns=labels))
    return _grid(s, name, pd.concat(frames, axis=1))


for _component in RUNNING:
    single(
        f"{_component}_marginal_cost_curve_p", f"{_component}_marginal_cost_curve_cost"
    )(_cost_curve)
for _component in NOMINAL:
    single(
        f"{_component}_capital_cost_curve_nom", f"{_component}_capital_cost_curve_cost"
    )(_cost_curve)
for _component, _column in PORTS.items():
    _attr = "efficiency" if _component == "Link" else "rate"
    single(f"{_component}_{_attr}_curve_p", f"{_component}_{_attr}_curve_{_column}")(
        _port_curve
    )
