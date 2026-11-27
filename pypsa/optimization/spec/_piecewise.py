# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Data prep of the spec: piecewise curves, and the patch that applies piecewise options to them."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

from pypsa.optimization.spec.sources import NetworkSources, source

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mathspec.spec import Spec

    from pypsa.components import Components
    from pypsa.optimization.piecewise import PiecewiseOptions


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
    """Lay a (breakpoint, label) table out over the declared dims of `name`; a label over several dims is a tuple."""
    others = [d for d in s.declared(name) if d != "breakpoint"]
    coords = {d: s.coords(d) for d in [*others, "breakpoint"]}
    keys = frame.columns if len(others) > 1 else [(k,) for k in frame.columns]
    at = np.array(
        [coords[d].get_indexer([k[i] for k in keys]) for i, d in enumerate(others)]
    ).reshape(len(others), -1)
    found = (at != -1).all(axis=0)
    values = frame.reindex(index=coords["breakpoint"]).to_numpy(dtype=float).T
    data = np.full([len(c) for c in coords.values()], np.nan)
    data[tuple(at[:, found])] = values[found]
    da = xr.DataArray(data, coords, tuple(coords))
    return da.transpose(*s.declared(name))


@source("breakpoint")
def _breakpoint(s: NetworkSources, name: str) -> pd.Index:
    """Read the breakpoint positions of every curve of the network."""
    index = pd.Index([], dtype=int)
    for c in s.units.values():
        for curves in c.piecewise.values():
            index = index.union(curves.index)
    return pd.Index(index, dtype=int)


@source(
    "COMPONENT_marginal_cost_curve_p",
    "COMPONENT_marginal_cost_curve_cost",
    "COMPONENT_capital_cost_curve_nom",
    "COMPONENT_capital_cost_curve_cost",
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
    frames = [pd.DataFrame(index=pd.RangeIndex(0, name="breakpoint"))]
    for k in s.ports(c)["port"].unique():
        x, y = _table(s, c, c._port_coefficient_attr(str(k)))
        frame = x if name.endswith("_p") else (x * y).fillna(0).where(x.notna())
        frames.append(frame.set_axis([(u, k) for u in frame.columns], axis=1))
    return _grid(s, name, pd.concat(frames, axis=1))


def _link(link: Any, sign: str) -> list[str]:
    """Return a piecewise link as the spec writes it, which leaves out the default `==`."""
    return [link.expression, link.values, *([sign] if sign != "==" else [])]


def _piecewise_patch(base: Spec, options: Sequence[PiecewiseOptions]) -> dict:
    """Return the spec patch that states each curve by the method and sign its option names.

    The spec states linopy's `method='auto'` for PyPSA's default signs as
    two entries per curve, one picked by data. An option keeps the entry of
    its method, ungated from that pick, and removes the other; its sign
    goes on the curve's last link, the y-axis. `auto` with an equality
    takes the incremental entry, as linopy does.

    Raises:
        NotImplementedError: An option names units, two options name one
            curve, the curve has no entry of the method, or `auto` asks
            for another inequality than the spec's.
        ValueError: An option names a curve the spec does not hold.

    """
    patch: dict[str, Any] = {}
    seen: set[tuple[str, str]] = set()
    for option in options:
        curve = (option.component, option.attribute)
        if option.name:
            msg = f"A piecewise option for named units ({option.name}) is not supported; give one for all units of {option.component}."
            raise NotImplementedError(msg)
        if curve in seen:
            msg = f"Several piecewise options for {option.component}-{option.attribute} are not supported; give one."
            raise NotImplementedError(msg)
        seen.add(curve)
        prefix = f"{option.component}_{option.attribute}"
        entries = {
            k: v
            for k, v in base.piecewise.items()
            if re.fullmatch(rf"{prefix}_(lp|incremental)", k)
        }
        if not entries:
            msg = f"No piecewise curve {option.component}-{option.attribute} to apply the option to."
            raise ValueError(msg)
        method = option.method
        if method == "auto" and option.sign == "==":
            method = "incremental"
        elif method == "auto":
            if any(
                list(e.links.values())[-1].sign != option.sign for e in entries.values()
            ):
                msg = f"method='auto' with sign '{option.sign}' is not supported for {option.component}-{option.attribute}; name the method."
                raise NotImplementedError(msg)
            continue
        kept = f"{prefix}_{method}"
        if kept not in entries:
            msg = f"method='{option.method}' is not supported for {option.component}-{option.attribute}."
            raise NotImplementedError(msg)
        entry = entries[kept]
        *head, (y, last) = entry.links.items()
        patch |= dict.fromkeys(set(entries) - {kept})
        patch[kept] = {
            "method": method,
            "links": {k: _link(v, v.sign) for k, v in head}
            | {y: _link(last, option.sign)},
        }
        if entry.where:
            patch[kept]["where"] = re.sub(
                rf"(NOT )?{prefix}_as_lp AND ", "", entry.where
            )
    return {"piecewise": patch}
