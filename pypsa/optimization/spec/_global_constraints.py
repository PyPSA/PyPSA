# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Data prep of the spec: global constraints and carrier growth limits."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

from pypsa.optimization.spec.sources import (
    NO_SCENARIO,
    NetworkSources,
    of_scenario,
    source,
)

if TYPE_CHECKING:
    from pypsa.components import Components

Cell = Callable[[NetworkSources, "Components", Any, pd.Series, pd.DataFrame], pd.Series]


def _rows(s: NetworkSources, attr: str, labels: pd.Index | None = None) -> pd.DataFrame:
    """Return a GlobalConstraint attribute, one row per constraint and one column per scenario."""
    column = s.n.components.global_constraints.static[attr]
    if s.n.has_scenarios:
        table = column.unstack("scenario")
    else:
        table = column.to_frame(NO_SCENARIO)
    if labels is None:
        labels = s.coords("global_constraint")
    return table.reindex(index=labels, columns=s.scenarios)


def _periods(s: NetworkSources, labels: pd.Index | None = None) -> pd.DataFrame:
    """Return the investment period each row states, NaN where it states none."""
    period = _rows(s, "investment_period", labels).astype(float)
    stated = period.notna().any(axis=1)
    if not s.multi_invest and stated.any():
        msg = (
            f"GlobalConstraint {stated.index[stated].tolist()} set "
            "`investment_period`, but the network has no investment periods."
        )
        raise ValueError(msg)
    return period


def _grid(s: NetworkSources, table: pd.DataFrame) -> xr.DataArray:
    return xr.DataArray(
        table.to_numpy(),
        coords={
            "global_constraint": table.index.to_numpy(dtype=object),
            "scenario": s.scenarios,
        },
        dims=("global_constraint", "scenario"),
    )


def _cells(
    s: NetworkSources, kind: str, labels: pd.Index | None = None
) -> Iterator[tuple[int, int, Any, pd.Series]]:
    """Each (row, scenario) of a constraint type with its static data."""
    types = _rows(s, "type", labels)
    period = _periods(s, labels)
    static = s.n.components.global_constraints.static
    for i, gc in enumerate(types.index):
        for j, scenario in enumerate(types.columns):
            if types.iat[i, j] != kind:
                continue
            glc = static.loc[(scenario, gc) if s.n.has_scenarios else gc].copy()
            glc["investment_period"] = period.iat[i, j]
            yield i, j, scenario, glc


def _stated(glc: pd.Series) -> int | None:
    period = glc["investment_period"]
    return None if np.isnan(period) else int(period)


def _counted_periods(s: NetworkSources, glc: pd.Series) -> pd.Index:
    """Return the periods of the run a row counts."""
    period = _stated(glc)
    return s.periods if period is None else s.periods.intersection([period])


def _unit_grid(
    s: NetworkSources, name: str, c: Components, kind: str, cell: Cell
) -> xr.DataArray:
    """Return a (unit, row, scenario) grid, each cell a series over the units in its set."""
    dim = c._unit_dim
    units = s.coords(dim)
    rows = s.coords("global_constraint")
    values = np.full((len(units), len(rows), len(s.scenarios)), np.nan)
    for i, j, scenario, glc in _cells(s, kind):
        static = of_scenario(c.static, s.n, scenario)
        values[:, i, j] = cell(s, c, scenario, glc, static).reindex(units)
    da = xr.DataArray(
        values,
        coords={
            dim: units.to_numpy(dtype=object),
            "global_constraint": rows.to_numpy(dtype=object),
            "scenario": s.scenarios,
        },
        dims=(dim, "global_constraint", "scenario"),
    )
    if "scenario" not in s.declared(name):
        da = da.isel(scenario=0, drop=True)
    return da


def _ones(static: pd.DataFrame, mask: pd.Series) -> pd.Series:
    return pd.Series(1.0, index=static.index[mask.to_numpy(dtype=bool)])


def _non_cyclic(c: Components, static: pd.DataFrame) -> pd.Series | bool:
    """Whether each unit stores no cyclic level, true for a component that stores none."""
    return ~static[c._level_attrs[1]].astype(bool) if c._level_attrs else True


def _operational(
    s: NetworkSources,
    c: Components,
    scenario: Any,
    glc: pd.Series,
    static: pd.DataFrame,
) -> pd.Series:
    """One for each active unit of the row's carrier; storage only where non-cyclic."""
    if _counted_periods(s, glc).empty:
        return pd.Series(dtype=float)
    mask = (static["carrier"] == glc["carrier_attribute"]) & static["active"]
    return _ones(static, mask & _non_cyclic(c, static))


def _emissions(s: NetworkSources, scenario: Any, glc: pd.Series) -> pd.Series:
    carriers = of_scenario(s.n.components.carriers.static, s.n, scenario)
    emissions = carriers[glc["carrier_attribute"]]
    return emissions[emissions != 0]


def _primary(
    s: NetworkSources,
    c: Components,
    scenario: Any,
    glc: pd.Series,
    static: pd.DataFrame,
) -> pd.Series:
    """Return the carrier's attribute for each unit active in the row's period; storage only where non-cyclic."""
    if _counted_periods(s, glc).empty:
        return pd.Series(dtype=float)
    emissions = _emissions(s, scenario, glc)
    active = c._active_names(_stated(glc))
    mask = static["carrier"].isin(emissions.index) & static.index.isin(active)
    mask &= _non_cyclic(c, static)
    return static["carrier"][mask].map(emissions).astype(float)


def _tech_capacity(
    s: NetworkSources,
    c: Components,
    scenario: Any,
    glc: pd.Series,
    static: pd.DataFrame,
) -> pd.Series:
    """One for each extendable unit of the row's carrier at its bus, active in its period."""
    extendable = static[c._operational_attrs["nom_extendable"]].astype(bool)
    mask = (static["carrier"] == glc["carrier_attribute"]) & extendable
    mask &= static.index.isin(c._active_names(_stated(glc)))
    bus = glc.get("bus")
    if pd.notna(bus) and bus != "":
        column = "bus0" if c.name in s.n.branch_components else "bus"
        mask &= static[column] == str(bus)
    return _ones(static, mask)


def _transmission_set(
    s: NetworkSources, c: Components, glc: pd.Series, static: pd.DataFrame
) -> tuple[pd.Index, Any]:
    """Return the extendable branches of the row's carriers and the periods they must be active in."""
    listed = re.sub(r"[\[\]\(\)]", "", glc["carrier_attribute"]).split(",")
    carriers = [carrier.strip() for carrier in listed]
    period: int | list[int] | None = _stated(glc)
    if period is None and s.multi_invest:
        period = list(s.periods)
    extendable = static[c._operational_attrs["nom_extendable"]].astype(bool)
    mask = static["carrier"].isin(carriers) & extendable
    mask &= static.index.isin(c._active_names(period))
    return static.index[mask.to_numpy(dtype=bool)], period


def _volume(
    s: NetworkSources,
    c: Components,
    scenario: Any,
    glc: pd.Series,
    static: pd.DataFrame,
) -> pd.Series:
    """Return the length of each branch in the row's set."""
    names, _ = _transmission_set(s, c, glc, static)
    return static["length"].loc[names].astype(float)


def _expansion_cost(
    s: NetworkSources,
    c: Components,
    scenario: Any,
    glc: pd.Series,
    static: pd.DataFrame,
) -> pd.Series:
    """Return the capital cost of each branch in the row's set, period-weighted where the row spans all periods."""
    names, period = _transmission_set(s, c, glc, static)
    if names.empty:
        return pd.Series(dtype=float)
    cost = c.capital_cost
    if s.n.has_scenarios:
        cost = cost.xs(scenario, level="scenario")
    cost = cost.loc[names].astype(float)
    if not isinstance(period, list):
        return cost
    weights = s.n.investment_period_weightings["objective"]
    active = pd.DataFrame(
        {p: names.isin(c._active_names(p)) for p in period}, index=names
    )
    return cost * (active @ weights.loc[period])


#: The units each type of row sums, by the weight the spec names it with.
_WEIGHTS: dict[str, tuple[str, Cell]] = {
    "operational_limit_weight": ("operational_limit", _operational),
    "primary_energy_weight": ("primary_energy", _primary),
    "tech_capacity_weight": ("tech_capacity_expansion_limit", _tech_capacity),
    "volume_weight": ("transmission_volume_expansion_limit", _volume),
    "expansion_cost_weight": ("transmission_expansion_cost_limit", _expansion_cost),
}


@source(*(f"Unit_{suffix}" for suffix in _WEIGHTS))
def _weight(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    """Return a row's weight of each unit; a curved unit's primary energy is read off its curve."""
    suffix = name.removeprefix(f"{c.name}_")
    da = _unit_grid(s, name, c, *_WEIGHTS[suffix])
    curve = f"{c.name}_efficiency_curve_p"
    if suffix == "primary_energy_weight" and curve in s:
        curved = s[curve].notnull().any("breakpoint")
        da = da / s.da(c, "efficiency").where(~curved, 1)
    return da


@source("GlobalConstraint_type")
def _type(s: NetworkSources, name: str) -> xr.DataArray:
    types = _rows(s, "type")
    varies = types.nunique(axis=1) > 1
    if varies.any():
        msg = f"GlobalConstraint {varies.index[varies].tolist()} change type across scenarios."
        raise ValueError(msg)
    return _grid(s, types).isel(scenario=0, drop=True).astype(object)


@source("GlobalConstraint_sense")
def _sense(s: NetworkSources, name: str) -> xr.DataArray:
    return _grid(s, _rows(s, "sense").astype(object))


@source("GlobalConstraint_counts_snapshot")
def _counts_snapshot(s: NetworkSources, name: str) -> xr.DataArray:
    stated = _grid(s, _periods(s))
    return stated.isnull() | (s.over_snapshots(s.period_of) == stated)


def _storage_fold(
    s: NetworkSources, cell: Cell, scenario: Any, glc: pd.Series
) -> float:
    """Return the initial charge a row's storage draws from, weighted as the row counts it."""
    periods = _counted_periods(s, glc)
    if periods.empty:
        return 0.0
    years = float(s["period_weight_years"].sel(period=list(periods)).sum())
    fold = 0.0
    for c in s.units.values():
        if not c._level_attrs:
            continue
        static = of_scenario(c.static, s.n, scenario)
        weight = cell(s, c, scenario, glc, static)
        initial = c._level_attrs[0]
        level = static[initial].loc[weight.index].astype(float)
        per_period = static[f"{initial}_per_period"].loc[weight.index].astype(bool)
        times = np.where(per_period & s.multi_invest, years, 1.0)
        fold += float((weight * level * times).sum())
    return fold


@source("GlobalConstraint_constant")
def _constant(s: NetworkSources, name: str) -> xr.DataArray:
    constant = _rows(s, "constant").astype(float)
    for kind, cell in (
        ("operational_limit", _operational),
        ("primary_energy", _primary),
    ):
        for i, j, scenario, glc in _cells(s, kind):
            constant.iat[i, j] -= _storage_fold(s, cell, scenario, glc)
    return _grid(s, constant)


def _carrier_limit(s: NetworkSources, attr: str) -> xr.DataArray:
    """Return a carrier attribute, the least over the scenarios."""
    column = s.n.components.carriers.static[attr]
    if s.n.has_scenarios:
        column = column.groupby(level="name").min()
    labels = s.coords("carrier")
    values = column.reindex(labels).to_numpy(dtype=float)
    coords = {"carrier": labels.to_numpy(dtype=object)}
    return xr.DataArray(values, coords=coords, dims="carrier")


@source("Carrier_max_growth")
def _max_growth(s: NetworkSources, name: str) -> xr.DataArray:
    if not s.multi_invest:
        return s.filled(name, np.nan)
    growth = _carrier_limit(s, "max_growth")
    return growth.where(growth != np.inf)


@source("Carrier_max_relative_growth")
def _max_relative_growth(s: NetworkSources, name: str) -> xr.DataArray:
    return _carrier_limit(s, "max_relative_growth").fillna(0.0)


@source("global_constraint")
def _global_constraint(s: NetworkSources, name: str) -> pd.Index:
    """Return the rows PyPSA builds: every `flow_based` row and every other row whose set holds a unit."""
    labels = pd.Index(s.n.components.global_constraints.names, dtype=object)
    types = _rows(s, "type", labels)
    built = set(labels[(types == "flow_based").any(axis=1)])
    for suffix, (kind, cell) in _WEIGHTS.items():
        summed = [c for c in s.units.values() if f"{c.name}_{suffix}" in s]
        for i, _, scenario, glc in _cells(s, kind, labels):
            if labels[i] in built:
                continue
            for c in summed:
                static = of_scenario(c.static, s.n, scenario)
                units = cell(s, c, scenario, glc, static).index
                if units.isin(s.names(c)).any():
                    built.add(labels[i])
                    break
    return labels[labels.isin(built)]
