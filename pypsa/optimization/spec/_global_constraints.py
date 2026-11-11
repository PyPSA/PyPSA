# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Data prep of `pypsa.yaml`: global constraints and carrier growth limits."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from pypsa.descriptors import nominal_attrs
from pypsa.optimization.spec.sources import (
    NO_SCENARIO,
    UNITS,
    NetworkSources,
    _filled,
    dimension,
    single,
)

Cell = Callable[[NetworkSources, str, Any, pd.Series, pd.DataFrame], pd.Series]

_CYCLIC = {"StorageUnit": "cyclic_state_of_charge", "Store": "e_cyclic"}
_INITIAL = {"StorageUnit": "state_of_charge_initial", "Store": "e_initial"}
_PER_PERIOD = {
    "StorageUnit": "state_of_charge_initial_per_period",
    "Store": "e_initial_per_period",
}


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


def _unit_static(s: NetworkSources, component: str, scenario: Any) -> pd.DataFrame:
    static = s.n.components[component].static
    if not s.n.has_scenarios:
        return static
    if static.empty:
        return static.droplevel("scenario")
    return static.xs(scenario, level="scenario")


def _unit_grid(
    s: NetworkSources, name: str, component: str, kind: str, cell: Cell
) -> xr.DataArray:
    """Return a (unit, row, scenario) grid, each cell a series over the units in its set."""
    dim = UNITS[component]
    units = s.coords(dim)
    rows = s.coords("global_constraint")
    values = np.full((len(units), len(rows), len(s.scenarios)), np.nan)
    for i, j, scenario, glc in _cells(s, kind):
        static = _unit_static(s, component, scenario)
        values[:, i, j] = cell(s, component, scenario, glc, static).reindex(units)
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


def _operational(
    s: NetworkSources,
    component: str,
    scenario: Any,
    glc: pd.Series,
    static: pd.DataFrame,
) -> pd.Series:
    """One for each active unit of the row's carrier; storage only where non-cyclic."""
    if _counted_periods(s, glc).empty:
        return pd.Series(dtype=float)
    mask = (static["carrier"] == glc["carrier_attribute"]) & static["active"]
    if component in _CYCLIC:
        mask &= ~static[_CYCLIC[component]].astype(bool)
    return _ones(static, mask)


def _emissions(s: NetworkSources, scenario: Any, glc: pd.Series) -> pd.Series:
    carriers = _unit_static(s, "Carrier", scenario)[glc["carrier_attribute"]]
    return carriers[carriers != 0]


def _primary(
    s: NetworkSources,
    component: str,
    scenario: Any,
    glc: pd.Series,
    static: pd.DataFrame,
) -> pd.Series:
    """Return the carrier's attribute for each unit active in the row's period; storage only where non-cyclic."""
    if _counted_periods(s, glc).empty:
        return pd.Series(dtype=float)
    emissions = _emissions(s, scenario, glc)
    active = s.n.components[component]._active_names(_stated(glc))
    mask = static["carrier"].isin(emissions.index) & static.index.isin(active)
    if component in _CYCLIC:
        mask &= ~static[_CYCLIC[component]].astype(bool)
    return static["carrier"][mask].map(emissions).astype(float)


def _tech_capacity(
    s: NetworkSources,
    component: str,
    scenario: Any,
    glc: pd.Series,
    static: pd.DataFrame,
) -> pd.Series:
    """One for each extendable unit of the row's carrier at its bus, active in its period."""
    nominal = nominal_attrs[component]
    active = s.n.components[component]._active_names(_stated(glc))
    mask = (static["carrier"] == glc["carrier_attribute"]) & static[
        f"{nominal}_extendable"
    ].astype(bool)
    mask &= static.index.isin(active)
    bus = glc.get("bus")
    if pd.notna(bus) and bus != "":
        column = "bus0" if component in s.n.branch_components else "bus"
        mask &= static[column] == str(bus)
    return _ones(static, mask)


def _transmission_set(
    s: NetworkSources, component: str, glc: pd.Series, static: pd.DataFrame
) -> tuple[pd.Index, Any]:
    """Return the extendable branches of the row's carriers and the periods they must be active in."""
    carriers = [
        c.strip()
        for c in re.sub(r"[\[\]\(\)]", "", glc["carrier_attribute"]).split(",")
    ]
    period = _stated(glc)
    if period is None and s.multi_invest:
        period = list(s.periods)
    nominal = nominal_attrs[component]
    active = s.n.components[component]._active_names(period)
    mask = static["carrier"].isin(carriers) & static[f"{nominal}_extendable"].astype(
        bool
    )
    mask &= static.index.isin(active)
    return static.index[mask.to_numpy(dtype=bool)], period


def _volume(
    s: NetworkSources,
    component: str,
    scenario: Any,
    glc: pd.Series,
    static: pd.DataFrame,
) -> pd.Series:
    """Return the length of each branch in the row's set."""
    names, _ = _transmission_set(s, component, glc, static)
    return static["length"].loc[names].astype(float)


def _expansion_cost(
    s: NetworkSources,
    component: str,
    scenario: Any,
    glc: pd.Series,
    static: pd.DataFrame,
) -> pd.Series:
    """Return the capital cost of each branch in the row's set, period-weighted where the row spans all periods."""
    names, period = _transmission_set(s, component, glc, static)
    if names.empty:
        return pd.Series(dtype=float)
    c = s.n.components[component]
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


_WEIGHTS: dict[str, tuple[str, Cell, tuple[str, ...]]] = {
    "operational_limit_weight": (
        "operational_limit",
        _operational,
        ("Generator", "StorageUnit", "Store"),
    ),
    "primary_energy_weight": (
        "primary_energy",
        _primary,
        ("Generator", "StorageUnit", "Store"),
    ),
    "tech_capacity_weight": (
        "tech_capacity_expansion_limit",
        _tech_capacity,
        ("Generator", "Line", "Link", "Process", "StorageUnit", "Store"),
    ),
    "volume_weight": ("transmission_volume_expansion_limit", _volume, ("Line", "Link")),
    "expansion_cost_weight": (
        "transmission_expansion_cost_limit",
        _expansion_cost,
        ("Line", "Link"),
    ),
}


def _weight(s: NetworkSources, name: str) -> xr.DataArray:
    component, _, suffix = name.partition("_")
    kind, cell, _ = _WEIGHTS[suffix]
    da = _unit_grid(s, name, component, kind, cell)
    if name == "Generator_primary_energy_weight":
        curved = s["Generator_efficiency_curve_p"].notnull().any("breakpoint")
        da = da / s.da("Generator", "efficiency").where(~curved, 1)
    return s.frame(name, da)


for _suffix, (_, _, _components) in _WEIGHTS.items():
    single(*(f"{c}_{_suffix}" for c in _components))(_weight)


@single("GlobalConstraint_type")
def _type(s: NetworkSources, name: str) -> xr.DataArray:
    types = _rows(s, "type")
    varies = types.nunique(axis=1) > 1
    if varies.any():
        msg = f"GlobalConstraint {varies.index[varies].tolist()} change type across scenarios."
        raise ValueError(msg)
    first = types.iloc[:, 0]
    return s.frame(
        name,
        xr.DataArray(
            first.to_numpy(dtype=object),
            coords={"global_constraint": first.index.to_numpy(dtype=object)},
            dims="global_constraint",
        ),
    )


@single("GlobalConstraint_sense")
def _sense(s: NetworkSources, name: str) -> xr.DataArray:
    return s.frame(name, _grid(s, _rows(s, "sense").astype(object)))


@single("GlobalConstraint_counts_snapshot")
def _counts_snapshot(s: NetworkSources, name: str) -> xr.DataArray:
    stated = _grid(s, _periods(s))
    of = xr.DataArray(s.period_of, coords={"snapshot": s.snapshots}, dims="snapshot")
    return s.frame(name, stated.isnull() | (of == stated))


def _storage_fold(
    s: NetworkSources, cell: Cell, scenario: Any, glc: pd.Series
) -> float:
    """Return the initial charge a row's storage draws from, weighted as the row counts it."""
    periods = _counted_periods(s, glc)
    if periods.empty:
        return 0.0
    years = float(s.period_weight("years").sel(period=list(periods)).sum())
    fold = 0.0
    for component in _CYCLIC:
        static = _unit_static(s, component, scenario)
        weight = cell(s, component, scenario, glc, static)
        initial = static[_INITIAL[component]].loc[weight.index].astype(float)
        per_period = static[_PER_PERIOD[component]].loc[weight.index].astype(bool)
        times = np.where(per_period & s.multi_invest, years, 1.0)
        fold += float((weight * initial * times).sum())
    return fold


@single("GlobalConstraint_constant")
def _constant(s: NetworkSources, name: str) -> xr.DataArray:
    constant = _rows(s, "constant").astype(float)
    for kind, cell in (
        ("operational_limit", _operational),
        ("primary_energy", _primary),
    ):
        for i, j, scenario, glc in _cells(s, kind):
            constant.iat[i, j] -= _storage_fold(s, cell, scenario, glc)
    return s.frame(name, _grid(s, constant))


def _carrier_limit(s: NetworkSources, attr: str) -> pd.Series:
    """Return a carrier attribute, the least over the scenarios."""
    column = s.n.components.carriers.static[attr]
    if s.n.has_scenarios:
        column = column.groupby(level="name").min()
    return column.reindex(s.coords("carrier"))


def _carrier_grid(s: NetworkSources, name: str, values: pd.Series) -> xr.DataArray:
    return s.frame(
        name,
        xr.DataArray(
            values.to_numpy(dtype=float),
            coords={"carrier": s.coords("carrier").to_numpy(dtype=object)},
            dims="carrier",
        ),
    )


@single("Carrier_max_growth")
def _max_growth(s: NetworkSources, name: str) -> xr.DataArray:
    if not s.multi_invest:
        return _filled(s, name, np.nan)
    growth = _carrier_limit(s, "max_growth").replace(np.inf, np.nan)
    return _carrier_grid(s, name, growth)


@single("Carrier_max_relative_growth")
def _max_relative_growth(s: NetworkSources, name: str) -> xr.DataArray:
    return _carrier_grid(s, name, _carrier_limit(s, "max_relative_growth").fillna(0.0))


@dimension("global_constraint")
def _global_constraint(s: NetworkSources) -> pd.Index:
    """Return the rows PyPSA builds: every `flow_based` row and every other row whose set holds a unit."""
    labels = pd.Index(s.n.components.global_constraints.names, dtype=object)
    types = _rows(s, "type", labels)
    built = set(labels[(types == "flow_based").any(axis=1)])
    for kind, cell, components in _WEIGHTS.values():
        for i, _, scenario, glc in _cells(s, kind, labels):
            if labels[i] in built:
                continue
            for component in components:
                static = _unit_static(s, component, scenario)
                units = cell(s, component, scenario, glc, static).index
                if units.isin(s.names(component)).any():
                    built.add(labels[i])
                    break
    return labels[labels.isin(built)]
