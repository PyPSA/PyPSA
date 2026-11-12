# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Build PyPSA's linopy model from the spec and the network's data."""

from __future__ import annotations

import functools
import re
from importlib.resources import files
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr
from linopy import Model
from linopy.constants import BREAKPOINT_DIM
from linopy.constraints import Constraint
from linopy.expressions import LinearExpression
from linopy.variables import Variable

from pypsa.optimization.common import get_bus_counts
from pypsa.optimization.scaling import KVL, check_sub_networks, global_constraint_factor
from pypsa.optimization.spec._passive_network import (
    SecurityGroup,
    flow_based_zones,
    security_groups,
)
from pypsa.optimization.spec._render import formulated, render
from pypsa.optimization.spec.checks import check
from pypsa.optimization.spec.sources import NetworkSources

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mathspec.spec import Spec

    from pypsa import Network
    from pypsa.optimization.piecewise import PiecewiseOptions

SPEC = files("pypsa.optimization.spec")


@functools.cache
def _unit_dims() -> set[str]:
    """Return the spec's unit dims, one per component with a formulation."""
    import pypsa  # noqa: PLC0415

    return {c._unit_dim for c in formulated(pypsa.Network())}


@functools.cache
def spec(
    linearized_unit_commitment: bool = False,
    piecewise_options: tuple[PiecewiseOptions, ...] = (),
) -> Spec:
    """Return the spec of `n.optimize()`, patched for linearised unit commitment and piecewise options on request."""
    import mathspec  # noqa: PLC0415

    base = mathspec.to_spec(render(SPEC))
    patches: list[Any] = []
    if linearized_unit_commitment:
        patches.append(render(SPEC / "linearized_uc"))
    if piecewise_options:
        patches.append(_piecewise_patch(base, piecewise_options))
    return mathspec.override(base, patches) if patches else base


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


def pypsa_name(name: str, description: str | None, mode: str | None = None) -> str:
    """Return the PyPSA name of a spec entry: the backticked name its description opens with.

    An entry PyPSA names by loss mode lists one name per mode; `mode` picks it.
    """
    head = (description or "").split(" — ", 1)[0]
    found = re.findall(r"`([^`]+)`", head)
    if not found:
        return name
    return next((f for f in found if mode and f"_{mode}" in f), found[0])


#: PyPSA's row factor of Kirchhoff's voltage law, whose `x_pu_eff` weights are tiny.
KVL_SCALING = 1e5
#: Spec entries whose PyPSA name is not the opening of their description.
RENAMED = {"GlobalConstraint_flow_based_ub": "GlobalConstraint-ram"}
#: Dims PyPSA names otherwise on one entry.
DIMS = {"Carrier-growth_limit": {"carrier": "Carrier", "period": "periods"}}


def _pypsa_dims(dims: tuple[str, ...], constraint: bool) -> dict[str, str]:
    """Return the renaming of spec dims to PyPSA's: the one unit dim becomes `name`.

    A constraint over buses or GlobalConstraint labels names them `name` too.
    The breakpoints of a curve take linopy's name, which marks an auxiliary entry.
    """
    named = _unit_dims() | ({"bus", "global_constraint"} if constraint else set())
    found = [d for d in dims if d in named]
    renamed = {found[0]: "name"} if len(found) == 1 else {}
    return renamed | ({"breakpoint": BREAKPOINT_DIM} if "breakpoint" in dims else {})


def _order(dims: tuple[str, ...], has_scenarios: bool) -> list[str]:
    """PyPSA's dim order, the other dims after it in their own order.

    An entry over snapshots leads with `snapshot` and `name`, or with
    `scenario`, `name` and `snapshot` in a stochastic run; one without
    snapshots leads with `name` and `scenario`.
    """
    if "snapshot" not in dims:
        lead = ["name", "scenario"]
    elif has_scenarios:
        lead = ["scenario", "name", "snapshot"]
    else:
        lead = ["snapshot", "name"]
    front = [d for d in lead if d in dims]
    return front + [d for d in dims if d not in front]


def _coords_in_order(ds: xr.Dataset) -> xr.Dataset:
    """`ds` with its coords listed in the order of its dims, which linopy prints and derives from."""
    dims = list(ds.labels.dims)
    names = [d for d in dims if d in ds.coords]
    names += [k for k in ds.coords if k not in names]
    coords = xr.Coordinates(
        {k: ds.coords[k].variable for k in names},
        indexes={k: ds.xindexes[k] for k in names if k in ds.xindexes},
    )
    return ds.drop_vars(names).assign_coords(coords)


def _renumbered(ds: xr.Dataset, order: list[str], mapping: np.ndarray) -> xr.Dataset:
    """`ds` laid out in `order`, with fresh labels, trimmed to its built units.

    PyPSA builds a variable or a constraint only for the units that have
    one, so a unit whose every label is masked is dropped. linopy reads a
    solution or a dual by reshaping one C-ordered label block, so the kept
    labels are renumbered from the block's start; `mapping` records old to
    new.
    """
    if "name" in ds.labels.dims and ds.sizes["name"]:
        others = [d for d in ds.labels.dims if d != "name"]
        ds = ds.isel(name=(ds.labels != -1).any(others).values)
    ds = _coords_in_order(ds.transpose(*order, ...))
    start = ds.attrs["label_range"][0]
    fresh = start + np.arange(ds.labels.size).reshape(ds.labels.shape)
    moved = ds.labels.values
    live = moved != -1
    mapping[moved[live]] = fresh[live]
    ds["labels"] = ds.labels.copy(data=np.where(live, fresh, -1))
    ds.attrs["label_range"] = (start, start + ds.labels.size)
    return ds


#: linopy's suffix of a piecewise row or column, where the spec's expansion names it otherwise.
PIECEWISE_SUFFIX = {"order": "order_binary", "cost": "output_link"}
#: The dim a `{x}` placeholder in a PyPSA name splits a constraint along.
SPLITS = {"{s}": "scenario", "{k}": "segment"}
#: The placeholder suffix of the security rows, which PyPSA names per component and sub-network.
SECURITY = "-for-{c}-outage-in-sub-network-{n}"
#: The port dims a `{port}` placeholder in a PyPSA name splits along, with the unit dim of each part.
PORTS = {"link_output": "link", "process_output": "process"}


Part = tuple[LinearExpression, xr.Dataset]


def _cells(con: Constraint) -> xr.Dataset:
    """Return the fields of `con` on its grid, without the terms, which `con.data` densifies."""
    fields = {"sign": con.sign, "rhs": con.rhs, "labels": con.labels}
    return xr.Dataset(fields | {"scaling": con.scaling}, attrs=dict(con.attrs))


def _constraint(
    m: Model, lhs: LinearExpression, cells: xr.Dataset, name: str
) -> Constraint:
    """Return the constraint of `lhs` and the fields in `cells`, on the store of `lhs`."""
    fields = {str(k): v for k, v in cells.data_vars.items()}
    return Constraint(lhs, model=m, name=name, attrs=cells.attrs, **fields)


def _part(con: Constraint, indexers: dict[str, Any], drop: bool = False) -> Part:
    """Return the lhs and the fields of `con` at the positions `indexers`."""
    return con.lhs.isel(indexers, drop=drop), _cells(con).isel(indexers, drop=drop)


def _terms(con: Constraint) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the flat cell position, column and coefficient of each term of `con`, masked cells included.

    `Constraint.to_polars` reads a sparse lhs without densifying it, but by
    label and for the live cells only; so each cell is labelled by its
    position first.
    """
    positions = _positions(con.labels)
    df = Constraint(con.lhs, con.sign, con.rhs, con.model, labels=positions).to_polars()
    return df["labels"].to_numpy(), df["vars"].to_numpy(), df["coeffs"].to_numpy()


def _on_units[T: (xr.Dataset, LinearExpression)](
    obj: T, dim: str, ports: np.ndarray, unit: str, units: np.ndarray
) -> T:
    """Return `obj` at the `ports` of `dim`, laid over the `units` they belong to."""
    return obj.sel({dim: ports}).rename({dim: unit}).assign_coords({unit: units})


def _positions(da: xr.DataArray) -> xr.DataArray:
    """Return the flat position of each cell of `da`."""
    return da.copy(data=np.arange(da.size).reshape(da.shape))


def _relaid(
    con: Constraint, cells: xr.Dataset, variable_map: np.ndarray
) -> LinearExpression:
    """Return the lhs of `con` on the grid of `cells`, its columns renumbered by `variable_map`.

    `cells.cell` holds the old flat position of each new cell. linopy
    lays a sparse lhs out on its grid only by its terms' cells, so the
    terms move cell by cell.
    """
    rows, columns, coeffs = _terms(con)
    moved = np.full(con.labels.size, -1)
    moved[cells.cell.values.ravel()] = np.arange(cells.cell.size)
    keep = moved[rows] != -1
    dims = cells.labels.dims
    lhs = LinearExpression.from_coo(
        con.model,
        {d: cells.indexes[d] for d in dims},
        moved[rows[keep]],
        variable_map[columns[keep]],
        coeffs[keep],
    )
    aux = {k: v for k, v in cells.coords.items() if k not in dims}
    return lhs.assign_coords(aux) if aux else lhs


def _split_ports(m: Model, names: dict, sources: NetworkSources) -> None:
    """Split each entry PyPSA names per port number into one entry per number, over the units.

    The parts tile the label block of the entry, so `_relabel` renumbers
    each within its own share.
    """
    for section, container in (
        ("variables", m.variables),
        ("constraints", m.constraints),
    ):
        for key in [k for k in container if "{port}" in names[section].get(k, "")]:
            entry = container.data.pop(key)
            dim = next(d for d in PORTS if d in entry.labels.dims)
            unit = PORTS[dim]
            ports = sources.ports(sources.units[unit])
            start = entry.attrs["label_range"][0]
            for number, group in ports.groupby("k"):
                selected = group["port"].to_numpy(dtype=object)
                units = group[unit].to_numpy(dtype=object)
                new = f"{key}{number}"
                names[section][new] = names[section][key].replace("{port}", str(number))
                if isinstance(entry, Variable):
                    part = _on_units(entry.data, dim, selected, unit, units)
                    size = part.labels.size
                    part.attrs["label_range"] = (start, start + size)
                    container.data[new] = Variable(part, m, new)
                else:
                    cells = _on_units(_cells(entry), dim, selected, unit, units)
                    size = cells.labels.size
                    cells = cells.assign_attrs(label_range=(start, start + size))
                    lhs = _on_units(entry.lhs, dim, selected, unit, units)
                    container.data[new] = _constraint(m, lhs, cells, new)
                start += size


def _fresh(m: Model, lhs: LinearExpression, cells: xr.Dataset, name: str) -> Constraint:
    """Return the constraint of `lhs` and `cells` with a fresh label block at the end of the model's."""
    live = cells.labels.values != -1
    start = m._cCounter
    size = cells.labels.size
    fresh = start + np.arange(size).reshape(cells.labels.shape)
    cells = cells.assign(labels=cells.labels.copy(data=np.where(live, fresh, -1)))
    m._cCounter += size
    return _constraint(
        m, lhs, cells.assign_attrs(label_range=(start, start + size)), name
    )


def _security_parts(
    n: Network, name: str, con: Constraint, groups: list[SecurityGroup]
) -> dict[str, Part]:
    """Return the security rows of `con` split by sub-network, period and outaged component.

    Each part holds the outages of one component in one sub-network as
    `<component>-outage`, and the fixed or the extendable branches of that
    sub-network, as PyPSA selects them.
    """
    component = name.split("-", 1)[0]
    extendable = n.components[component].extendables
    kind = name.split("-")[1] == "ext"
    names, outages = con.indexes["name"], con.indexes["outage"]
    parts = {}
    for group in groups:
        affected = group.branches[group.branches.get_level_values(0) == component]
        affected = affected.get_level_values(1)
        affected = affected[affected.isin(extendable) == kind]
        for outaged in group.outages.unique(0):
            labels = group.outages[group.outages.get_level_values(0) == outaged]
            indexers = {
                "snapshot": group.snapshots,
                "outage": outages.get_indexer(labels.get_level_values(1)),
                "name": names.get_indexer(names.intersection(affected, sort=False)),
            }
            lhs, cells = _part(con, indexers)
            if not (cells.labels != -1).any():
                continue
            new = name.replace("{c}", outaged).replace("{n}", group.suffix)
            renamed = {"outage": f"{outaged}-outage"}
            parts[new] = (lhs.rename(renamed), cells.rename(renamed))
    return parts


def _meshed(n: Network, thresholds: Sequence[int] | None) -> dict[str, pd.Index]:
    """PyPSA's split of the nodal balance by how many components a bus has attached."""
    counts = get_bus_counts(n).reindex(n.c.buses.names, fill_value=0)
    bounds: list[float] = [
        0,
        *sorted(set(thresholds if thresholds is not None else [30, 100, 400])),
    ]
    bounds.append(np.inf)
    return {
        f"Bus-meshed-{low}-nodal_balance" if low else "Bus-nodal_balance": counts.index[
            (counts > low) & (counts <= high)
        ]
        for low, high in zip(bounds, bounds[1:], strict=False)
    }


def _split(
    m: Model,
    n: Network,
    constraints: dict[str, Constraint],
    groups: list[SecurityGroup],
    meshed: dict[str, pd.Index],
) -> dict[str, Constraint]:
    """Split each constraint whose PyPSA name has a placeholder.

    A `{s}` or `{k}` splits along its dim; the security rows split by
    sub-network, period and outaged component; the nodal balance splits by
    how meshed a bus is. Each part gets a fresh label block.
    """
    out = {}
    for name, con in constraints.items():
        if SECURITY in name:
            for new, part in _security_parts(n, name, con, groups).items():
                out[new] = _fresh(m, *part, new)
            continue
        names = con.indexes.get("name")
        if name == "Bus-nodal_balance" and not names.equals(meshed[name]):
            for new, buses in meshed.items():
                kept = names.get_indexer(names.intersection(buses, sort=False))
                lhs, cells = _part(con, {"name": kept})
                if (cells.labels.values != -1).any():
                    out[new] = _fresh(m, lhs, cells, new)
            continue
        placeholder = next((p for p in SPLITS if p in name), None)
        if placeholder is None or SPLITS[placeholder] not in con.labels.dims:
            out[name] = con
            continue
        dim = SPLITS[placeholder]
        for position, label in enumerate(con.indexes[dim]):
            lhs, cells = _part(con, {dim: position}, drop=True)
            if not (cells.labels.values != -1).any():
                continue
            new = name.replace(placeholder, str(label))
            out[new] = _fresh(m, lhs, cells, new)
    return out


def _relabel(
    m: Model,
    n: Network,
    names: dict,
    snapshots: pd.Index,
    network_index: pd.Index,
    sources: NetworkSources,
    meshed: dict[str, pd.Index],
) -> None:
    """Rename every variable, constraint and dim of `m` as PyPSA names and lays them out.

    Entries without a single row are dropped, as PyPSA does not build them,
    and the placeholder scenario of a network without scenarios is squeezed.
    The spec's named expressions are dropped too: they keep the spec's dims
    and the labels before renumbering.
    """
    squeeze = not n.has_scenarios
    levels = (
        {
            name: ("snapshot", network_index.get_level_values(name).to_numpy())
            for name in network_index.names
        }
        if isinstance(network_index, pd.MultiIndex)
        and not isinstance(snapshots, pd.MultiIndex)
        else {}
    )
    variable_map = np.arange(m._xCounter)
    constraint_map = np.arange(m._cCounter)

    def frame(ds: xr.Dataset, constraint: bool, mapping: np.ndarray) -> xr.Dataset:
        ds = ds.rename(_pypsa_dims(tuple(ds.labels.dims), constraint))
        if squeeze and "scenario" in ds.dims:
            ds = ds.squeeze("scenario", drop=True)
        if "snapshot" in ds.dims:
            ds = ds.assign_coords({"snapshot": snapshots, **levels})
        return _renumbered(ds, _order(ds.labels.dims, n.has_scenarios), mapping)

    variables = {}
    for name, var in m.variables.items():
        if not (var.labels != -1).any():
            continue
        new = RENAMED.get(name, names["variables"].get(name, name))
        ds = var.data
        if new == "FlowBased-net_position":
            ds = ds.sel(bus=flow_based_zones(sources))
        variables[new] = Variable(frame(ds, False, variable_map), m, new)

    def remap(ds: xr.Dataset) -> xr.Dataset:
        vars_ = ds.vars.values
        ds["vars"] = ds.vars.copy(data=np.where(vars_ != -1, variable_map[vars_], -1))
        return ds

    constraints = {}
    for name, con in m.constraints.items():
        if not (con.labels != -1).any():
            continue
        new = RENAMED.get(name, names["constraints"].get(name, name))
        if con.lhs.is_sparse:
            cells = _cells(con).assign(cell=_positions(con.labels))
            cells = frame(cells, True, constraint_map).rename(DIMS.get(new, {}))
            lhs = _relaid(con, cells, variable_map)
            constraints[new] = _constraint(m, lhs, cells.drop_vars("cell"), new)
            continue
        ds = remap(frame(con.data, True, constraint_map)).rename(DIMS.get(new, {}))
        constraints[new] = Constraint.from_dataset(ds, m, new)

    constraints = _split(m, n, constraints, security_groups(sources), meshed)
    m.variables.data.clear()
    m.variables.data.update(variables)
    m.constraints.data.clear()
    m.constraints.data.update(constraints)
    m.variables._label_position_index = None
    m.constraints._label_position_index = None
    m.expressions.data.clear()
    objective = m.objective
    expression = objective.expression
    if expression.is_sparse:
        columns, coeffs = expression.linear_terms()
        rows = np.zeros(columns.size, dtype=int)
        terms = LinearExpression.from_coo(m, {}, rows, variable_map[columns], coeffs)
        remapped = terms + expression.const.item()
    else:
        remapped = type(expression)(remap(expression.data.copy()), m)
    m.add_objective(remapped, overwrite=True, sense=objective.sense)


GC_SENSES = re.compile(r"GlobalConstraint_(?!flow_based).+_(ub|lb|eq)")
GC = "global_constraint"


def _merged_terms(parts: list[xr.Dataset]) -> xr.Dataset:
    """Return the terms of the cells the parts hold, each cell live in exactly one part."""
    terms = max(p.sizes["_term"] for p in parts)
    padded = []
    for p in parts:
        p = p.pad(_term=(0, terms - p.sizes["_term"]))
        live = p.labels != -1
        p["coeffs"] = p.coeffs.where(live & p.vars.notnull(), 0).fillna(0)
        p["vars"] = p.vars.where(live, -1).fillna(-1).astype(int)
        padded.append(p)
    merged = padded[0].copy()
    for p in padded[1:]:
        merged["coeffs"] = merged.coeffs + p.coeffs
        merged["vars"] = xr.where(p.vars != -1, p.vars, merged.vars)
    return merged


def _merged(m: Model, parts: list[Part]) -> Part:
    """Return one lhs and one set of fields of the cells the parts hold, each live in one part."""
    cells = parts[0][1].copy()
    for _, p in parts[1:]:
        live = p.labels != -1
        for field in ("sign", "rhs", "labels"):
            cells[field] = xr.where(live, p[field], cells[field])
    if parts[0][0].is_sparse:
        masked = [e.where(c.labels != -1, 0) for e, c in parts]
        return sum(masked[1:], masked[0]), cells
    terms = [e.data[["coeffs", "vars"]].assign(labels=c.labels) for e, c in parts]
    merged = _merged_terms(terms)
    return LinearExpression(merged[["coeffs", "vars"]], m), cells


def _merge_global_constraints(m: Model, squeeze: bool) -> None:
    """Re-add the sense rows of each GlobalConstraint label as one `GlobalConstraint-<label>`.

    The spec states each type once per sense, while PyPSA names one
    constraint per label, whose sense may differ by scenario.
    """
    names = [k for k in m.constraints if GC_SENSES.fullmatch(k)]
    cons = [m.constraints[k] for k in names]
    labels: list = []
    for con in cons:
        built = (con.labels != -1).any([d for d in con.labels.dims if d != GC])
        labels += [g for g in built.indexes[GC][built.values] if g not in labels]
    for g in labels:
        live = [
            _part(con, {"global_constraint": con.indexes[GC].get_loc(g)}, drop=True)
            for con in cons
            if (con.labels.sel({GC: g}) != -1).any()
        ]
        lhs, cells = _merged(m, live)
        if squeeze and "scenario" in cells.dims:
            lhs = lhs.isel(scenario=0, drop=True)
            cells = cells.squeeze("scenario", drop=True)
        m.add_constraints(
            Constraint(lhs, cells.sign, cells.rhs, m),
            name=f"GlobalConstraint-{g}",
            mask=cells.labels != -1,
        )
    for k in names:
        m.remove_constraints(k)


#: Columns PyPSA never scales: unit commitment, modules, maintenance and angles.
UNSCALED = {
    "status",
    "start_up",
    "shut_down",
    "n_mod",
    "maintenance",
    "maintenance_start",
    "maintenance_status",
    "phase_shift",
}
#: The rows that take the geometric mean of the smallest and largest factor of the assets they sum.
GROWTH_LIMIT = "Carrier-growth_limit"


def _scale_columns(
    m: Model, n: Network, objective: float
) -> tuple[np.ndarray, np.ndarray]:
    """Give every column PyPSA's factor; return per label its bus factor and whether it is a cost.

    A cost column takes the objective factor. A continuous column of a
    component on a bus takes `Bus.scaling` of its bus; the weights and
    binaries of a piecewise curve, and the columns in `UNSCALED`, keep `1`.
    A column without a bus factor reads NaN.
    """
    bus = np.full(m._xCounter, np.nan)
    cost = np.zeros(m._xCounter, dtype=bool)
    attached = n.one_port_components | n.branch_components
    for name, var in m.variables.items():
        component, _, attr = name.partition("-")
        labels = var.labels.values
        live = labels != -1
        if component == "CVaR" or attr.endswith("cost_piecewise"):
            var.scaling = objective
            cost[labels[live]] = True
        elif (
            component in attached
            and "name" in var.labels.dims
            and attr not in UNSCALED
            and "_piecewise_" not in attr
            and not (var.attrs.get("binary") or var.attrs.get("integer"))
        ):
            var.scaling = n.c[component]._scaling(var.labels.indexes["name"])
            factor = var.scaling.broadcast_like(var.labels)
            bus[labels[live]] = factor.transpose(*var.labels.dims).values[live]
    return bus, cost


def _reduced(
    ufunc: np.ufunc, groups: np.ndarray, values: np.ndarray, size: int
) -> np.ndarray:
    """Return `values` reduced by `ufunc` per group, skipping NaN; NaN for a group with none."""
    out = np.full(size, np.nan)
    ufunc.at(out, groups, values)
    return out


def _scale_rows(
    m: Model, n: Network, bus: np.ndarray, cost: np.ndarray, objective: float
) -> None:
    """Give every constraint PyPSA's factor, read from the columns it holds.

    A global constraint takes `GlobalConstraint.scaling`. The Kirchhoff
    voltage law takes per row the largest factor of its branches, a growth
    limit per carrier the geometric mean of the smallest and largest factor
    of its assets. Otherwise a constraint holding a cost column takes the
    objective factor, one holding a column with a bus factor the factor of
    its own component, and every other keeps `1`. PyPSA decides per
    constraint, so a row masked out of a constraint takes its factor too.
    """
    attached = n.one_port_components | n.branch_components | {"Bus"}
    stated = n.c.global_constraints.names
    for name, con in m.constraints.items():
        component, _, rest = name.partition("-")
        if component == "GlobalConstraint" and rest in stated:
            con.scaling = global_constraint_factor(n, rest)
            continue
        rows, columns, coeffs = _terms(con)
        live = coeffs != 0
        rows, columns = rows[live], columns[live]
        factors = bus[columns]
        if name == KVL:
            largest = _reduced(np.fmax, rows, factors, con.labels.size)
            shaped = largest.reshape(con.labels.shape)
            con.scaling = con.labels.copy(data=shaped).fillna(1.0)
        elif name == GROWTH_LIMIT:
            carriers = con.indexes["Carrier"]
            axis = con.labels.dims.index("Carrier")
            carrier = np.unravel_index(rows, con.labels.shape)[axis]
            low = _reduced(np.fmin, carrier, factors, len(carriers))
            high = _reduced(np.fmax, carrier, factors, len(carriers))
            mean = xr.DataArray(np.sqrt(low * high), coords={"Carrier": carriers})
            con.scaling = mean.fillna(1.0)
        elif cost[columns].any():
            con.scaling = objective
        elif (
            component in attached
            and "name" in con.labels.dims
            and np.isfinite(factors).any()
        ):
            con.scaling = n.c[component]._scaling(con.labels.indexes["name"])


def _scale(m: Model, n: Network, objective: float) -> None:
    """Apply PyPSA's numerical scaling to the built model.

    The Kirchhoff voltage law rows carry PyPSA's 1e5 on their coefficients,
    so their duals keep PyPSA's units.
    """
    m.objective.scaling = objective
    if KVL in m.constraints:
        check_sub_networks(n)
        con = m.constraints[KVL]
        con.update(lhs=con.lhs * KVL_SCALING, rhs=con.rhs * KVL_SCALING)
    bus, cost = _scale_columns(m, n, objective)
    _scale_rows(m, n, bus, cost, objective)


def build_model(
    n: Network,
    network_index: pd.Index,
    model_index: pd.Index,
    *,
    linearized_unit_commitment: bool = False,
    model_kwargs: dict | None = None,
    meshed_thresholds: Sequence[int] | None = None,
    piecewise_options: Sequence[Any] = (),
    objective_scaling: float = 1.0,
    **options: Any,
) -> Model:
    """Build the model of `n` over the build's snapshots from the spec."""
    check(
        n,
        network_index,
        linearized_unit_commitment=linearized_unit_commitment,
        piecewise_options=piecewise_options,
        **options,
    )
    model_kwargs = model_kwargs or {}
    n.calculate_dependent_values()
    s = spec(linearized_unit_commitment, tuple(piecewise_options))
    sources = NetworkSources(n, s, network_index, model_index, **options)
    m = Model.from_spec(s, sources, **model_kwargs)
    mode = (options.get("transmission_losses") or {}).get("mode")
    names = {
        "variables": {k: pypsa_name(k, v.description) for k, v in s.variables.items()},
        "constraints": {
            k: pypsa_name(k, v.description, mode) for k, v in s.constraints.items()
        },
    }
    for entry, curve in s.piecewise.items():
        aux = pypsa_name(entry, curve.description)
        for section in names.values():
            for k in [*m.variables, *m.constraints]:
                if k.startswith(f"{entry}_"):
                    suffix = k.removeprefix(f"{entry}_")
                    section[k] = f"{aux}_{PIECEWISE_SUFFIX.get(suffix, suffix)}"
    _merge_global_constraints(m, squeeze=not n.has_scenarios)
    _split_ports(m, names, sources)
    _relabel(
        m, n, names, model_index, network_index, sources, _meshed(n, meshed_thresholds)
    )
    m.constraints.sanitize_zeros()
    _scale(m, n, objective_scaling)
    return m
