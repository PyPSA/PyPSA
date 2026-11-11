# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Build PyPSA's linopy model from `pypsa.yaml` and the network's data."""

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
from pypsa.optimization.scaling import KVL
from pypsa.optimization.spec._passive_network import (
    SecurityGroup,
    flow_based_zones,
    security_groups,
)
from pypsa.optimization.spec.checks import check
from pypsa.optimization.spec.sources import UNITS, NetworkSources

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mathspec.spec import Spec

    from pypsa import Network

UNIT_DIMS = set(UNITS.values())
SPEC = files("pypsa.optimization.spec")


@functools.cache
def spec(linearized_unit_commitment: bool = False) -> Spec:
    """Return the spec of `n.optimize()`, with the linearised unit commitment patch on request."""
    import mathspec  # noqa: PLC0415

    base = mathspec.to_spec((SPEC / "pypsa.yaml").read_text())
    if not linearized_unit_commitment:
        return base
    return mathspec.override(base, [(SPEC / "linearized_uc.yaml").read_text()])


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
    named = UNIT_DIMS | ({"bus", "global_constraint"} if constraint else set())
    found = [d for d in dims if d in named]
    renamed = {found[0]: "name"} if len(found) == 1 else {}
    return renamed | ({"breakpoint": BREAKPOINT_DIM} if "breakpoint" in dims else {})


def _order(dims: tuple[str, ...], has_scenarios: bool) -> list[str]:
    """PyPSA's dim order: `scenario` first in a stochastic run, else `snapshot` first."""
    lead = "scenario" if has_scenarios else "snapshot"
    return [d for d in dims if d == lead] + [d for d in dims if d != lead]


def _renumbered(
    ds: xr.Dataset, order: list[str], mapping: np.ndarray, trim: bool
) -> xr.Dataset:
    """`ds` laid out in `order`, with fresh labels, trimmed to its built units on request.

    PyPSA builds a variable only for the units that have one, so with `trim`
    a unit whose every label is masked is dropped; a constraint keeps it
    masked. linopy reads a solution or a dual by reshaping one C-ordered
    label block, so the kept labels are renumbered from the block's start;
    `mapping` records old to new.
    """
    if trim and "name" in ds.labels.dims and ds.sizes["name"]:
        others = [d for d in ds.labels.dims if d != "name"]
        ds = ds.isel(name=(ds.labels != -1).any(others).values)
    ds = ds.transpose(*order, ...)
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
            ds = container.data.pop(key).data
            dim = next(d for d in PORTS if d in ds.dims)
            ports = (
                sources.link_ports if dim == "link_output" else sources.process_ports
            )
            start = ds.attrs["label_range"][0]
            for number, group in ports.groupby("k"):
                part = ds.sel({dim: group["port"].to_numpy(dtype=object)})
                units = group[PORTS[dim]].to_numpy(dtype=object)
                part = part.rename({dim: PORTS[dim]}).assign_coords({PORTS[dim]: units})
                part.attrs["label_range"] = (start, start + part.labels.size)
                start += part.labels.size
                new = f"{key}{number}"
                names[section][new] = names[section][key].replace("{port}", str(number))
                container.data[new] = (
                    Variable(part, m, new)
                    if section == "variables"
                    else Constraint.from_dataset(part, m, new)
                )


def _fresh(m: Model, part: xr.Dataset, name: str) -> Constraint:
    """`part` as a constraint with a fresh, contiguous label block at the end of the model's."""
    live = part.labels.values != -1
    start = m._cCounter
    fresh = start + np.arange(part.labels.size).reshape(part.labels.shape)
    part["labels"] = part.labels.copy(data=np.where(live, fresh, -1))
    part.attrs["label_range"] = (start, start + part.labels.size)
    m._cCounter += part.labels.size
    return Constraint.from_dataset(part, m, name)


def _security_parts(
    n: Network, name: str, ds: xr.Dataset, groups: list[SecurityGroup]
) -> dict[str, xr.Dataset]:
    """Return the security rows of `ds` split by sub-network, period and outaged component.

    Each part holds the outages of one component in one sub-network as
    `<component>-outage`, and the fixed or the extendable branches of that
    sub-network, as PyPSA selects them.
    """
    component = name.split("-", 1)[0]
    extendable = n.components[component].extendables
    kind = name.split("-")[1] == "ext"
    parts = {}
    for group in groups:
        affected = group.branches[group.branches.get_level_values(0) == component]
        affected = affected.get_level_values(1)
        affected = affected[affected.isin(extendable) == kind]
        for outaged in group.outages.unique(0):
            labels = group.outages[group.outages.get_level_values(0) == outaged]
            part = ds.isel(snapshot=group.snapshots)
            part = part.sel(outage=list(labels.get_level_values(1)))
            part = part.sel(
                name=part.indexes["name"].intersection(affected, sort=False)
            )
            if not (part.labels != -1).any():
                continue
            new = name.replace("{c}", outaged).replace("{n}", group.suffix)
            parts[new] = part.rename(outage=f"{outaged}-outage")
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
            for new, part in _security_parts(n, name, con.data, groups).items():
                out[new] = _fresh(m, part, new)
            continue
        names = con.data.indexes.get("name")
        if name == "Bus-nodal_balance" and not names.equals(meshed[name]):
            for new, buses in meshed.items():
                part = con.data.sel(name=names.intersection(buses, sort=False))
                if (part.labels.values != -1).any():
                    out[new] = _fresh(m, part, new)
            continue
        placeholder = next((p for p in SPLITS if p in name), None)
        if placeholder is None or SPLITS[placeholder] not in con.labels.dims:
            out[name] = con
            continue
        dim = SPLITS[placeholder]
        ds = con.data
        for label in ds.indexes[dim]:
            part = ds.sel({dim: label}, drop=True)
            if not (part.labels.values != -1).any():
                continue
            new = name.replace(placeholder, str(label))
            out[new] = _fresh(m, part, new)
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
        return _renumbered(
            ds, _order(ds.labels.dims, n.has_scenarios), mapping, not constraint
        )

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
        ds = remap(frame(con.data, True, constraint_map)).rename(DIMS.get(new, {}))
        constraints[new] = Constraint.from_dataset(ds, m, new)

    constraints = _split(m, n, constraints, security_groups(sources), meshed)
    m.variables.data.clear()
    m.variables.data.update(variables)
    m.constraints.data.clear()
    m.constraints.data.update(constraints)
    m.variables._label_position_index = None
    m.constraints._label_position_index = None
    objective = m.objective
    expression = objective.expression
    m.add_objective(
        type(expression)(remap(expression.data.copy()), m),
        overwrite=True,
        sense=objective.sense,
    )


GC_SENSES = re.compile(r"GlobalConstraint_(?!flow_based).+_(ub|lb|eq)")


def _merged(parts: list[xr.Dataset]) -> xr.Dataset:
    """One dataset of the cells the parts hold, each cell live in exactly one part."""
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
        live = p.labels != -1
        for field in ("sign", "rhs", "labels"):
            merged[field] = xr.where(live, p[field], merged[field])
        merged["coeffs"] = merged.coeffs + p.coeffs
        merged["vars"] = xr.where(p.vars != -1, p.vars, merged.vars)
    return merged


def _merge_global_constraints(m: Model, squeeze: bool) -> None:
    """Re-add the sense rows of each GlobalConstraint label as one `GlobalConstraint-<label>`.

    The spec states each type once per sense, while PyPSA names one
    constraint per label, whose sense may differ by scenario.
    """
    names = [k for k in m.constraints if GC_SENSES.fullmatch(k)]
    parts = [m.constraints[k].data for k in names]
    labels: list = []
    for part in parts:
        built = (part.labels != -1).any(
            [d for d in part.labels.dims if d != "global_constraint"]
        )
        labels += [
            g
            for g in built.indexes["global_constraint"][built.values]
            if g not in labels
        ]
    for g in labels:
        live = [
            p.sel(global_constraint=g, drop=True)
            for p in parts
            if (p.labels.sel(global_constraint=g) != -1).any()
        ]
        ds = _merged(live)
        if squeeze and "scenario" in ds.dims:
            ds = ds.squeeze("scenario", drop=True)
        lhs = LinearExpression(ds[["coeffs", "vars"]], m)
        m.add_constraints(
            lhs, ds.sign, ds.rhs, name=f"GlobalConstraint-{g}", mask=ds.labels != -1
        )
    for k in names:
        m.remove_constraints(k)


def build_model(
    n: Network,
    network_index: pd.Index,
    model_index: pd.Index,
    *,
    linearized_unit_commitment: bool = False,
    model_kwargs: dict | None = None,
    meshed_thresholds: Sequence[int] | None = None,
    piecewise_options: Sequence[Any] = (),
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
    s = spec(linearized_unit_commitment)
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
    if KVL in m.constraints:
        m.constraints[KVL].scaling = KVL_SCALING
    return m
