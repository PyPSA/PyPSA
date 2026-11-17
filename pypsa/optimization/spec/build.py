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
from linopy.constants import (
    BREAKPOINT_DIM,
    PWL_ACTIVE_BOUND_SUFFIX,
    PWL_BINARY_ORDER_SUFFIX,
    PWL_CHORD_SUFFIX,
    PWL_DELTA_BOUND_SUFFIX,
    PWL_FILL_ORDER_SUFFIX,
    PWL_OUTPUT_LINK_SUFFIX,
)
from linopy.constraints import Constraint
from linopy.expressions import LinearExpression
from linopy.variables import Variable

from pypsa.components.common import invariant
from pypsa.optimization.common import get_bus_counts
from pypsa.optimization.scaling import (
    KVL,
    check_sub_networks,
    global_constraint_factors,
)
from pypsa.optimization.spec._passive_network import (
    SecurityGroup,
    cycle_scaling,
    security_groups,
)
from pypsa.optimization.spec._render import (
    Names,
    Scalings,
    formulated,
    renamer,
    render,
)
from pypsa.optimization.spec.checks import check
from pypsa.optimization.spec.sources import NetworkSources

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from importlib.resources.abc import Traversable

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
def _rendered(linearized_unit_commitment: bool) -> tuple[Spec, Names, Scalings]:
    """Return the spec, its PyPSA names and its scaling rules, rendered once per process."""
    return render(SPEC, _patch(linearized_unit_commitment))


@functools.cache
def spec(
    linearized_unit_commitment: bool = False,
    piecewise_options: tuple[PiecewiseOptions, ...] = (),
) -> Spec:
    """Return the spec of `n.optimize()`, patched for linearised unit commitment and piecewise options on request."""
    import mathspec  # noqa: PLC0415

    base = _rendered(linearized_unit_commitment)[0]
    if not piecewise_options:
        return base
    return mathspec.override(base, [_piecewise_patch(base, piecewise_options)])


def _patch(linearized_unit_commitment: bool) -> Traversable | None:
    """Return the patch folder of linearised unit commitment, where asked for."""
    return SPEC / "linearized_uc" if linearized_unit_commitment else None


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


@functools.cache
def spec_names(linearized_unit_commitment: bool = False) -> Names:
    """Return the PyPSA names the spec states, patched for linearised unit commitment on request."""
    return _rendered(linearized_unit_commitment)[1]


@functools.cache
def spec_scalings(linearized_unit_commitment: bool = False) -> Scalings:
    """Return the scaling rules the spec states, patched for linearised unit commitment on request."""
    return _rendered(linearized_unit_commitment)[2]


def pypsa_name(name: str, named: dict, mode: str | None = None) -> str:
    """Return the PyPSA name of a spec entry: its `name:`, or its key where it states none.

    An entry PyPSA names by loss mode states one name per mode; `mode` picks it.
    """
    found = named.get(name, name)
    return found if isinstance(found, str) else found.get(mode, name)


def _renamer(s: Spec, named: Names, mode: str | None) -> Callable[[str], str]:
    """Return the PyPSA name of each entry linopy builds from `s`.

    An entry a piecewise curve expands into takes the curve's PyPSA name
    and linopy's suffix of its kind.
    """
    curves = {f"{k}_": pypsa_name(k, named["piecewise"]) for k in s.piecewise}

    def rename(key: str) -> str:
        for prefix, curve in curves.items():
            if key.startswith(prefix):
                suffix = key.removeprefix(prefix)
                return f"{curve}_{PIECEWISE_SUFFIX.get(suffix, suffix)}"
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
    named = _unit_dims() | ({"bus", "global_constraint"} if constraint else set())
    found = [d for d in dims if d in named]
    renamed = {found[0]: "name"} if len(found) == 1 else {}
    return renamed | ({"breakpoint": BREAKPOINT_DIM} if "breakpoint" in dims else {})


#: linopy's suffix of a piecewise row or column, where the spec's expansion names it otherwise.
PIECEWISE_SUFFIX = {"order": "order_binary", "cost": "output_link"}
#: The dim a `{x}` placeholder in a PyPSA name splits a constraint along.
SPLITS = {"{s}": "scenario", "{k}": "segment"}
#: The placeholder suffix of the security rows, which PyPSA names per component and sub-network.
SECURITY = "-for-{c}-outage-in-sub-network-{n}"
#: The port dims a `{port}` placeholder in a PyPSA name splits along, with the unit dim of each part.
PORTS = {"Link_port": "Link", "Process_port": "Process"}


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


def _on_units[T: (xr.Dataset, LinearExpression)](
    obj: T, dim: str, ports: np.ndarray, unit: str, units: np.ndarray
) -> T:
    """Return `obj` at the `ports` of `dim`, laid over the `units` they belong to."""
    return obj.sel({dim: ports}).rename({dim: unit}).assign_coords({unit: units})


def _split_ports(m: Model, sources: NetworkSources) -> None:
    """Split each entry PyPSA names per port number into one entry per number, over the units.

    linopy frames an entry in the name order of its dims, so the port dim
    leads and the ports of one number, which the sources list together,
    hold one run of the entry's label block: each part keeps its labels.

    Raises:
        ValueError: The port dim does not lead the entry.

    """
    for container in (m.variables, m.constraints):
        for key in [k for k in container if "{port}" in k]:
            entry = container[key]
            dim = next(d for d in PORTS if d in entry.labels.dims)
            if entry.labels.dims[0] != dim:
                msg = f"{key} must lead with {dim} to split by port, not {entry.labels.dims[0]}."
                raise ValueError(msg)
            container.remove(key)
            unit = PORTS[dim]
            ports = sources.ports(sources.units[unit])
            start = entry.attrs["label_range"][0]
            for number, group in ports.groupby("k"):
                selected = group["port"].to_numpy(dtype=object)
                units = group[unit].to_numpy(dtype=object)
                new = key.replace("{port}", str(number))
                if isinstance(entry, Variable):
                    part = _on_units(entry.data, dim, selected, unit, units)
                    size = part.labels.size
                    part.attrs["label_range"] = (start, start + size)
                    container.add(Variable(part, m, new))
                else:
                    cells = _on_units(_cells(entry), dim, selected, unit, units)
                    size = cells.labels.size
                    cells = cells.assign_attrs(label_range=(start, start + size))
                    lhs = _on_units(entry.lhs, dim, selected, unit, units)
                    container.add(_constraint(m, lhs, cells, new))
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
    *,
    snapshots: pd.Index,
    network_index: pd.Index,
    sources: NetworkSources,
    meshed: dict[str, pd.Index],
) -> None:
    """Rename every dim of `m` as PyPSA names it, in linopy's order and with linopy's labels.

    Entries without a single row are removed, as PyPSA does not build them,
    and the placeholder scenario of a network without scenarios is squeezed.
    A unit without a row stays as a masked coordinate.
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

    def frame[T: (Variable, LinearExpression, xr.Dataset)](
        obj: T, dims: tuple[str, ...], renamed: dict[str, str]
    ) -> T:
        if renamed:
            obj = obj.rename(renamed)
        if squeeze and "scenario" in dims:
            obj = obj.isel(scenario=0, drop=True)
        if "snapshot" in dims:
            obj = obj.assign_coords({"snapshot": snapshots, **levels})
        return obj

    for name, var in list(m.variables.items()):
        if not (var.labels != -1).any():
            m.variables.remove(name)
            continue
        dims = tuple(var.labels.dims)
        m.variables.add(frame(var, dims, _pypsa_dims(dims, False)))

    constraints = {}
    for name, con in m.constraints.items():
        if not (con.labels != -1).any():
            continue
        dims = tuple(con.labels.dims)
        renamed = _pypsa_dims(dims, True) | DIMS.get(name, {})
        lhs = frame(con.lhs, dims, renamed)
        constraints[name] = _constraint(m, lhs, frame(_cells(con), dims, renamed), name)
    constraints = _split(m, n, constraints, security_groups(sources), meshed)
    for name in list(m.constraints):
        m.constraints.remove(name)
    for con in constraints.values():
        m.constraints.add(con)


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
#: linopy's rows of a piecewise curve that hold only its weights and binaries.
WEIGHT_ROWS = {
    suffix.removeprefix("_")
    for suffix in (
        PWL_ACTIVE_BOUND_SUFFIX,
        PWL_BINARY_ORDER_SUFFIX,
        PWL_DELTA_BOUND_SUFFIX,
        PWL_FILL_ORDER_SUFFIX,
    )
}
#: linopy's rows of a cost curve that hold the cost besides those of its `cost` link.
COST_ROWS = {s.removeprefix("_") for s in (PWL_CHORD_SUFFIX, PWL_OUTPUT_LINK_SUFFIX)}
#: The placeholder of a PyPSA name, which a split replaces by a label.
PLACEHOLDER = re.compile(r"\\\{\w+\\\}")


def _scale_columns(m: Model, n: Network, objective: float) -> None:
    """Give every column PyPSA's factor.

    A cost column takes the objective factor. A continuous column of a
    component on a bus takes `Bus.scaling` of its bus; the weights and
    binaries of a piecewise curve, and the columns in `UNSCALED`, keep `1`.
    """
    attached = n.one_port_components | n.branch_components
    for name, var in m.variables.items():
        component, _, attr = name.partition("-")
        if component == "CVaR" or attr.endswith("cost_piecewise"):
            var.scaling = objective
        elif (
            component in attached
            and "name" in var.labels.dims
            and attr not in UNSCALED
            and "_piecewise_" not in attr
            and not (var.attrs.get("binary") or var.attrs.get("integer"))
        ):
            var.scaling = n.c[component]._scaling(var.labels.indexes["name"])


def _rules(
    scalings: Scalings, rename: Callable[[str], str]
) -> list[tuple[re.Pattern, str]]:
    """Return the scaling rule of each entry, matched by its PyPSA name or the names it splits into."""
    rules = []
    for key, rule in scalings.items():
        name = re.escape(rename(key))
        rules.append((re.compile(PLACEHOLDER.sub(".+", name)), rule))
    return rules


def _rule(
    name: str, con: Constraint, n: Network, rules: list[tuple[re.Pattern, str]]
) -> str:
    """Return the scaling rule of the constraint `name`: its `scaling:`, else one by its kind.

    A row of a piecewise curve that holds only its weights and binaries
    takes `none`, one of a cost curve that holds the cost `objective`. A
    row over the units of a component on a bus takes `bus`, any other
    `none`.
    """
    for pattern, rule in rules:
        if pattern.fullmatch(name):
            return rule
    curve, piecewise, kind = name.rpartition("_piecewise_")
    if piecewise and kind in WEIGHT_ROWS:
        return "none"
    cost = kind in COST_ROWS or kind.startswith("cost")
    if piecewise and curve.endswith("cost") and cost:
        return "objective"
    component = name.partition("-")[0]
    attached = n.one_port_components | n.branch_components | {"Bus"}
    return "bus" if component in attached and "name" in con.labels.dims else "none"


def _carrier_factors(
    n: Network, sources: NetworkSources, carriers: pd.Index
) -> xr.DataArray:
    """Return per carrier the geometric mean of the smallest and largest factor of its expandable assets."""
    low = np.full(len(carriers), np.nan)
    high = np.full(len(carriers), np.nan)
    for c in formulated(n, "carrier_expansion"):
        name = renamer(c)
        built = (
            sources[name("COMPONENT_NOM_extendable")]
            & sources[name("COMPONENT_active_asset")]
            & (sources[name("COMPONENT_first_active")] != 0).any("period")
        )
        others = [d for d in built.dims if d != c._unit_dim]
        assets = built.indexes[c._unit_dim][built.any(others).values]
        carrier = sources[name("COMPONENT_carrier")].set_index(c._unit_dim)["carrier"]
        factors = np.broadcast_to(np.asarray(c._scaling(assets)), len(assets))
        at = carriers.get_indexer(carrier.reindex(assets).to_numpy())
        found = at != -1
        np.fmin.at(low, at[found], factors[found])
        np.fmax.at(high, at[found], factors[found])
    return xr.DataArray(np.sqrt(low * high), coords={"Carrier": carriers}).fillna(1.0)


def _cycle_factors(sources: NetworkSources, con: Constraint) -> xr.DataArray:
    """Return per row of Kirchhoff's voltage law the largest factor of the branches in its cycle."""
    grid = cycle_scaling(sources).reindex(cycle=con.labels.indexes["cycle"])
    period = sources.periods.get_indexer(sources.period_of)
    rows = xr.DataArray(grid.values[period], dims=("snapshot", "cycle"))
    rows = rows.broadcast_like(con.labels).transpose(*con.labels.dims)
    return con.labels.copy(data=rows.values).fillna(1.0)


def _scale_rows(
    m: Model,
    n: Network,
    sources: NetworkSources,
    rules: list[tuple[re.Pattern, str]],
    objective: float,
) -> None:
    """Give every constraint the factor its scaling rule takes from the network data.

    `bus` takes `Bus.scaling` of each unit's bus, `objective` the objective
    factor, `global` `GlobalConstraint.scaling`, `cycle` per row the largest
    factor of its cycle's branches, `carrier` per carrier the geometric mean
    of the smallest and largest factor of its expandable assets, and
    `none` keeps `1`.
    """
    for name, con in m.constraints.items():
        match _rule(name, con, n, rules):
            case "bus":
                component = name.partition("-")[0]
                con.scaling = n.c[component]._scaling(con.labels.indexes["name"])
            case "objective":
                con.scaling = objective
            case "global":
                con.scaling = global_constraint_factors(n, con.labels.indexes["name"])
            case "cycle":
                con.scaling = _cycle_factors(sources, con)
            case "carrier":
                con.scaling = _carrier_factors(
                    n, sources, con.labels.indexes["Carrier"]
                )


def _unscaled(n: Network, objective: float) -> bool:
    """Return whether the objective factor, `Bus.scaling` and `GlobalConstraint.scaling` are all `1`."""
    buses = invariant(n.c.buses.static.scaling)
    limits = invariant(n.c.global_constraints.static.scaling)
    return objective == 1 and bool((buses == 1).all() and (limits == 1).all())


def _scale(
    m: Model,
    n: Network,
    sources: NetworkSources,
    rules: list[tuple[re.Pattern, str]],
    objective: float,
) -> None:
    """Apply PyPSA's numerical scaling to the built model.

    A network whose factors are all `1` keeps linopy's default factors.
    """
    if _unscaled(n, objective):
        return
    m.objective.scaling = objective
    if KVL in m.constraints:
        check_sub_networks(n)
    _scale_columns(m, n, objective)
    _scale_rows(m, n, sources, rules, objective)


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
    mode = (options.get("transmission_losses") or {}).get("mode")
    rename = _renamer(s, spec_names(linearized_unit_commitment), mode)
    m = Model.from_spec(s, sources, expressions=False, rename=rename, **model_kwargs)
    _split_ports(m, sources)
    _relabel(
        m,
        n,
        snapshots=model_index,
        network_index=network_index,
        sources=sources,
        meshed=_meshed(n, meshed_thresholds),
    )
    m.constraints.sanitize_zeros()
    rules = _rules(spec_scalings(linearized_unit_commitment), rename)
    _scale(m, n, sources, rules, objective_scaling)
    return m
