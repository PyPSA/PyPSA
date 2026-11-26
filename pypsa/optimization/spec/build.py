# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Build PyPSA's linopy model from the spec and the network's data."""

from __future__ import annotations

import functools
import json
import re
from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr
from linopy import Model
from linopy.constants import BREAKPOINT_DIM
from linopy.constraints import Constraint
from linopy.expressions import LinearExpression
from linopy.variables import Variable

from pypsa.components.common import invariant
from pypsa.optimization.scaling import (
    KVL,
    check_sub_networks,
    global_constraint_factors,
)
from pypsa.optimization.spec._passive_network import cycle_scaling
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
def _unit_dims() -> tuple[str, ...]:
    """Return the spec's unit dims, one per component with a formulation, which every entry lists first."""
    import pypsa  # noqa: PLC0415

    return tuple(sorted(c._unit_dim for c in formulated(pypsa.Network())))


@functools.cache
def _rendered(linearized_unit_commitment: bool) -> tuple[Spec, Names, Scalings]:
    """Return the spec, its PyPSA names and its scaling rules, rendered once per process."""
    return render(SPEC, _patch(linearized_unit_commitment))


#: A layer over the spec: a YAML file, YAML text or a mapping, written as a `mathspec.override` patch.
Override = str | Path | Mapping[str, Any]


def _layer(override: Override) -> dict[str, Any]:
    """Return *override* as the mapping it states.

    Raises:
        FileNotFoundError: A `str` with no newline names no file.
        TypeError: *override* is not a path, YAML text or mapping.

    """
    import yaml  # noqa: PLC0415

    if isinstance(override, Mapping):
        return dict(override)
    if isinstance(override, str) and "\n" in override:
        return yaml.safe_load(override)
    if isinstance(override, str | Path):
        return yaml.safe_load(Path(override).read_text())
    msg = f"An override is a YAML path, YAML text or a mapping, not {type(override).__name__}."
    raise TypeError(msg)


def _expanded(s: Spec) -> tuple[Spec, Spec]:
    return s, s.expand("piecewise")


@functools.cache
def _base(linearized_unit_commitment: bool) -> tuple[Spec, Spec]:
    return _expanded(_rendered(linearized_unit_commitment)[0])


#: The most patched formulations kept in memory; the oldest go first.
KEEP = 8
_PATCHED: dict[str, tuple[Spec, Spec]] = {}


def _formulation(
    linearized_unit_commitment: bool = False,
    piecewise_options: Sequence[PiecewiseOptions] = (),
    overrides: Sequence[Override] = (),
) -> tuple[Spec, Spec]:
    """Return the spec of `spec()` and its piecewise curves written out, which linopy builds from."""
    import mathspec  # noqa: PLC0415

    base = _base(linearized_unit_commitment)
    piecewise = (
        [_piecewise_patch(base[0], piecewise_options)] if piecewise_options else []
    )
    patches = [*piecewise, *map(_layer, overrides)]
    if not patches:
        return base
    key = json.dumps(
        [linearized_unit_commitment, patches], sort_keys=True, default=repr
    )
    if key not in _PATCHED:
        _PATCHED[key] = _expanded(mathspec.override(base[0], patches))
        while len(_PATCHED) > KEEP:
            del _PATCHED[next(iter(_PATCHED))]
    return _PATCHED[key]


def spec(
    linearized_unit_commitment: bool = False,
    piecewise_options: Sequence[PiecewiseOptions] = (),
    overrides: Sequence[Override] = (),
) -> Spec:
    """Return the spec of `n.optimize()` for a network without composites, patched for linearised unit commitment and piecewise options on request, with *overrides* laid over it in turn."""
    return _formulation(linearized_unit_commitment, piecewise_options, overrides)[0]


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

    for name, expr in list(m.expressions.items()):
        dims = tuple(expr.coord_dims)
        m.remove_expressions(name)
        m.add_expressions(frame(expr, dims, _pypsa_dims(dims, False)), name=name)

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
#: The rows of a piecewise curve that hold only its weights and binaries.
WEIGHT_ROWS = {"active_bound", "binary_order", "delta_bound", "fill_order"}
#: The rows of a cost curve that hold the cost: its chords and its `cost` link.
COST_ROWS = {"chord", "cost"}


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


def _rules(scalings: Scalings, rename: Callable[[str], str]) -> dict[str, str]:
    """Return the scaling rule of each entry by its PyPSA name."""
    return {rename(key): rule for key, rule in scalings.items()}


def _rule(name: str, con: Constraint, n: Network, rules: dict[str, str]) -> str:
    """Return the scaling rule of the constraint `name`: its `scaling:`, else one by its kind.

    A row of a piecewise curve that holds only its weights and binaries
    takes `none`, one of a cost curve that holds the cost `objective`. A
    row over the units of a component on a bus takes `bus`, any other
    `none`.
    """
    if name in rules:
        return rules[name]
    curve, piecewise, kind = name.rpartition("_piecewise_")
    if piecewise and kind in WEIGHT_ROWS:
        return "none"
    cost = kind.removesuffix("_ungated") in COST_ROWS
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
    for c in formulated(n, "capacity"):
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
    rules: dict[str, str],
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
    rules: dict[str, str],
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
    piecewise_options: Sequence[Any] = (),
    overrides: Sequence[Override] = (),
    objective_scaling: float = 1.0,
    expressions: bool = False,
    **options: Any,
) -> Model:
    """Build the model of `n` over the build's snapshots from the spec, with its composites and then *overrides* laid over it."""
    check(
        n,
        network_index,
        linearized_unit_commitment=linearized_unit_commitment,
        piecewise_options=piecewise_options,
        **options,
    )
    model_kwargs = model_kwargs or {}
    n.calculate_dependent_values()
    overrides = [*n.composites.patches(), *overrides]
    s, expanded = _formulation(linearized_unit_commitment, piecewise_options, overrides)
    sources = NetworkSources(
        n,
        s,
        network_index,
        model_index,
        providers=n.composites.providers(),
        **options,
    )
    mode = (options.get("transmission_losses") or {}).get("mode")
    rename = _renamer(s, spec_names(linearized_unit_commitment), mode)
    m = Model.from_spec(
        expanded,
        sources,
        expressions=expressions,
        rename=rename,
        dim_order=_unit_dims(),
        **model_kwargs,
    )
    _relabel(m, snapshots=model_index, network_index=network_index)
    rules = _rules(spec_scalings(linearized_unit_commitment), rename)
    _scale(m, n, sources, rules, objective_scaling)
    _split_by_label(m)
    return m
