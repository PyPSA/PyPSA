# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Data prep of the spec: passive network.

The cycle basis of the KVL rows, the cuts of the loss curves, the outage
factors of a security-constrained run and the flow-based domain.
"""

from __future__ import annotations

from math import sqrt
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd
import xarray as xr

from pypsa.optimization.flow_based import _build_domain
from pypsa.optimization.spec.sources import NetworkSources, source

if TYPE_CHECKING:
    from pypsa.components import Components
    from pypsa.optimization.flow_based import _FlowBasedDomain

DEG2RAD = np.pi / 180
NO_DOMAIN = "flow_based"


def _investment_period(s: NetworkSources, period: int) -> int | None:
    return period if s.multi_invest else None


def _over(s: NetworkSources, c: Components, frames: dict, out: str) -> xr.DataArray:
    """Per-period `(type, name) x label` frames as a `(period, unit, out)` grid, NaN off them."""
    dim = c._unit_dim
    units = s.names(c).to_numpy(dtype=object)
    labels = s.coords(out).to_numpy(dtype=object)
    grid = np.full((len(s.periods), len(units), len(labels)), np.nan)
    for i, frame in enumerate(frames.values()):
        if frame.empty or c.name not in frame.index.unique(0):
            continue
        rows = frame.xs(c.name, level=0)
        grid[i] = rows.reindex(index=units, columns=labels).to_numpy(dtype=float)
    return xr.DataArray(
        grid,
        coords={"period": s.periods, dim: units, out: labels},
        dims=("period", dim, out),
    )


# -- Kirchhoff's voltage law ------------------------------------------------


def _cycle_matrices(s: NetworkSources) -> dict[int, tuple[pd.DataFrame, pd.DataFrame]]:
    """Each period's cycle matrix, plain and weighted, NaN off its cycles."""
    matrices = {}
    for period in s.periods:
        pair = []
        for weighted in (False, True):
            C = s.n.cycle_matrix(
                investment_period=_investment_period(s, period),
                apply_weights=weighted,
                cycle_basis_method=s.cycle_basis_method,
            ).rename(columns=str)
            pair.append(C.where(C != 0))
        matrices[period] = tuple(pair)
    return matrices


def _cycles(s: NetworkSources, weighted: bool, c: Components) -> xr.DataArray:
    matrices = s.memo("cycles", lambda: _cycle_matrices(s))
    return _over(s, c, {p: pair[weighted] for p, pair in matrices.items()}, "cycle")


@source("cycle")
def _cycle_labels(s: NetworkSources, name: str) -> pd.Index:
    matrices = s.memo("cycles", lambda: _cycle_matrices(s)).values()
    width = max((plain.shape[1] for plain, _ in matrices), default=0)
    return pd.Index([str(k) for k in range(width)], dtype=object)


@source("Cycle_in_basis")
def _cycle_in_basis(s: NetworkSources, name: str) -> xr.DataArray:
    labels = s.coords("cycle")
    reached = [
        labels.isin(plain.columns[plain.notna().any()])
        for plain, _ in s.memo("cycles", lambda: _cycle_matrices(s)).values()
    ]
    grid = np.array(reached, dtype=bool).reshape(len(s.periods), len(labels))
    coords = {"period": s.periods, "cycle": labels.to_numpy(dtype=object)}
    return xr.DataArray(grid, coords=coords, dims=("period", "cycle"))


@source("${C}_cycle_weight")
def _cycle_weight(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    return _cycles(s, True, c)


def _varying(s: NetworkSources) -> xr.DataArray:
    """Whether a transformer's phase shift is a decision, per scenario."""
    c = s.n.components.transformers
    return s.da(c, "phase_shift_min") < s.da(c, "phase_shift_max")


@source("Transformer_phase_shift_varying")
def _phase_shift_varying(s: NetworkSources, name: str) -> xr.DataArray:
    return _varying(s)


@source("Transformer_phase_shift_fixed")
def _phase_shift_fixed(s: NetworkSources, name: str) -> xr.DataArray:
    return s.da(s.n.components.transformers, "phase_shift").astype(float)


@source("Transformer_phase_shift_cycle_weight")
def _phase_shift_cycle_weight(s: NetworkSources, name: str) -> xr.DataArray:
    sign = _cycles(s, False, s.n.components.transformers).where(_varying(s))
    return (sign * DEG2RAD).fillna(0.0)


@source("Transformer_phase_shift_weight")
def _phase_shift_weight(s: NetworkSources, name: str) -> xr.DataArray:
    c = s.n.components.transformers
    sign = _cycles(s, False, c).where(~_varying(s))
    by_snapshot = sign.sel(period=s.over_snapshots(s.period_of)).drop_vars("period")
    weight = by_snapshot * s.da(c, "phase_shift") * DEG2RAD
    return weight.where(weight != 0)


# -- transmission losses ----------------------------------------------------


def _loss_inputs(s: NetworkSources, c: Components) -> tuple[xr.DataArray, ...]:
    """`r_pu_eff` and the rating the loss curve spans, `s_max_pu * s_nom_max`."""
    extendable = s.da(c, "s_nom_extendable").astype(bool)
    s_nom_max = s.da(c, "s_nom_max").where(extendable, s.da(c, "s_nom"))
    if not np.isfinite(s_nom_max).all():
        msg = (
            f"Loss approximation requires finite 's_nom_max' for extendable "
            f"branches:\n {s_nom_max.where(~np.isfinite(s_nom_max), drop=True)}"
        )
        raise ValueError(msg)
    return s.da(c, "r_pu_eff"), s.da(c, "s_max_pu") * s_nom_max


def _secant_options(s: NetworkSources) -> tuple[float, float, int]:
    config = s.transmission_losses or {}
    atol = config.get("atol", 1)
    rtol = config.get("rtol", 0.1)
    max_segments = config.get("max_segments", 20)
    if atol <= 0:
        msg = f"'atol' must be positive, got {atol}"
        raise ValueError(msg)
    if rtol < 0:
        msg = f"'rtol' must be non-negative, got {rtol}"
        raise ValueError(msg)
    if max_segments < 1:
        msg = f"'max_segments' must be >= 1, got {max_segments}"
        raise ValueError(msg)
    return atol, rtol, max_segments


def _secant_factors(s: NetworkSources, c: Components) -> np.ndarray:
    """PyPSA's secant breakpoints of a component, as multiples of the first."""
    atol, rtol, max_segments = _secant_options(s)
    r, rating = _loss_inputs(s, c)
    lossy = r > 0
    first = 2 * np.sqrt(atol / r.where(lossy))
    target = (rating.max("snapshot") / first).where(lossy, 0)
    factors = [0.0, 1.0]
    while (factors[-1] < target).any():
        k = len(factors)
        step = max(k / (k - 1), 1 + 2 * (rtol + sqrt(rtol + rtol**2)))
        factors.append(factors[-1] * step)
        if k >= max_segments:
            msg = (
                "Secant loop hit max_segments; check atol/rtol or line parameters; "
                f"current inputs would result in {2 * max_segments} additional "
                "constraints per line"
            )
            raise RuntimeError(msg)
    return np.asarray(factors)


def _tangent_count(s: NetworkSources) -> int:
    segments = (s.transmission_losses or {}).get("segments")
    if not isinstance(segments, int) or segments < 1:
        msg = f"'segments' must be a positive integer, got {segments!r}"
        raise ValueError(msg)
    return segments


@source("segment")
def _segment_labels(s: NetworkSources, name: str) -> pd.Index:
    if s.transmission_losses is None:
        return pd.Index([], dtype=int)
    if (s.transmission_losses or {}).get("mode") == "tangents":
        return pd.Index(range(1, _tangent_count(s) + 1), dtype=int)
    lossy = [c for c in s.units.values() if f"{c.name}_loss_max" in s]
    counts = [len(_secant_factors(s, c)) - 1 for c in lossy if len(s.names(c))]
    return pd.Index(range(max(counts, default=0)), dtype=int)


@source("${C}_loss_max")
def _loss_max(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    if s.transmission_losses is None:
        return s.filled(name, 0.0)
    r, rating = _loss_inputs(s, c)
    return r * rating**2


def _cuts(s: NetworkSources, c: Components) -> tuple[xr.DataArray, xr.DataArray]:
    """Slope and offset of each cut to a component's loss curve, over `segment`."""
    segment = xr.DataArray(s.coords("segment"), dims="segment")
    segment = segment.assign_coords(segment=s.coords("segment"))
    r, rating = _loss_inputs(s, c)
    if (s.transmission_losses or {}).get("mode") == "tangents":
        flow = segment / len(segment) * rating
        slope = 2 * r * flow
        return slope, r * flow**2 - slope * flow
    atol = _secant_options(s)[0]
    factors = _secant_factors(s, c)
    pad = (0, len(segment) - len(factors) + 1)
    a = segment.copy(data=np.pad(factors[:-1], pad))
    b = segment.copy(data=np.pad(factors[1:], pad))
    lossy = r > 0
    slope = 2 * (atol * r.where(lossy, 0)) ** 0.5 * (a + b)
    offset = (-4 * atol * a * b).where(lossy, 0.0)
    return slope, offset


@source("${C}_loss_slope", "${C}_loss_offset")
def _loss_cut(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    if not len(s.coords("segment")):
        return s.filled(name, 0.0)
    slope, offset = s.memo(f"{c.name} cuts", lambda: _cuts(s, c))
    return slope if name.endswith("slope") else offset


# -- security-constrained runs ----------------------------------------------


def _outages(s: NetworkSources) -> pd.MultiIndex:
    """Read the outaged branches as `(component, name)`; a plain list names lines."""
    outages = s.branch_outages
    if outages is None:
        return pd.MultiIndex.from_tuples([], names=["type", "name"])
    if not isinstance(outages, pd.MultiIndex):
        outages = pd.MultiIndex.from_product([("Line",), outages])
    active = [name in s.names(s.n.components[c]) for c, name in outages]
    outages = outages[active]
    clash = outages.get_level_values(1)[outages.get_level_values(1).duplicated()]
    if not clash.empty:
        msg = f"Branch outages {clash.tolist()} name both a Line and a Transformer."
        raise ValueError(msg)
    return outages.set_names(["type", "name"])


@source("outage")
def _outage_labels(s: NetworkSources, name: str) -> pd.Index:
    return pd.Index(_outages(s).get_level_values(1), dtype=object)


@source("Outage_${c}")
def _outage_branch(s: NetworkSources, name: str, c: Components) -> pd.DataFrame:
    outages = _outages(s)
    names = outages.get_level_values(1)[outages.get_level_values(0) == c.name]
    labels = names.to_numpy(dtype=object)
    return pd.DataFrame({"outage": labels, c._unit_dim: labels}, dtype=object)


class SecurityGroup(NamedTuple):
    """The outages of one sub-network in one period, which PyPSA names apart."""

    suffix: str
    snapshots: np.ndarray
    branches: pd.MultiIndex
    outages: pd.MultiIndex


def security_groups(s: NetworkSources) -> list[SecurityGroup]:
    """Return the sub-network and period groups of the security rows, in PyPSA's order."""
    return s.memo("bodf", lambda: _bodf(s))[1]


def _bodf(s: NetworkSources) -> tuple[dict[int, pd.DataFrame], list[SecurityGroup]]:
    """Each period's outage factors, branches by outages, of its sub-networks, and the groups they form."""
    n = s.n
    outages = _outages(s)
    groups: list[SecurityGroup] = []
    if outages.empty:
        return {period: pd.DataFrame() for period in s.periods}, groups
    topology = n.get_scenario(n.scenarios[0]) if n.has_scenarios else n
    factors = {}
    for period in s.periods:
        topology.determine_network_topology(
            investment_period=_investment_period(s, period), skip_isolated_buses=True
        )
        frames = []
        for sub_network in topology.c.sub_networks.static.obj:
            branches = sub_network.branches_i()
            outaged = branches.intersection(outages)
            if outaged.empty:
                continue
            suffix = (
                f"{sub_network.name}-period-{period}"
                if s.multi_invest
                else sub_network.name
            )
            snapshots = np.flatnonzero(s.period_of == period)
            groups.append(SecurityGroup(suffix, snapshots, branches, outaged))
            sub_network.calculate_BODF()
            frame = pd.DataFrame(sub_network.BODF, index=branches, columns=branches)
            frames.append(frame[outaged].droplevel(0, axis=1))
        factors[period] = pd.concat(frames) if frames else pd.DataFrame()
    return factors, groups


@source("${C}_BODF")
def _branch_bodf(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    return _over(s, c, s.memo("bodf", lambda: _bodf(s))[0], "outage")


# -- flow-based market coupling ---------------------------------------------


def _domain(s: NetworkSources) -> _FlowBasedDomain | None:
    return s.memo("flow_based", lambda: _build_domain(s.n, s.network_index))


def _zone_domain(s: NetworkSources) -> pd.Series:
    """Map each zone bus to its flow-based domain."""
    domain = _domain(s)
    if domain is None:
        return pd.Series([], dtype=object)
    if domain.zone_domain is None:
        return pd.Series(NO_DOMAIN, index=pd.Index(domain.zones, dtype=object))
    return domain.zone_domain.astype(object)


def flow_based_zones(s: NetworkSources) -> pd.Index | None:
    """Return the zone buses of the flow-based domain in PyPSA's column order, None without one."""
    domain = _domain(s)
    return None if domain is None else pd.Index(domain.zones, dtype=object)


@source("flow_based_domain")
def _flow_based_domains(s: NetworkSources, name: str) -> pd.Index:
    return pd.Index(_zone_domain(s).unique(), dtype=object)


@source("Bus_flow_based_domain")
def _bus_flow_based_domain(s: NetworkSources, name: str) -> pd.DataFrame:
    zones = _zone_domain(s)
    return pd.DataFrame(
        {
            "bus": zones.index.to_numpy(dtype=object),
            "flow_based_domain": zones.to_numpy(),
        },
        dtype=object,
    )


@source("Bus_flow_based_zone")
def _bus_flow_based_zone(s: NetworkSources, name: str) -> xr.DataArray:
    buses = s.coords("bus")
    coords = {"bus": buses.to_numpy(dtype=object)}
    return xr.DataArray(buses.isin(_zone_domain(s).index), coords=coords, dims="bus")


def _flow_based_array(s: NetworkSources, attr: str) -> xr.DataArray:
    """Read a GlobalConstraint array in the spec's dims, NaN off flow-based rows."""
    c = s.n.components.global_constraints
    da = c.da[attr]
    if "snapshot" in da.dims:
        da = da.isel(snapshot=s.positions)
    da = da.drop_vars([v for v in da.coords if v not in da.dims or v == "snapshot"])
    if "snapshot" in da.dims:
        da = da.assign_coords(snapshot=s.snapshots)
    rows = c._flow_based_names()
    da = da.where(da["name"].isin(rows))
    da = da.rename(name="global_constraint")
    if "scenario" in da.dims and not s.n.has_scenarios:
        da = da.squeeze("scenario", drop=True)
    if "scenario" not in da.dims:
        da = da.expand_dims(scenario=s.scenarios)
    labels = s.coords("global_constraint").to_numpy(dtype=object)
    return da.reindex(global_constraint=labels)


@source("GlobalConstraint_ram")
def _ram(s: NetworkSources, name: str) -> xr.DataArray:
    if _domain(s) is None:
        return s.filled(name, np.nan)
    return _flow_based_array(s, "ram").astype(float)


@source("GlobalConstraint_zonal_ptdf", "GlobalConstraint_link_ptdf")
def _ptdf(s: NetworkSources, name: str) -> xr.DataArray:
    domain = _domain(s)
    if domain is None:
        return s.filled(name, np.nan)
    dim = "bus" if name.endswith("zonal_ptdf") else "link"
    columns = domain.zones if dim == "bus" else domain.links
    ptdf = _flow_based_array(s, "zonal_ptdf").sel(bus=list(columns))
    ptdf = ptdf.rename(bus=dim).reindex({dim: s.coords(dim).to_numpy(dtype=object)})
    return ptdf.astype(float)
