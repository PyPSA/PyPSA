# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Data prep of `pypsa.yaml`: passive network.

The cycle basis of the KVL rows, the cuts of the loss curves, the outage
factors of a security-constrained run and the flow-based domain.
"""

from __future__ import annotations

from math import sqrt
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

from pypsa.optimization.flow_based import _build_domain
from pypsa.optimization.spec.sources import (
    UNITS,
    NetworkSources,
    _filled,
    _static,
    dimension,
    single,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pypsa.optimization.flow_based import _FlowBasedDomain

PASSIVE = ("Line", "Transformer")
DEG2RAD = np.pi / 180
NO_DOMAIN = "flow_based"


def _memo(s: NetworkSources, key: str, derive: Callable[[NetworkSources], Any]) -> Any:
    """Compute `derive(s)` once per build."""
    store = s.__dict__.setdefault("_passive_network", {})
    if key not in store:
        store[key] = derive(s)
    return store[key]


def _investment_period(s: NetworkSources, period: int) -> int | None:
    return period if s.multi_invest else None


def _over(s: NetworkSources, component: str, frames: dict, out: str) -> xr.DataArray:
    """Per-period `(type, name) x label` frames as a `(period, unit, out)` grid, NaN off them."""
    dim = UNITS[component]
    units = s.coords(dim).to_numpy(dtype=object)
    labels = s.coords(out).to_numpy(dtype=object)
    grid = np.full((len(s.periods), len(units), len(labels)), np.nan)
    for i, frame in enumerate(frames.values()):
        if frame.empty or component not in frame.index.unique(0):
            continue
        rows = frame.xs(component, level=0)
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


def _cycles(s: NetworkSources, weighted: bool, component: str) -> xr.DataArray:
    frames = {
        p: pair[weighted] for p, pair in _memo(s, "cycles", _cycle_matrices).items()
    }
    return _over(s, component, frames, "cycle")


@dimension("cycle")
def _cycle_labels(s: NetworkSources) -> pd.Index:
    matrices = _memo(s, "cycles", _cycle_matrices).values()
    width = max((plain.shape[1] for plain, _ in matrices), default=0)
    return pd.Index([str(k) for k in range(width)], dtype=object)


@single("Line_cycle_weight", "Transformer_cycle_weight")
def _cycle_weight(s: NetworkSources, name: str) -> xr.DataArray:
    return s.frame(name, _cycles(s, True, name.partition("_")[0]))


def _varying(s: NetworkSources) -> xr.DataArray:
    static = _static(s, "Transformer")
    varying = static["phase_shift_min"] < static["phase_shift_max"]
    return xr.DataArray(
        varying.to_numpy(dtype=bool),
        coords={"transformer": static.index.to_numpy(dtype=object)},
        dims="transformer",
    )


def _phase_shift(s: NetworkSources) -> xr.DataArray:
    """Read the fixed phase shift in degrees of the first scenario."""
    return s.da("Transformer", "phase_shift").isel(scenario=0, drop=True)


@single("Transformer_phase_shift_varying")
def _phase_shift_varying(s: NetworkSources, name: str) -> xr.DataArray:
    return s.frame(name, _varying(s))


@single("Transformer_phase_shift_fixed")
def _phase_shift_fixed(s: NetworkSources, name: str) -> xr.DataArray:
    return s.frame(name, _phase_shift(s).astype(float))


@single("Transformer_phase_shift_cycle_weight")
def _phase_shift_cycle_weight(s: NetworkSources, name: str) -> xr.DataArray:
    sign = _cycles(s, False, "Transformer").where(_varying(s))
    return s.frame(name, (sign * DEG2RAD).fillna(0.0))


@single("Transformer_phase_shift_weight")
def _phase_shift_weight(s: NetworkSources, name: str) -> xr.DataArray:
    sign = _cycles(s, False, "Transformer").where(~_varying(s))
    period = xr.DataArray(
        s.period_of, coords={"snapshot": s.snapshots}, dims="snapshot"
    )
    by_snapshot = sign.sel(period=period).drop_vars("period")
    weight = by_snapshot * _phase_shift(s) * DEG2RAD
    return s.frame(name, weight.where(weight != 0))


@single("Line_x_pu_eff", "Transformer_x_pu_eff")
def _x_pu_eff(s: NetworkSources, name: str) -> xr.DataArray:
    s.n.calculate_dependent_values()
    component, _, attr = name.partition("_")
    return s.attribute(component, attr, name)


@single("Line_v_ang_max", "Transformer_v_ang_max")
def _v_ang_max(s: NetworkSources, name: str) -> xr.DataArray:
    component, _, attr = name.partition("_")
    limit = s.attribute(component, attr, name)
    return limit.where(np.isfinite(limit))


# -- transmission losses ----------------------------------------------------


def _loss_inputs(s: NetworkSources, component: str) -> tuple[xr.DataArray, ...]:
    """`r_pu_eff` and the rating the loss curve spans, `s_max_pu * s_nom_max`."""
    s.n.calculate_dependent_values()
    extendable = s.da(component, "s_nom_extendable").astype(bool)
    s_nom_max = s.da(component, "s_nom_max").where(extendable, s.da(component, "s_nom"))
    if not np.isfinite(s_nom_max).all():
        msg = (
            f"Loss approximation requires finite 's_nom_max' for extendable "
            f"branches:\n {s_nom_max.where(~np.isfinite(s_nom_max), drop=True)}"
        )
        raise ValueError(msg)
    return s.da(component, "r_pu_eff"), s.da(component, "s_max_pu") * s_nom_max


def _secant_options(s: NetworkSources) -> tuple[float, float, int]:
    config = s.transmission_losses
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


def _secant_factors(s: NetworkSources, component: str) -> np.ndarray:
    """PyPSA's secant breakpoints of a component, as multiples of the first."""
    atol, rtol, max_segments = _secant_options(s)
    r, rating = _loss_inputs(s, component)
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
    segments = s.transmission_losses.get("segments")
    if not isinstance(segments, int) or segments < 1:
        msg = f"'segments' must be a positive integer, got {segments!r}"
        raise ValueError(msg)
    return segments


@dimension("segment")
def _segment_labels(s: NetworkSources) -> pd.Index:
    if s.transmission_losses is None:
        return pd.Index([], dtype=int)
    if s.transmission_losses["mode"] == "tangents":
        return pd.Index(range(1, _tangent_count(s) + 1), dtype=int)
    counts = [len(_secant_factors(s, c)) - 1 for c in PASSIVE if len(s.names(c))]
    return pd.Index(range(max(counts, default=0)), dtype=int)


@single("Line_loss_max", "Transformer_loss_max")
def _loss_max(s: NetworkSources, name: str) -> xr.DataArray:
    if s.transmission_losses is None:
        return _filled(s, name, 0.0)
    r, rating = _loss_inputs(s, name.partition("_")[0])
    return s.frame(name, r * rating**2)


def _cuts(s: NetworkSources, component: str) -> tuple[xr.DataArray, xr.DataArray]:
    """Slope and offset of each cut to a component's loss curve, over `segment`."""
    segment = xr.DataArray(s.coords("segment"), dims="segment")
    segment = segment.assign_coords(segment=s.coords("segment"))
    r, rating = _loss_inputs(s, component)
    if s.transmission_losses["mode"] == "tangents":
        flow = segment / len(segment) * rating
        slope = 2 * r * flow
        return slope, r * flow**2 - slope * flow
    atol = _secant_options(s)[0]
    factors = _secant_factors(s, component)
    pad = (0, len(segment) - len(factors) + 1)
    a = segment.copy(data=np.pad(factors[:-1], pad))
    b = segment.copy(data=np.pad(factors[1:], pad))
    lossy = r > 0
    slope = 2 * np.sqrt(atol * r.where(lossy, 0)) * (a + b)
    offset = (-4 * atol * a * b).where(lossy, 0.0)
    return slope, offset


@single(
    "Line_loss_slope",
    "Line_loss_offset",
    "Transformer_loss_slope",
    "Transformer_loss_offset",
)
def _loss_cut(s: NetworkSources, name: str) -> xr.DataArray:
    if not len(s.coords("segment")):
        return _filled(s, name, 0.0)
    component = name.partition("_")[0]
    slope, offset = _memo(s, f"cuts_{component}", lambda s: _cuts(s, component))
    return s.frame(name, slope if name.endswith("slope") else offset)


# -- security-constrained runs ----------------------------------------------


def _outages(s: NetworkSources) -> pd.MultiIndex:
    """Read the outaged branches as `(component, name)`; a plain list names lines."""
    outages = s.branch_outages
    if outages is None:
        return pd.MultiIndex.from_tuples([], names=["type", "name"])
    if not isinstance(outages, pd.MultiIndex):
        outages = pd.MultiIndex.from_product([("Line",), outages])
    active = [name in s.names(c) for c, name in outages]
    outages = outages[active]
    clash = outages.get_level_values(1)[outages.get_level_values(1).duplicated()]
    if not clash.empty:
        msg = f"Branch outages {clash.tolist()} name both a Line and a Transformer."
        raise ValueError(msg)
    return outages.set_names(["type", "name"])


@dimension("outage")
def _outage_labels(s: NetworkSources) -> pd.Index:
    return pd.Index(_outages(s).get_level_values(1), dtype=object)


@single("Outage_line", "Outage_transformer")
def _outage_branch(s: NetworkSources, name: str) -> pd.DataFrame:
    column = name.removeprefix("Outage_")
    component = column.capitalize()
    outages = _outages(s)
    names = outages.get_level_values(1)[outages.get_level_values(0) == component]
    labels = names.to_numpy(dtype=object)
    return pd.DataFrame({"outage": labels, column: labels}, dtype=object)


def _bodf(s: NetworkSources) -> dict[int, pd.DataFrame]:
    """Each period's outage factors, branches by outages, of its sub-networks."""
    n = s.n
    topology = n.get_scenario(n.scenarios[0]) if n.has_scenarios else n
    outages = _outages(s)
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
            sub_network.calculate_BODF()
            frame = pd.DataFrame(sub_network.BODF, index=branches, columns=branches)
            frames.append(frame[outaged].droplevel(0, axis=1))
        factors[period] = pd.concat(frames) if frames else pd.DataFrame()
    return factors


@single("Line_BODF", "Transformer_BODF")
def _branch_bodf(s: NetworkSources, name: str) -> xr.DataArray:
    frames = _memo(s, "bodf", _bodf)
    return s.frame(name, _over(s, name.partition("_")[0], frames, "outage"))


# -- flow-based market coupling ---------------------------------------------


def _domain(s: NetworkSources) -> _FlowBasedDomain | None:
    return _build_domain(s.n, s.network_index)


def _zone_domain(s: NetworkSources) -> pd.Series:
    """Map each zone bus to its flow-based domain."""
    domain = _memo(s, "flow_based", _domain)
    if domain is None:
        return pd.Series([], dtype=object)
    if domain.zone_domain is None:
        return pd.Series(NO_DOMAIN, index=pd.Index(domain.zones, dtype=object))
    return domain.zone_domain.astype(object)


@dimension("flow_based_domain")
def _flow_based_domains(s: NetworkSources) -> pd.Index:
    return pd.Index(_zone_domain(s).unique(), dtype=object)


@single("Bus_flow_based_domain")
def _bus_flow_based_domain(s: NetworkSources, name: str) -> pd.DataFrame:
    zones = _zone_domain(s)
    return pd.DataFrame(
        {
            "bus": zones.index.to_numpy(dtype=object),
            "flow_based_domain": zones.to_numpy(),
        },
        dtype=object,
    )


@single("Bus_flow_based_zone")
def _bus_flow_based_zone(s: NetworkSources, name: str) -> xr.DataArray:
    buses = s.coords("bus")
    zone = xr.DataArray(
        buses.isin(_zone_domain(s).index),
        coords={"bus": buses.to_numpy(dtype=object)},
        dims="bus",
    )
    return s.frame(name, zone)


def _flow_based_array(s: NetworkSources, attr: str) -> xr.DataArray:
    """Read a GlobalConstraint array in the spec's dims, NaN off flow-based rows."""
    c = s.n.components.global_constraints
    da = c.da[attr]
    da = da.drop_vars([v for v in da.coords if v not in da.dims])
    rows = c._flow_based_names()
    da = da.where(da["name"].isin(rows))
    da = da.rename(name="global_constraint")
    if "snapshot" in da.dims:
        da = da.isel(snapshot=s.positions).assign_coords(snapshot=s.snapshots)
    if "scenario" in da.dims and not s.n.has_scenarios:
        da = da.squeeze("scenario", drop=True)
    if "scenario" not in da.dims:
        da = da.expand_dims(scenario=s.scenarios)
    labels = s.coords("global_constraint").to_numpy(dtype=object)
    return da.reindex(global_constraint=labels)


@single("GlobalConstraint_ram")
def _ram(s: NetworkSources, name: str) -> xr.DataArray:
    if _memo(s, "flow_based", _domain) is None:
        return _filled(s, name, np.nan)
    return s.frame(name, _flow_based_array(s, "ram").astype(float))


@single("GlobalConstraint_zonal_ptdf", "GlobalConstraint_link_ptdf")
def _ptdf(s: NetworkSources, name: str) -> xr.DataArray:
    domain = _memo(s, "flow_based", _domain)
    if domain is None:
        return _filled(s, name, np.nan)
    dim = "bus" if name.endswith("zonal_ptdf") else "link"
    columns = domain.zones if dim == "bus" else domain.links
    ptdf = _flow_based_array(s, "zonal_ptdf").sel(bus=list(columns))
    ptdf = ptdf.rename(bus=dim).reindex({dim: s.coords(dim).to_numpy(dtype=object)})
    return s.frame(name, ptdf.astype(float))
