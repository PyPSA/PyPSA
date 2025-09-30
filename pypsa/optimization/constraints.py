"""Define optimisation constraints from PyPSA networks with Linopy."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import linopy
import numpy as np
import pandas as pd
import xarray as xr
from linopy import merge
from numpy import inf, isfinite
from xarray import DataArray, concat

from pypsa._options import options
from pypsa.common import as_index, expand_series
from pypsa.components.common import as_components
from pypsa.descriptors import nominal_attrs
from pypsa.optimization.common import reindex

if TYPE_CHECKING:
    from collections.abc import Sequence

    from xarray import DataArray  # noqa: TC004

    from pypsa import Network

    ArgItem = list[str | int | float | DataArray]

logger = logging.getLogger(__name__)

# TODO move to constants.py
lookup = pd.read_csv(
    Path(__file__).parent / ".." / "data" / "variables.csv",
    index_col=["component", "variable"],
)


def _infer_big_m_scale(n: Network, component: str) -> float:
    """Infer a reasonable big-M scale from network data."""
    candidates: list[float] = []

    # Peak total load over time provides a natural system-scale bound
    load = n.get_switchable_as_dense("Load", "p_set")
    peak_load = load.sum(axis=1).abs().max()
    candidates.append(peak_load)

    # Use existing nominal values as additional hints
    c = as_components(n, component)
    if not c.static.empty:
        nom_attr = c._operational_attrs["nom"]
        nom_series = c.static[nom_attr]
        finite_nominal = nom_series[np.isfinite(nom_series) & (nom_series > 0)]
        if not finite_nominal.empty:
            candidates.append(float(finite_nominal.max()))

        nom_max_series = c.static[f"{nom_attr}_max"]
        finite_max = nom_max_series[np.isfinite(nom_max_series) & (nom_max_series > 0)]
        if not finite_max.empty:
            candidates.append(finite_max.max())

    if not candidates:
        return 1e6

    fallback = max(candidates) * 10
    if not np.isfinite(fallback) or fallback <= 0:
        return 1e6
    return fallback


def define_operational_constraints_for_non_extendables(
    n: Network, sns: pd.Index, component: str, attr: str, transmission_losses: int
) -> None:
    """Define operational constraints (lower-/upper bound).

    Sets operational constraints for a subset of non-extendable
    and non-committable components based on their bounds. For each component,
    the constraint enforces:

    lower_bound ≤ dispatch ≤ upper_bound

    where lower_bound and upper_bound are computed from the component's nominal
    capacity and min/max per unit values.

    Applies to Components
    ---------------------
    Generator (p), Line (s), Transformer (s), Link (p), Store (e), StorageUnit (p_dispatch, p_store, state_of_charge)

    Parameters
    ----------
    n : pypsa.Network
        Network instance containing the model and component data
    sns : pd.Index
        Set of snapshots for which to define the constraints
    component : str
        Name of the network component (e.g. "Generator", "Link")
    attr : str
        Name of the attribute to constrain (e.g. "p" for active power)
    transmission_losses : int
        Number of segments for transmission loss linearization; if non-zero,
        losses are considered in the constraints for passive branches

    Returns
    -------
    None

    Notes
    -----
    For passive branches with transmission losses, the constraint accounts for
    the losses in both directions, see justification in [1]_.

    References
    ----------
    .. [1] F. Neumann, T. Brown, "Transmission losses in power system
       optimization models: A comparison of heuristic and exact solution methods,"
       Applied Energy, 2022, https://doi.org/10.1016/j.apenergy.2022.118859

    """
    c = as_components(n, component)
    fix_i = c.fixed.difference(c.committables).difference(c.inactive_assets)

    if fix_i.empty:
        return

    nominal_fix = c.da[c._operational_attrs["nom"]].sel(name=fix_i)
    min_pu, max_pu = c.get_bounds_pu(attr=attr)
    max_pu = max_pu.sel(name=fix_i)
    min_pu = min_pu.sel(name=fix_i)
    if "snapshot" in min_pu.dims:
        min_pu = min_pu.sel(snapshot=sns)
        max_pu = max_pu.sel(snapshot=sns)

    lower = min_pu * nominal_fix
    upper = max_pu * nominal_fix

    active = c.da.active.sel(name=fix_i, snapshot=sns)

    dispatch = n.model[f"{c.name}-{attr}"].sel(name=fix_i)

    if c.name in n.passive_branch_components and transmission_losses:
        loss = n.model[f"{c.name}-loss"].sel(name=fix_i)
        lhs_lower = dispatch - loss
        lhs_upper = dispatch + loss
    else:
        lhs_lower = lhs_upper = dispatch

    n.model.add_constraints(
        lhs_lower, ">=", lower, name=f"{c.name}-fix-{attr}-lower", mask=active
    )
    n.model.add_constraints(
        lhs_upper, "<=", upper, name=f"{c.name}-fix-{attr}-upper", mask=active
    )


def define_operational_constraints_for_extendables(
    n: Network, sns: pd.Index, component: str, attr: str, transmission_losses: int
) -> None:
    """Define operational constraints (lower-/upper bound) for extendable components.

    Sets operational constraints for extendable components based on their bounds.
    For each component, the constraint enforces:

    lower_bound ≤ dispatch ≤ upper_bound

    where lower_bound and upper_bound are computed from the component's nominal
    capacity and min/max per unit values.

    Applies to Components
    ---------------------
    Generator (p), Line (s), Transformer (s), Link (p), Store (e), StorageUnit (p_dispatch, p_store, state_of_charge)

    Parameters
    ----------
    n : pypsa.Network
        Network instance containing the model and component data
    sns : pd.Index
        Set of snapshots for which to define the constraints
    component : str
        Name of the network component (e.g. "Generator", "Link")
    attr : str
        Name of the attribute to constrain (e.g. "p" for active power)
    transmission_losses : int
        Number of segments for transmission loss linearization; if non-zero,
        losses are considered in the constraints for passive branches

    Returns
    -------
    None

    """
    c = as_components(n, component)
    sns = as_index(n, sns, "snapshots")

    ext_i = c.extendables.difference(c.inactive_assets)
    com_ext_i = c.committables.intersection(ext_i)
    ext_i = ext_i.difference(com_ext_i)

    if ext_i.empty:
        return
    if isinstance(ext_i, pd.MultiIndex):
        ext_i = ext_i.unique(level="name")

    min_pu, max_pu = c.get_bounds_pu(attr=attr)
    min_pu = min_pu.sel(name=ext_i)
    max_pu = max_pu.sel(name=ext_i)
    if "snapshot" in min_pu.dims:
        min_pu = min_pu.sel(snapshot=sns)
        max_pu = max_pu.sel(snapshot=sns)

    dispatch = n.model[f"{c.name}-{attr}"].sel(name=ext_i)
    capacity = n.model[f"{c.name}-{nominal_attrs[c.name]}"].sel(name=ext_i)
    active = c.da.active.sel(name=ext_i, snapshot=sns)

    lhs_lower = dispatch - min_pu * capacity
    lhs_upper = dispatch - max_pu * capacity

    if c.name in n.passive_branch_components and transmission_losses:
        loss = n.model[f"{c.name}-loss"].sel(name=ext_i)
        lhs_lower = lhs_lower - loss
        lhs_upper = lhs_upper + loss

    n.model.add_constraints(
        lhs_lower, ">=", 0, name=f"{c.name}-ext-{attr}-lower", mask=active
    )
    n.model.add_constraints(
        lhs_upper, "<=", 0, name=f"{c.name}-ext-{attr}-upper", mask=active
    )


def define_operational_constraints_for_committables(
    n: Network, sns: pd.Index, component: str
) -> None:
    r"""Define operational constraints for committable components.

    Sets operational constraints for components with unit commitment
    decisions. Supports both fixed-capacity and extendable committable
    components using a big-M formulation for the latter.

    The constraints include:

    1. Power output limits based on commitment status
    2. State transition constraints (start-up/shut-down)
    3. Minimum up and down time constraints
    4. Ramp rate constraints for committed units

    For committable-only components (fixed capacity):
    .. math::
        p_{i,t}^{min} u_{i,t} \\leq p_{i,t} \\leq p_{i,t}^{max} u_{i,t}

    For committable+extendable components (big-M formulation):
    .. math::
        p_{i,t} \\geq p_{i,t}^{min,pu} \\cdot p_{i}^{nom} - M \\cdot (1 - u_{i,t})

    .. math::
        p_{i,t} \\leq M \\cdot u_{i,t}

    .. math::
        p_{i,t} \\leq p_{i,t}^{max,pu} \\cdot p_{i}^{nom}

    where :math:`M` is a sufficiently large constant (big-M), :math:`u_{i,t}`
    is the binary commitment status, and :math:`p_{i}^{nom}` is the optimized
    capacity variable.

    Applies to Components
    ---------------------
    Generator, Link (when they have unit commitment capabilities)

    Parameters
    ----------
    n : pypsa.Network
        Network instance containing the model and component data
    sns : pd.Index
        Set of snapshots for which to define the constraints
    component : str
        Name of the network component ("Generator" or "Link")

    Returns
    -------
    None

    Notes
    -----
    The linearized approximation of the unit commitment problem
    is possible with flag `n._linearized_uc`. Here linearization
    implies that p_min_pu is fractional, ie component can start up
    any fraction of its capacity. The linearization is based on
    [2]_.

    For components with equal start-up and shut-down costs, additional
    tightening constraints are applied to improve the linear relaxation.

    References
    ----------
    .. [2] Y. Hua, C. Liu, J. Zhang, "Representing Operational
       Flexibility in Generation Expansion Planning Through Convex Relaxation
       of Unit Commitment," IEEE Transactions on Power Systems, vol. 32,
       no. 5, pp. 3854-3865, 2017, https://doi.org/10.1109/TPWRS.2017.2735026

    """
    c = as_components(n, component)
    com_i: pd.Index = c.committables.difference(c.inactive_assets)

    if com_i.empty:
        return

    status = n.model[f"{c.name}-status"]
    start_up = n.model[f"{c.name}-start_up"]
    shut_down = n.model[f"{c.name}-shut_down"]
    status_diff = status - status.shift(snapshot=1)
    p = n.model[f"{c.name}-p"].sel(name=com_i)
    active = c.da.active.sel(name=com_i, snapshot=sns)

    ext_i: pd.Index = c.extendables.difference(c.inactive_assets)
    com_ext_i: pd.Index = com_i.intersection(ext_i)
    com_fix_i: pd.Index = com_i.difference(ext_i)

    # parameters
    nominal = c.da[c._operational_attrs["nom"]].sel(name=com_i)
    min_pu, max_pu = c.get_bounds_pu(attr="p")
    min_pu = min_pu.sel(name=com_i, snapshot=sns)
    max_pu = max_pu.sel(name=com_i, snapshot=sns)

    lower_p = min_pu * nominal
    upper_p = max_pu * nominal
    min_up_time_set = c.da.min_up_time.sel(name=com_i)
    min_down_time_set = c.da.min_down_time.sel(name=com_i)

    ramp_up_limit = nominal * c.da.ramp_limit_up.sel(name=com_i).fillna(1)
    ramp_down_limit = nominal * c.da.ramp_limit_down.sel(name=com_i).fillna(1)
    ramp_start_up = nominal * c.da.ramp_limit_start_up.sel(name=com_i)
    ramp_shut_down = nominal * c.da.ramp_limit_shut_down.sel(name=com_i)
    up_time_before_set = c.da.up_time_before.sel(name=com_i)
    down_time_before_set = c.da.down_time_before.sel(name=com_i)
    initially_up = up_time_before_set.astype(bool)
    initially_down = down_time_before_set.astype(bool)

    # check if there are status calculated/fixed before given sns interval
    if sns[0] != n.snapshots[0]:
        start_i = n.snapshots.get_loc(sns[0])
        # get generators which are online until the first regarded snapshot
        until_start_up = c._as_dynamic(
            "status", n.snapshots[:start_i][::-1], inds=com_i
        )
        ref = range(1, len(until_start_up) + 1)
        up_time_before = DataArray(
            until_start_up[until_start_up.cumsum().eq(ref, axis=0)].sum()
        )
        up_time_before_set = up_time_before.clip(max=min_up_time_set)
        initially_up = up_time_before_set.astype(bool)
        # get number of snapshots for generators which are offline before the first regarded snapshot
        until_start_down = ~until_start_up.astype(bool)
        ref = range(1, len(until_start_down) + 1)
        down_time_before = DataArray(
            until_start_down[until_start_down.cumsum().eq(ref, axis=0)].sum()
        )
        down_time_before_set = down_time_before.clip(max=min_down_time_set)
        initially_down = down_time_before_set.astype(bool)

    if not com_ext_i.empty:
        p_nom_var = n.model[f"{c.name}-{c._operational_attrs['nom']}"]

        p_nom_max_vals = c.da.p_nom_max.sel(name=com_ext_i)
        max_pu_vals = max_pu.sel(name=com_ext_i).max("snapshot")

        big_m_default = options.params.optimize.committable_big_m
        if (
            big_m_default is None
            or not np.isfinite(big_m_default)
            or big_m_default <= 0
        ):
            big_m_default = _infer_big_m_scale(n, component)

        fallback_values = big_m_default * max_pu_vals.fillna(1)
        M_values = xr.where(
            isfinite(p_nom_max_vals) & (p_nom_max_vals > 0),
            p_nom_max_vals * max_pu_vals,
            fallback_values,
        )
        p_ext = p.sel(name=com_ext_i)
        status_ext = status.sel(name=com_ext_i)
        p_nom_ext = p_nom_var.sel(name=com_ext_i)
        min_pu_ext = min_pu.sel(name=com_ext_i)
        max_pu_ext = max_pu.sel(name=com_ext_i)

        active_ext = active.sel(name=com_ext_i)
        lhs_lower = (1, p_ext), (-min_pu_ext, p_nom_ext), (-M_values, status_ext)
        n.model.add_constraints(
            lhs_lower,
            ">=",
            -M_values,
            name=f"{c.name}-com-ext-p-lower",
            mask=active_ext,
        )

        lhs_upper = (1, p_ext), (-M_values, status_ext)
        n.model.add_constraints(
            lhs_upper,
            "<=",
            0,
            name=f"{c.name}-com-ext-p-upper-bigM",
            mask=active_ext,
        )

        lhs_upper_cap = (1, p_ext), (-max_pu_ext, p_nom_ext)
        n.model.add_constraints(
            lhs_upper_cap,
            "<=",
            0,
            name=f"{c.name}-com-ext-p-upper-cap",
            mask=active_ext,
        )

        dims_excl_name = [dim for dim in min_pu_ext.dims if dim != "name"]
        if dims_excl_name:
            nonneg_mask = (min_pu_ext >= 0).all(dim=dims_excl_name)
        else:
            nonneg_mask = min_pu_ext >= 0

        if nonneg_mask.any().item():
            nonneg_idx = nonneg_mask.to_series()
            nonneg_idx = nonneg_idx[nonneg_idx].index
            if len(nonneg_idx) > 0:
                p_nonneg = p_ext.sel(name=nonneg_idx)
                active_nonneg = active_ext.sel(name=nonneg_idx)
                lhs_nonneg = ((1, p_nonneg),)
                n.model.add_constraints(
                    lhs_nonneg,
                    ">=",
                    0,
                    name=f"{c.name}-com-ext-p-lower-nonneg",
                    mask=active_nonneg,
                )

    if not com_fix_i.empty:
        p_fix = p.sel(name=com_fix_i)
        status_fix = status.sel(name=com_fix_i)
        lower_p_fix = lower_p.sel(name=com_fix_i)
        upper_p_fix = upper_p.sel(name=com_fix_i)
        active_fix = active.sel(name=com_fix_i)

        lhs_lower_fix = (1, p_fix), (-lower_p_fix, status_fix)
        n.model.add_constraints(
            lhs_lower_fix,
            ">=",
            0,
            name=f"{c.name}-com-p-lower",
            mask=active_fix,
        )

        lhs_upper_fix = (1, p_fix), (-upper_p_fix, status_fix)
        n.model.add_constraints(
            lhs_upper_fix,
            "<=",
            0,
            name=f"{c.name}-com-p-upper",
            mask=active_fix,
        )

    # state-transition constraint
    rhs = pd.DataFrame(0, sns, com_i)
    # Convert xarray boolean to list of indices for DataFrame indexing
    initially_up_indices = com_i[initially_up.values]
    if not initially_up_indices.empty:
        rhs.loc[sns[0], initially_up_indices] = -1

    lhs_lower = start_up - status_diff
    n.model.add_constraints(
        lhs_lower, ">=", rhs, name=f"{c.name}-com-transition-start-up", mask=active
    )

    rhs = pd.DataFrame(0, sns, com_i)
    if not initially_up_indices.empty:
        rhs.loc[sns[0], initially_up_indices] = 1

    lhs_lower = shut_down + status_diff
    n.model.add_constraints(
        lhs_lower, ">=", rhs, name=f"{c.name}-com-transition-shut-down", mask=active
    )

    # min up time
    min_up_time_i = com_i[min_up_time_set.astype(bool)]
    if not min_up_time_i.empty:
        expr = []
        for g in min_up_time_i:
            su = start_up.loc[:, g]
            # Retrieve the minimum up time value for generator g and convert it to a scalar
            up_time_value = min_up_time_set.sel(name=g).item()
            expr.append(su.rolling(snapshot=up_time_value).sum())
        lhs_lower = -status.loc[:, min_up_time_i] + merge(expr, dim=com_i.name)
        lhs_lower = lhs_lower.sel(snapshot=sns[1:])
        n.model.add_constraints(
            lhs_lower,
            "<=",
            0,
            name=f"{c.name}-com-up-time",
            mask=active.loc[sns[1:], min_up_time_i],
        )

    # min down time
    min_down_time_i = com_i[min_down_time_set.astype(bool)]
    if not min_down_time_i.empty:
        expr = []
        for g in min_down_time_i:
            su = shut_down.loc[:, g]
            down_time_value = min_down_time_set.sel(
                {min_down_time_set.dims[0]: g}
            ).item()
            expr.append(su.rolling(snapshot=down_time_value).sum())
        lhs_lower = status.loc[:, min_down_time_i] + merge(expr, dim=com_i.name)
        lhs_lower = lhs_lower.sel(snapshot=sns[1:])
        n.model.add_constraints(
            lhs_lower,
            "<=",
            1,
            name=f"{c.name}-com-down-time",
            mask=active.loc[sns[1:], min_down_time_i],
        )
    # up time before
    timesteps = xr.DataArray(
        [range(1, len(sns) + 1)] * len(com_i),
        coords=[com_i, sns],
        dims=[com_i.name, "snapshot"],
    )
    if initially_up.any():
        must_stay_up = (min_up_time_set - up_time_before_set).clip(min=0)
        mask = (must_stay_up >= timesteps) & initially_up
        name = f"{c.name}-com-status-min_up_time_must_stay_up"
        mask = mask & active if active is not None else mask
        n.model.add_constraints(status, "=", 1, name=name, mask=mask)

    # down time before
    if initially_down.any():
        must_stay_down = (min_down_time_set - down_time_before_set).clip(min=0)
        mask = (must_stay_down >= timesteps) & initially_down
        name = f"{c.name}-com-status-min_down_time_must_stay_up"
        mask = mask & active if active is not None else mask
        n.model.add_constraints(status, "=", 0, name=name, mask=mask)

    # linearized approximation because committable can partly start up and shut down
    start_up_cost = c.da.start_up_cost.sel(name=com_i)
    shut_down_cost = c.da.shut_down_cost.sel(name=com_i)
    cost_equal = (start_up_cost == shut_down_cost).values

    # only valid additional constraints if start up costs equal to shut down costs
    if n._linearized_uc and not cost_equal.all():
        logger.warning(
            "The linear relaxation of the unit commitment cannot be "
            "tightened for all generators since the start up costs "
            "are not equal to the shut down costs. Proceed with the "
            "linear relaxation without the tightening by additional "
            "constraints for these. This might result in a longer "
            "solving time."
        )
    if n._linearized_uc and cost_equal.any():
        # dispatch limit for partly start up/shut down for t-1
        p_ce = p.loc[:, cost_equal]
        start_up_ce = start_up.loc[:, cost_equal]
        status_ce = status.loc[:, cost_equal]
        active_ce = DataArray(active.loc[:, cost_equal]).sel(snapshot=sns[1:])

        # parameters
        upper_p_ce = upper_p.loc[:, cost_equal]
        lower_p_ce = lower_p.loc[:, cost_equal]
        ramp_shut_down_ce = ramp_shut_down.loc[cost_equal]
        ramp_start_up_ce = ramp_start_up.loc[cost_equal]
        ramp_up_limit_ce = ramp_up_limit.loc[cost_equal]
        ramp_down_limit_ce = ramp_down_limit.loc[cost_equal]

        lhs_lower = (
            p_ce.shift(snapshot=1)
            - ramp_shut_down_ce * status_ce.shift(snapshot=1)
            - (upper_p_ce - ramp_shut_down_ce) * (status_ce - start_up_ce)
        )
        lhs_lower = lhs_lower.sel(snapshot=sns[1:])
        n.model.add_constraints(
            lhs_lower,
            "<=",
            0,
            name=f"{c.name}-com-p-before",
            mask=active_ce,
        )

        # dispatch limit for partly start up/shut down for t
        lhs_lower = (
            p_ce
            - upper_p_ce * status_ce
            + (upper_p_ce - ramp_start_up_ce) * start_up_ce
        )
        lhs_lower = lhs_lower.sel(snapshot=sns[1:])
        n.model.add_constraints(
            lhs_lower,
            "<=",
            0,
            name=f"{c.name}-com-p-current",
            mask=active_ce,
        )

        # ramp up if committable is only partly active and some capacity is starting up
        lhs_lower = (
            p_ce
            - p_ce.shift(snapshot=1)
            - (lower_p_ce + ramp_up_limit_ce) * status_ce
            + lower_p_ce * status_ce.shift(snapshot=1)
            + (lower_p_ce + ramp_up_limit_ce - ramp_start_up_ce) * start_up_ce
        )
        lhs_lower = lhs_lower.sel(snapshot=sns[1:])
        n.model.add_constraints(
            lhs_lower,
            "<=",
            0,
            name=f"{c.name}-com-partly-start-up",
            mask=active_ce,
        )

        # ramp down if committable is only partly active and some capacity is shutting up
        lhs_lower = (
            p_ce.shift(snapshot=1)
            - p_ce
            - ramp_shut_down_ce * status_ce.shift(snapshot=1)
            + (ramp_shut_down_ce - ramp_down_limit_ce) * status_ce
            - (lower_p_ce + ramp_down_limit_ce - ramp_shut_down_ce) * start_up_ce
        )
        lhs_lower = lhs_lower.sel(snapshot=sns[1:])
        n.model.add_constraints(
            lhs_lower,
            "<=",
            0,
            name=f"{c.name}-com-partly-shut-down",
            mask=active_ce,
        )


def define_nominal_constraints_for_extendables(
    n: Network, component: str, attr: str
) -> None:
    """Define capacity constraints for extendable components.

    Sets capacity expansion constraints for components with extendable
    capacities. For each component, the constraint enforces:

    min_capacity ≤ capacity ≤ max_capacity

    where capacity is a decision variable representing the component's
    optimal capacity.

    Applies to Components
    ---------------------
    Generator (p_nom), Line (s_nom), Transformer (s_nom), Link (p_nom),
    Store (e_nom), StorageUnit (p_nom)

    Parameters
    ----------
    n : pypsa.Network
        Network instance containing the model and component data
    component : str
        Name of the network component (e.g. "Generator", "StorageUnit")
    attr : str
        Name of the capacity attribute (e.g. "p_nom" for nominal power)

    Returns
    -------
    None

    Notes
    -----
    Components with infinite max_capacity values are handled through masking
    to avoid solver issues, particularly with GLPK which doesn't accept
    infinite values in constraints.

    """
    c = as_components(n, component)
    ext_i = c.extendables.difference(c.inactive_assets)

    if ext_i.empty:
        return

    capacity = n.model[f"{c.name}-{attr}"]
    lower = c.da[attr + "_min"].sel(name=ext_i)
    upper = c.da[attr + "_max"].sel(name=ext_i)

    n.model.add_constraints(capacity, ">=", lower, name=f"{c.name}-ext-{attr}-lower")

    is_finite = upper != inf
    if is_finite.any():
        n.model.add_constraints(
            capacity, "<=", upper, name=f"{c.name}-ext-{attr}-upper", mask=is_finite
        )


def define_ramp_limit_constraints(
    n: Network, sns: pd.Index, component: str, attr: str
) -> None:
    """Define ramp rate limit constraints for components.

    Sets ramp rate constraints to limit the change in output between
    consecutive time periods. The constraints are defined for fixed,
    extendable, and committable components, with different formulations
    for each case.

    Applies to Components
    ---------------------
    Generator (p), Line (s), Transformer (s), Link (p), Store (e), StorageUnit (p_dispatch, p_store, state_of_charge)

    Parameters
    ----------
    n : pypsa.Network
        Network instance containing the model and component data
    sns : pd.Index
        Set of snapshots for which to define the constraints
    component : str
        Name of the network component (e.g. "Generator")
    attr : str
        Name of the dispatch attribute (e.g. "p" for active power)

    Returns
    -------
    None

    Notes
    -----
    For rolling horizon optimization, the function handles linking between
    optimization windows by including the previous snapshot's dispatch value.

    For committable components, ramp constraints incorporate the unit commitment
    status and special ramp limits for start-up and shut-down periods.

    For extendable components, ramp constraints are defined relative to the
    variable capacity, ensuring consistency in the optimization.

    """
    m = n.model
    c = as_components(n, component)

    # Fix for as_dynamic function breaking with scenarios. TODO fix it OR leave this if clause
    if c.static.size == 0:
        return

    if {"ramp_limit_up", "ramp_limit_down"}.isdisjoint(c.static.columns):
        return

    ramp_limit_up = c.da.ramp_limit_up.sel(snapshot=sns)
    ramp_limit_down = c.da.ramp_limit_down.sel(snapshot=sns)

    # Skip if there are no ramp limits defined or if all are set to 1 (no limit)
    if (ramp_limit_up.isnull() & ramp_limit_down.isnull()).all():
        return
    if (ramp_limit_up == 1).all() and (ramp_limit_down == 1).all():
        return

    # ---------------- Check if ramping is at start of n.snapshots --------------- #

    attr = {"p", "p0"}.intersection(c.dynamic.keys()).pop()
    start_i = n.snapshots.get_loc(sns[0]) - 1
    p_start = c.dynamic[attr].iloc[start_i]

    # Get the dispatch value from previous snapshot if not at beginning
    is_rolling_horizon = sns[0] != n.snapshots[0] and not p_start.empty
    p = m[f"{c.name}-{attr}"]

    # Get different component groups for constraint application
    com_i = c.committables.difference(c.inactive_assets)
    fix_i = c.fixed.difference(c.inactive_assets)
    fix_i = fix_i.difference(com_i).rename(fix_i.name)
    ext_i = c.extendables.difference(c.inactive_assets)

    # Auxiliary variables for constraint application
    ext_dim = ext_i.name if ext_i.name else c.name
    original_ext_i = ext_i.copy()
    original_com_i = com_i.copy()

    if is_rolling_horizon:
        active = c.da.active.sel(name=fix_i, snapshot=sns)
        rhs_start = pd.DataFrame(0.0, index=sns, columns=c.static.index)
        rhs_start.loc[sns[0]] = p_start

        def p_actual(idx: pd.Index) -> DataArray:
            return p.sel(name=idx)

        def p_previous(idx: pd.Index) -> DataArray:
            return p.sel(name=idx).shift(snapshot=1)

    else:
        active = c.da.active.sel(name=fix_i, snapshot=sns[1:])
        rhs_start = pd.DataFrame(0.0, index=sns[1:], columns=c.static.index)
        rhs_start.index.name = "snapshot"

        def p_actual(idx: pd.Index) -> DataArray:
            return p.sel(name=idx).sel(snapshot=sns[1:])

        def p_previous(idx: pd.Index) -> DataArray:
            return p.sel(name=idx).shift(snapshot=1).sel(snapshot=sns[1:])

    rhs_start = DataArray(rhs_start)

    # ----------------------------- Fixed Components ----------------------------- #
    if not fix_i.empty:
        ramp_limit_up_fix = ramp_limit_up.sel(name=fix_i)
        ramp_limit_down_fix = ramp_limit_down.sel(name=fix_i)
        rhs_start_fix = rhs_start
        p_nom = c.da[c._operational_attrs["nom"]].sel(name=fix_i)

        # Ramp up constraints for fixed components
        non_null_up = ~ramp_limit_up_fix.isnull().all()
        if non_null_up.any():
            lhs = p_actual(fix_i) - p_previous(fix_i)
            rhs = (ramp_limit_up_fix * p_nom) + rhs_start_fix
            mask = active & non_null_up
            m.add_constraints(
                lhs, "<=", rhs, name=f"{c.name}-fix-{attr}-ramp_limit_up", mask=mask
            )

        # Ramp down constraints for fixed components
        non_null_down = ~ramp_limit_down_fix.isnull().all()
        if non_null_down.any():
            lhs = p_actual(fix_i) - p_previous(fix_i)
            rhs = (-ramp_limit_down_fix * p_nom) + rhs_start
            mask = active & non_null_down
            m.add_constraints(
                lhs, ">=", rhs, name=f"{c.name}-fix-{attr}-ramp_limit_down", mask=mask
            )

    # ----------------------------- Extendable Components ----------------------------- #
    if not ext_i.empty:
        # Redefine active mask over ext_i
        active_ext = (
            c.da.active.sel(name=ext_i, snapshot=sns)
            if is_rolling_horizon
            else c.da.active.sel(name=ext_i, snapshot=sns[1:])
        )

        ramp_limit_up_ext = ramp_limit_up.reindex(
            {"snapshot": active_ext.coords["snapshot"].values, "name": ext_i}
        ).rename({"name": ext_dim})
        ramp_limit_down_ext = ramp_limit_down.reindex(
            {"snapshot": active_ext.coords["snapshot"].values, "name": ext_i}
        ).rename({"name": ext_dim})
        rhs_start_ext = rhs_start.sel({"name": ext_i}).rename({"name": ext_dim})

        # For extendables, nominal capacity is a decision variable
        p_nom_var = m[f"{c.name}-{c._operational_attrs['nom']}"]

        if not ramp_limit_up_ext.isnull().all():
            lhs = (
                p_actual(original_ext_i)
                - p_previous(original_ext_i)
                - (ramp_limit_up_ext * p_nom_var)
            )
            mask = active_ext & (~ramp_limit_up_ext.isnull())
            m.add_constraints(
                lhs,
                "<=",
                rhs_start_ext,
                name=f"{c.name}-ext-{attr}-ramp_limit_up",
                mask=mask,
            )

        if not ramp_limit_down_ext.isnull().all():
            lhs = (
                p_actual(original_ext_i)
                - p_previous(original_ext_i)
                + (ramp_limit_down_ext * p_nom_var)
            )
            mask = active_ext & (~ramp_limit_down_ext.isnull())
            m.add_constraints(
                lhs,
                ">=",
                rhs_start_ext,
                name=f"{c.name}-ext-{attr}-ramp_limit_down",
                mask=mask,
            )
    # ----------------------------- Committable Components ----------------------------- #
    if not com_i.empty:
        # Redefine active mask over com_i and get parameters directly using component methods
        active_com = (
            c.da.active.sel(name=com_i, snapshot=sns)
            if is_rolling_horizon
            else c.da.active.sel(name=com_i, snapshot=sns[1:])
        )

        ramp_limit_up_com = ramp_limit_up.reindex(
            {"snapshot": active_com.coords["snapshot"].values, "name": com_i}
        )
        ramp_limit_down_com = ramp_limit_down.reindex(
            {"snapshot": active_com.coords["snapshot"].values, "name": com_i}
        )

        ramp_limit_start_up_com = c.da.ramp_limit_start_up.sel(name=com_i)
        ramp_limit_shut_down_com = c.da.ramp_limit_shut_down.sel(name=com_i)
        p_nom_com = c.da[c._operational_attrs["nom"]].sel(name=original_com_i)

        # Transform rhs_start for committable components
        rhs_start_com = rhs_start.sel(name=com_i)

        # com up
        non_null_up = ~ramp_limit_up_com.isnull()
        if non_null_up.any():
            limit_start = p_nom_com * ramp_limit_start_up_com
            limit_up = p_nom_com * ramp_limit_up_com

            status = m[f"{c.name}-status"].sel(
                snapshot=active_com.coords["snapshot"].values
            )
            status_prev = (
                m[f"{c.name}-status"]
                .shift(snapshot=1)
                .sel(snapshot=active_com.coords["snapshot"].values)
            )

            lhs = (
                p_actual(original_com_i)
                - p_previous(original_com_i)
                + (limit_start - limit_up) * status_prev
                - limit_start * status
            )

            rhs = rhs_start_com.copy()
            if is_rolling_horizon:
                status_start = c.dynamic["status"].iloc[start_i]
                limit_diff = (limit_up - limit_start).isel(snapshot=0)
                rhs.loc[{"snapshot": rhs.coords["snapshot"].item(0)}] += (
                    limit_diff * status_start
                )

            mask = active_com & non_null_up
            m.add_constraints(
                lhs, "<=", rhs, name=f"{c.name}-com-{attr}-ramp_limit_up", mask=mask
            )

        # com down
        non_null_down = ~ramp_limit_down_com.isnull()
        if non_null_down.any():
            limit_shut = p_nom_com * ramp_limit_shut_down_com
            limit_down = p_nom_com * ramp_limit_down_com

            status = m[f"{c.name}-status"].sel(
                snapshot=active_com.coords["snapshot"].values
            )
            status_prev = (
                m[f"{c.name}-status"]
                .shift(snapshot=1)
                .sel(snapshot=active_com.coords["snapshot"].values)
            )

            lhs = (
                p_actual(original_com_i)
                - p_previous(original_com_i)
                + (limit_down - limit_shut) * status
                + limit_shut * status_prev
            )

            rhs = rhs_start_com.copy()
            if is_rolling_horizon:
                status_start = c.dynamic["status"].iloc[start_i]
                rhs.loc[{"snapshot": rhs.coords["snapshot"].item(0)}] += (
                    -limit_shut * status_start
                )

            mask = active_com & non_null_down
            m.add_constraints(
                lhs, ">=", rhs, name=f"{c.name}-com-{attr}-ramp_limit_down", mask=mask
            )


def define_nodal_balance_constraints(
    n: Network,
    sns: pd.Index,
    transmission_losses: int = 0,
    buses: Sequence | None = None,
    suffix: str = "",
) -> None:
    """Define energy balance constraints at each node.

    Creates constraints ensuring that the sum of power injections at each node
    equals the demand at that node for each snapshot. This is the core constraint
    implementing Kirchhoff's Current Law (KCL) in the power system model. However,
    the logic is not limited to power networks and spans to other energy carriers.

    Using an example of power system, the general form of the constraint is:

    sum(power_injections) = sum(power_withdrawals)

    where power injections include generation, storage discharge, and incoming branch flows,
    while power withdrawals include loads, storage charging, and outgoing branch flows.

    Applies to Components
    ---------------------
    Generator (p), Line (s), Transformer (s), Link (p), Store (p), Load (p), StorageUnit (p_dispatch, p_store)*

    Notes
    -----
    * StorageUnit net power (p_dispatch - p_store) is calculated after optimization
    * StorageUnit (spill) var is not in the nodal balance - it's handled internally within the storage unit energy balance

    Parameters
    ----------
    n : pypsa.Network
        Network instance containing the model and component data
    sns : pd.Index
        Set of snapshots for which to define the constraints
    transmission_losses : int, default 0
        Number of segments for transmission loss linearization; if non-zero,
        losses are included in the power balance
    buses : Sequence | None, default None
        Subset of buses for which to define constraints; if None, all buses are used
    suffix : str, default ""
        Optional suffix to append to constraint names and dimensions

    Returns
    -------
    None

    Notes
    -----
    Link components with multiple buses are handled with their respective
    efficiency factors for conversion between energy carriers.

    The function raises an error if there's a bus with non-zero load but no
    connected components to provide power.

    """
    m = n.model
    if buses is None:
        buses = n.c.buses.static.index.unique("name")

    links = as_components(n, "Link")

    args: list[Any] = [
        ["Generator", "p", "bus", 1],
        ["Store", "p", "bus", 1],
        ["StorageUnit", "p_dispatch", "bus", 1],
        ["StorageUnit", "p_store", "bus", -1],
        ["Line", "s", "bus0", -1],
        ["Line", "s", "bus1", 1],
        ["Transformer", "s", "bus0", -1],
        ["Transformer", "s", "bus1", 1],
        ["Link", "p", "bus0", -1],
        ["Link", "p", "bus1", links.da.efficiency.sel(snapshot=sns)],
    ]

    if not links.empty:
        for i in n.c.links.additional_ports:
            eff_attr = f"efficiency{i}" if i != "1" else "efficiency"
            eff = links.da[eff_attr].sel(snapshot=sns)
            args.append(["Link", "p", f"bus{i}", eff])

    if transmission_losses:
        args.extend(
            [
                ["Line", "loss", "bus0", -0.5],
                ["Line", "loss", "bus1", -0.5],
                ["Transformer", "loss", "bus0", -0.5],
                ["Transformer", "loss", "bus1", -0.5],
            ]
        )

    exprs = []

    for component, attr, column, sign in args:
        c = as_components(n, component)
        if c.static.empty:
            continue

        if "sign" in c.static:
            sign = sign * c.da.sign

        expr = sign * m[f"{c.name}-{attr}"]

        cbuses = c._as_xarray(column)
        cbuses = cbuses.sel(name=c.active_assets)
        # Only keep the first scenario if there are multiple
        if n.has_scenarios:
            cbuses = cbuses.isel(scenario=0, drop=True)
        cbuses = cbuses[cbuses.isin(buses)].rename("Bus")

        if not cbuses.size:
            continue

        #  drop non-existent multiport buses which are ''
        if column in ["bus" + i for i in n.c.links.additional_ports]:
            cbuses = cbuses[cbuses != ""]

        expr = expr.sel(name=cbuses.coords["name"].values)
        if expr.size:
            exprs.append(expr.groupby(cbuses).sum().rename(Bus="name"))

    lhs = merge(exprs, join="outer").reindex(name=buses)

    # Prepare the RHS
    loads = as_components(n, "Load")

    if loads.static.empty:
        rhs = DataArray(
            0.0,
            coords={"snapshot": sns, "name": buses},
            dims=["snapshot", "name"],
        )
    else:
        loads_values = loads.da.p_set.where(
            loads.da.active.sel(name=loads.active_assets, snapshot=sns)
        )
        loads_values = loads_values.reindex(name=loads.static.index.unique("name"))
        load_buses = loads._as_xarray("bus").rename("Bus")
        if n.has_scenarios:
            load_buses = load_buses.isel(scenario=0, drop=True)

        # group by bus, then reindex over *all* buses (fill zeros where no loads)
        rhs = (
            loads_values.groupby(load_buses)
            .sum()
            .rename(Bus="name")
            .reindex(name=buses, fill_value=0)
        )

    empty_nodal_balance = (lhs.vars == -1).all("_term")

    if empty_nodal_balance.any():
        if (empty_nodal_balance & (rhs != 0)).any().item():
            msg = "Empty LHS with non-zero RHS in nodal balance constraint."
            raise ValueError(msg)

        mask = ~empty_nodal_balance
    else:
        mask = None

    n.model.add_constraints(lhs, "=", rhs, name=f"Bus{suffix}-nodal_balance", mask=mask)


def define_kirchhoff_voltage_constraints(n: Network, sns: pd.Index) -> None:
    """Define Kirchhoff's Voltage Law constraints for networks.

    Creates constraints ensuring that the sum of potential differences across
    branches around all cycles in the network must sum to zero. For each cycle
    in the network graph, the constraint enforces:

    sum_{l in cycle} x_l * s_l = 0

    where
        x_l : series reactance or resistance of branch l (depending on AC/DC)
        s_l : branch flow variable for branch l in the cycle

    Applies to Components
    ---------------------
    Line, Transformer, Link (passive branch components)

    Parameters
    ----------
    n : pypsa.Network
        Network instance containing the model and component data
    sns : pd.Index
        Set of snapshots for which to define the constraints

    Returns
    -------
    None

    Notes
    -----
    While there are different formulations of KVL, the cycle-based
    formulation was found to be much faster than other formulations
    due to its sparsity, as shown in [3]_.

    The function first determines the network topology including cycles for each
    network component (AC and DC sub-networks), then creates constraints for
    each cycle.

    For multi-investment period models, the function creates separate constraints
    for each investment period, reflecting the changing network topology over time.

    The impedances are scaled by 1e5 to improve numerical conditioning.

    References
    ----------
    .. [3] J. Hörsch et al., "Linear optimal power flow using cycle flows,"
       Electric Power Systems Research, vol. 158, pp. 126-135, 2018,
       https://doi.org/10.1016/j.epsr.2020.106908

    """
    m = n.model
    n.calculate_dependent_values()

    periods = sns.unique("period") if n._multi_invest else [None]
    lhs = []
    for period in periods:
        snapshots = sns if period is None else sns[sns.get_loc(period)]
        C = n.cycle_matrix(investment_period=period, apply_weights=True)
        if C.empty:
            continue

        exprs = []
        for c in C.index.unique("type"):
            C_branch = DataArray(C.loc[c])
            flow = m[f"{c}-s"].sel(
                snapshot=snapshots,
                name=C_branch.indexes["name"].difference(n.c[c].inactive_assets),
            )
            exprs.append(flow @ C_branch * 1e5)
        lhs.append(sum(exprs))

    if lhs:
        lhs = merge(lhs, dim="snapshot")
        m.add_constraints(lhs == 0, name="Kirchhoff-Voltage-Law")


def define_fixed_nominal_constraints(n: Network, component: str, attr: str) -> None:
    """Define constraints for fixing component capacities to specified values.

    Sets constraints to fix nominal (capacity) variables of components to values
    specified in the corresponding '_set' attribute.

    Applies to Components
    ---------------------
    Generator (p_nom), Line (s_nom), Transformer (s_nom), Link (p_nom),
    Store (e_nom), StorageUnit (p_nom)

    Parameters
    ----------
    n : pypsa.Network
        Network instance containing the model and component data
    component : str
        Name of the network component (e.g. "Generator", "StorageUnit")
    attr : str
        Name of the capacity attribute (e.g. "p_nom" for nominal power)

    Returns
    -------
    None

    Notes
    -----
    The function only creates constraints for components that have non-NaN
    values in their '{attr}_set' attribute.

    """
    c = as_components(n, component)
    if attr + "_set" not in c.static:
        return

    dim = f"{component}-{attr}_set_i"
    fix = c.static[attr + "_set"].dropna().rename_axis(dim)

    if fix.empty:
        return

    var = n.model[f"{component}-{attr}"]
    var = reindex(var, var.dims[0], fix.index)
    n.model.add_constraints(var, "=", fix, name=f"{component}-{attr}_set")


def define_modular_constraints(n: Network, component: str, attr: str) -> None:
    """Define constraints for modular capacity expansion.

    Sets constraints ensuring that the optimal capacity of a component is
    an integer multiple of a specified module size. This implements discrete
    capacity expansion for components with modular units.

    For each modular component, the constraint enforces:

    capacity = n_modules * module_size

    where n_modules is an integer decision variable and module_size is the
    specified size of each module.

    Applies to Components
    ---------------------
    Generator (p_nom), Line (s_nom), Transformer (s_nom), Link (p_nom),
    Store (e_nom), StorageUnit (p_nom)

    Parameters
    ----------
    n : pypsa.Network
        Network instance containing the model and component data
    component : str
        Name of the network component (e.g. "Generator", "StorageUnit")
    attr : str
        Name of the capacity attribute (e.g. "p_nom" for nominal power)

    Returns
    -------
    None

    Notes
    -----
    This function is used for components where capacity expansion must occur
    in discrete steps rather than continuous values, reflecting the reality
    of many energy system technologies.

    The function only applies to components that are both extendable and have
    a positive module size specified in the '{attr}_mod' attribute.

    """
    m = n.model
    c = as_components(n, component)

    ext_attr = f"{attr}_extendable"
    mod_attr = f"{attr}_mod"

    # Mask components that are both extendable and have a positive modular capacity
    mask = c.static[ext_attr] & (c.static[mod_attr] > 0)
    mod_i = c.static.index[mask]

    if (mod_i).empty:
        return

    # Get modular capacity values
    modular_capacity = c.da[mod_attr].sel(name=mod_i)

    # Get variables
    modularity = m[f"{c.name}-n_mod"]
    capacity = m.variables[f"{c.name}-{attr}"].loc[mod_i]

    con = capacity - modularity * modular_capacity.values == 0
    n.model.add_constraints(con, name=f"{c.name}-{attr}_modularity", mask=None)


def define_fixed_operation_constraints(
    n: Network, sns: pd.Index, component: str, attr: str
) -> None:
    """Define constraints for fixing operational variables to specified values.

    Sets constraints to fix dispatch variables of components to values specified
    in the corresponding '_set' attribute.

    Applies to Components
    ---------------------
    Generator (p), Line (s), Transformer (s), Link (p), Store (e), StorageUnit (p_dispatch, p_store, state_of_charge)

    Parameters
    ----------
    n : pypsa.Network
        Network instance containing the model and component data
    sns : pd.Index
        Set of snapshots for which to define the constraints
    component : str
        Name of the network component (e.g. "Generator", "StorageUnit")
    attr : str
        Name of the dispatch attribute (e.g. "p" for active power)

    Returns
    -------
    None

    Notes
    -----
    This function is useful for modeling must-run generators, fixed imports/exports,
    or pre-committed dispatch decisions.

    The function only creates constraints for snapshots and components where
    the '{attr}_set' values are not NaN and the component is active.

    """
    c = as_components(n, component)
    attr_set = f"{attr}_set"

    if attr_set not in c.dynamic.keys() or c.dynamic[attr_set].empty:
        return

    fix = c.da[attr_set].sel(snapshot=sns, name=c.active_assets)

    if fix.isnull().all():
        return

    active = c.da.active.sel(snapshot=sns, name=fix.coords["name"].values)
    mask = active & (~fix.isnull())

    var = n.model[f"{c.name}-{attr}"]

    n.model.add_constraints(var, "=", fix, name=f"{c.name}-" + attr_set, mask=mask)


def define_storage_unit_constraints(n: Network, sns: pd.Index) -> None:
    """Define energy balance constraints for storage units.

    Creates constraints ensuring energy conservation for storage units over time.
    For each storage unit and snapshot, the constraint enforces:

    soc(t) = standing_eff * soc(t-1) + eff_store * p_store(t)
                - (1/eff_dispatch) * p_dispatch(t)
                - spill(t) + inflow(t)

    where soc is the state of charge, p_store and p_dispatch are the
    charging and discharging power variables, and the efficiencies account
    for energy losses.

    Applies to Components
    ---------------------
    StorageUnit (p_dispatch, p_store, state_of_charge, spill)

    Parameters
    ----------
    n : pypsa.Network
        Network instance containing the model and component data
    sns : pd.Index
        Set of snapshots for which to define the constraints

    Returns
    -------
    None

    Notes
    -----
    The function handles different storage operating modes:
    - Cyclic storage (returning to initial state at the end of the period)
    - Non-cyclic storage (with specified initial state of charge)

    For multi-investment period models, the function supports both cycling
    within each period and carrying state of charge between periods.

    Standing losses are applied based on the elapsed hours between snapshots.

    """
    m = n.model
    component = "StorageUnit"
    dim = "snapshot"
    c = as_components(n, component)
    active = c.da.active.sel(snapshot=sns, name=c.active_assets)

    if c.static.empty:
        return

    # elapsed hours
    elapsed_h = expand_series(n.snapshot_weightings.stores[sns], c.static.index)
    eh = DataArray(elapsed_h)
    try:
        eh = eh.unstack("dim_1")
    except ValueError:
        pass

    # efficiencies as xarray DataArrays
    eff_stand = (1 - c.da.standing_loss.sel(snapshot=sns)) ** eh
    eff_dispatch = c.da.efficiency_dispatch.sel(snapshot=sns)
    eff_store = c.da.efficiency_store.sel(snapshot=sns)

    soc = m[f"{component}-state_of_charge"]

    lhs = [
        (-1, soc),
        (-1 / eff_dispatch * eh, m[f"{component}-p_dispatch"]),
        (eff_store * eh, m[f"{component}-p_store"]),
    ]

    if f"{component}-spill" in m.variables:
        lhs += [(-eh, m[f"{component}-spill"])]

    # We create a mask `include_previous_soc` which excludes the first snapshot
    # for non-cyclic assets
    noncyclic_b = ~c.da.cyclic_state_of_charge
    include_previous_soc = (active.cumsum(dim) != 1).where(noncyclic_b, True)

    previous_soc = (
        soc.where(active)
        .ffill(dim)
        .roll(snapshot=1)
        .ffill(dim)
        .where(include_previous_soc)
    )

    # We add inflow and initial soc for noncyclic assets to rhs
    soc_init = c.da.state_of_charge_initial
    rhs = -c.da.inflow.sel(snapshot=sns) * eh

    if isinstance(sns, pd.MultiIndex):
        # If multi-horizon optimizing, we update the previous_soc and the rhs
        # for all assets which are cyclic/non-cyclic per period
        periods = soc.coords["period"]
        per_period = (
            c.da.cyclic_state_of_charge_per_period
            | c.da.state_of_charge_initial_per_period
        )

        # We calculate the previous soc per period while cycling within a period
        # Normally, we should use groupby, but is broken for multi-index
        # see https://github.com/pydata/xarray/issues/6836
        ps = sns.unique("period")
        sl = slice(None)
        previous_soc_pp_list = [
            soc.data.sel(snapshot=(p, sl)).roll(snapshot=1) for p in ps
        ]
        previous_soc_pp = concat(previous_soc_pp_list, dim="snapshot")

        # We create a mask `include_previous_soc_pp` which excludes the first
        # snapshot of each period for non-cyclic assets
        include_previous_soc_pp = (periods == periods.shift(snapshot=1)) & active
        include_previous_soc_pp = include_previous_soc_pp.where(noncyclic_b, True)

        # Ensure that dimension order is consistent for stochastic networks
        if n.has_scenarios:
            expected_dims = list(include_previous_soc_pp.dims)
            if list(previous_soc_pp.dims) != expected_dims:
                previous_soc_pp = previous_soc_pp.transpose(*expected_dims)

        # We take values still to handle internal xarray multi-index difficulties
        previous_soc_pp = previous_soc_pp.where(
            include_previous_soc_pp.values, linopy.variables.FILL_VALUE
        )

        # update the previous_soc variables and right hand side
        previous_soc = previous_soc.where(~per_period, previous_soc_pp)
        include_previous_soc = include_previous_soc_pp.where(
            per_period, include_previous_soc
        )

    lhs += [(eff_stand, previous_soc)]

    rhs = rhs.where(include_previous_soc, rhs - soc_init)

    m.add_constraints(lhs, "=", rhs, name=f"{component}-energy_balance", mask=active)


def define_store_constraints(n: Network, sns: pd.Index) -> None:
    """Define energy balance constraints for stores.

    Creates constraints ensuring energy conservation for store components over time.
    For each store and snapshot, the constraint enforces:

    e(t) = eff_stand * e(t-1) + p(t) * elapsed_hours

    where
        e(t)        : energy level at time t
        eff_stand   : standing efficiency (1 - standing_loss)^elapsed_hours
        e(t-1)      : energy level at previous time step
        p(t)        : energy charging (positive), or discharging (negative)
        elapsed_hours: duration of the time step

    Applies to Components
    ---------------------
    Store (e, p)

    Parameters
    ----------
    n : pypsa.Network
        Network instance containing the model and component data
    sns : pd.Index
        Set of snapshots for which to define the constraints

    Returns
    -------
    None

    Notes
    -----
    Stores differ from storage units in that they have a single power variable
    that can be positive (charging) or negative (discharging) rather than
    separate variables for charge and discharge.

    The function handles different store operating modes:
    - Cyclic storage (returning to initial energy level at the end of the period)
    - Non-cyclic storage (with specified initial energy level)

    For multi-investment period models, the function supports both cycling
    within each period and carrying energy between periods.

    Standing losses are applied based on the elapsed hours between snapshots.

    """
    m = n.model
    component = "Store"
    dim = "snapshot"
    c = as_components(n, component)
    active = c.da.active.sel(snapshot=sns, name=c.active_assets)

    if c.static.empty:
        return

    # elapsed hours
    elapsed_h = expand_series(n.snapshot_weightings.stores[sns], c.active_assets)
    eh = DataArray(elapsed_h)

    # Unstack in stochastic networks with MultiIndex snapshots
    if n.has_scenarios and "dim_1" in eh.dims:
        eh = eh.unstack("dim_1")

    # standing efficiency
    eff_stand = (1 - c.da.standing_loss.sel(snapshot=sns, name=c.active_assets)) ** eh

    e = m[f"{component}-e"]
    p = m[f"{component}-p"]

    # Define LHS expression
    lhs = [(-1, e), (-eh, p)]

    # We create a mask `include_previous_e` which excludes the first snapshot
    # for non-cyclic assets
    noncyclic_b = ~c.da.e_cyclic.sel(name=c.active_assets)
    include_previous_e = (active.cumsum(dim) != 1).where(noncyclic_b, True)

    # Calculate previous energy state with proper handling of boundaries
    previous_e = (
        e.where(active).ffill(dim).roll(snapshot=1).ffill(dim).where(include_previous_e)
    )

    # We add initial e for non-cyclic assets to rhs
    e_init = c.da.e_initial.sel(name=c.active_assets)
    rhs = DataArray(0)

    if isinstance(sns, pd.MultiIndex):
        # If multi-horizon optimization, we update previous_e and the rhs
        # for all assets which are cyclic/non-cyclic per period
        periods = e.coords["period"]
        per_period = c.da.e_cyclic_per_period.sel(
            name=c.active_assets
        ) | c.da.e_initial_per_period.sel(name=c.active_assets)

        # We calculate the previous e per period while cycling within a period
        # Normally, we should use groupby, but it's broken for multi-index
        # see https://github.com/pydata/xarray/issues/6836
        ps = sns.unique("period")
        sl = slice(None)
        previous_e_pp_list = [e.data.sel(snapshot=(p, sl)).roll(snapshot=1) for p in ps]
        previous_e_pp = concat(previous_e_pp_list, dim="snapshot")

        # We create a mask `include_previous_e_pp` which excludes the first
        # snapshot of each period for non-cyclic assets
        include_previous_e_pp = active & (periods == periods.shift(snapshot=1))
        include_previous_e_pp = include_previous_e_pp.where(noncyclic_b, True)

        # We take values still to handle internal xarray multi-index difficulties
        previous_e_pp = previous_e_pp.where(
            include_previous_e_pp.values, linopy.variables.FILL_VALUE
        )

        # update previous_e variables and rhs
        previous_e = previous_e.where(~per_period, previous_e_pp)
        include_previous_e = include_previous_e_pp.where(per_period, include_previous_e)

    # Add the previous energy term with standing efficiency factor
    lhs += [(eff_stand, previous_e)]

    # For snapshots where we don't include previous_e, we need to account for initial values
    rhs = -e_init.where(~include_previous_e, 0)

    m.add_constraints(lhs, "=", rhs, name=f"{component}-energy_balance", mask=active)


def define_loss_constraints(
    n: Network, sns: pd.Index, component: str, transmission_losses: int
) -> None:
    """Define power loss constraints for passive branches.

    This function approximates quadratic power losses using piecewise linear
    constraints. It creates tangent segments to the quadratic loss curve
    to model the relationship between power flow and losses.

    See equations (39)-(46) in [1]_ for further details on the formulation.

    Applies to Components
    ---------------------
    Line, Transformer (passive branch components when transmission_losses > 0)

    Parameters
    ----------
    n : pypsa.Network
        Network instance containing the model and branch data
    sns : pd.Index
        Set of snapshots for which to define the constraints
    component : str
        Name of the passive branch component (e.g. "Line", "Transformer")
    transmission_losses : int
        Number of tangent segments to use in the piecewise linearization
        of the quadratic loss function; higher values increase accuracy
        but also computational complexity

    Returns
    -------
    None

    Notes
    -----
    3 segments offer a good trade-off between accuracy and solver performance.

    References
    ----------
    .. [1] F. Neumann, T. Brown, "Transmission losses in power system
       optimization models: A comparison of heuristic and exact solution methods,"
       Applied Energy, 2022, https://doi.org/10.1016/j.apenergy.2022.118859

    """
    c = as_components(n, component)

    if c.static.empty or component not in n.passive_branch_components:
        return

    tangents = transmission_losses
    active = c.da.active.sel(snapshot=sns, name=c.active_assets)

    s_max_pu = c.da.s_max_pu.sel(snapshot=sns)

    # Define nominal capacity (depends on extendability of lines)
    is_extendable = c.da.s_nom_extendable
    s_nom_max = c.da.s_nom_max.where(is_extendable, c.da.s_nom)

    if not isfinite(s_nom_max).all():
        msg = (
            f"Loss approximation requires finite 's_nom_max' for extendable "
            f"branches:\n {s_nom_max.sel(name=~isfinite(s_nom_max))}"
        )
        raise ValueError(msg)

    r_pu_eff = c.da.r_pu_eff

    # Calculate upper bound on losses
    upper_limit = r_pu_eff * (s_max_pu * s_nom_max) ** 2

    # Get variables
    loss = n.model[f"{c.name}-loss"]
    flow = n.model[f"{c.name}-s"]

    # Add upper limit constraint
    n.model.add_constraints(
        loss <= upper_limit, name=f"{c.name}-loss_upper", mask=active
    )

    # Add linearization constraints for each tangent segment
    for k in range(1, tangents + 1):
        # Calculate linearization parameters for segment k
        p_k = k / tangents * s_max_pu * s_nom_max
        loss_k = r_pu_eff * p_k**2
        slope_k = 2 * r_pu_eff * p_k
        offset_k = loss_k - slope_k * p_k

        # Add constraints for both positive and negative flow
        for sign in [-1, 1]:
            lhs = n.model.linexpr((1, loss), (sign * slope_k, flow))
            n.model.add_constraints(
                lhs >= offset_k, name=f"{c.name}-loss_tangents-{k}-{sign}", mask=active
            )


def define_total_supply_constraints(
    n: Network, sns: Sequence, component: str = "Generator"
) -> None:
    """Define energy sum constraints for generators.

    Creates constraints limiting the total energy generated by each generator
    over the specified snapshots. The constraints can enforce both minimum
    and maximum energy production requirements.

    For generators with e_sum_min, the constraint enforces:

    sum(p(t) * weighting(t)) ≥ e_sum_min

    For generators with e_sum_max, the constraint enforces:

    sum(p(t) * weighting(t)) ≤ e_sum_max

    where the sum is taken over all snapshots and weighting accounts for the
    duration of each snapshot.

    Applies to Components
    ---------------------
    Generator (by default, but component parameter can be changed)

    Parameters
    ----------
    n : pypsa.Network
        Network instance containing the model and component data
    sns : Sequence
        Set of snapshots for which to define the constraints
    component : str, default "Generator"
        Name of the network component to apply the constraints to

    Returns
    -------
    None

    Notes
    -----
    These constraints are useful for modeling:
    - Minimum energy production requirements (e.g., contracted energy delivery)
    - Maximum energy production limits (e.g., fuel availability, water reservoir limits)
    - Must-run generators with flexibility in when to produce

    The constraints only apply to generators that have finite e_sum_min or
    e_sum_max values specified.

    """
    sns_ = as_index(n, sns, "snapshots")
    m = n.model
    c = as_components(n, component)

    if c.static.empty:
        return

    # elapsed hours
    eh = DataArray(
        expand_series(n.snapshot_weightings.generators[sns_], c.static.index)
    )
    # Unstack in stochastic networks with MultiIndex snapshots
    if n.has_scenarios:
        eh = eh.unstack("dim_1")

    def _extract_names(index: pd.Index) -> pd.Index:
        """Extract name level from MultiIndex or return as-is."""
        return (
            index.get_level_values("name")
            if isinstance(index, pd.MultiIndex)
            else index
        )

    # minimum energy production constraints
    e_sum_min_i = c.static.index[c.static.e_sum_min > -inf]
    if not e_sum_min_i.empty:
        names = _extract_names(e_sum_min_i)
        e_sum_min = c.da.e_sum_min.sel(name=names)
        p = m[f"{c.name}-p"].sel(name=names, snapshot=sns_)
        eh_selected = eh.sel(name=names)
        energy = (p * eh_selected).sum(dim="snapshot")
        m.add_constraints(energy, ">=", e_sum_min, name=f"{c.name}-e_sum_min")

    # maximum energy production constraints
    e_sum_max_i = c.static.index[c.static.e_sum_max < inf]
    if not e_sum_max_i.empty:
        names = _extract_names(e_sum_max_i)
        e_sum_max = c.da.e_sum_max.sel(name=names)
        p = m[f"{c.name}-p"].sel(name=names, snapshot=sns_)
        eh_selected = eh.sel(name=names)
        energy = (p * eh_selected).sum(dim="snapshot")
        m.add_constraints(energy, "<=", e_sum_max, name=f"{c.name}-e_sum_max")
