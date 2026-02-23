# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Define optimisation constraints from PyPSA networks with Linopy."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import linopy
import pandas as pd
import xarray as xr
from linopy import merge
from numpy import inf, isfinite, maximum, sqrt, tile
from xarray import DataArray, concat, where

from pypsa.common import as_index, expand_series
from pypsa.components._types.links import Links
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


def define_operational_constraints_for_non_extendables(
    n: Network,
    sns: pd.Index,
    component: str,
    attr: str,
    transmission_losses: bool | int | dict = False,
) -> None:
    """Define operational constraints (lower-/upper bound).

    Sets operational constraints for a subset of non-extendable
    and non-committable components based on their bounds. For each component,
    the constraint enforces:

    lower_bound ≤ dispatch ≤ upper_bound

    where lower_bound and upper_bound are computed from the component's nominal
    capacity and min/max per unit values.

    Applies to Generator (p), Line (s), Transformer (s), Link (p), Store (e),
    StorageUnit (p_dispatch, p_store, state_of_charge).

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
    transmission_losses : int | dict
        If truthy, transmission losses are considered in the operational
        constraints for passive branches.

    Notes
    -----
    For passive branches with transmission losses, the constraint accounts for
    the losses in both directions, see justification in [1]_.

    References
    ----------
    [1] F. Neumann, T. Brown, "Transmission losses in power system
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
    n: Network,
    sns: pd.Index,
    component: str,
    attr: str,
    transmission_losses: bool | int | dict = False,
) -> None:
    """Define operational constraints (lower-/upper bound) for extendable components.

    Sets operational constraints for extendable components based on their bounds.
    For each component, the constraint enforces:

    lower_bound ≤ dispatch ≤ upper_bound

    where lower_bound and upper_bound are computed from the component's nominal
    capacity and min/max per unit values.

    Applies to Generator (p), Line (s), Transformer (s), Link (p), Store (e),
    StorageUnit (p_dispatch, p_store, state_of_charge).

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
    transmission_losses : int | dict
        If truthy, transmission losses are considered in the operational
        constraints for passive branches.

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
    """Define operational constraints for committable components.

    Sets operational constraints for components with unit commitment
    decisions. Supports both fixed-capacity and extendable committable
    components (using big-M formulation for the latter).

    Applies to Components
    ---------------------
    Generator, Link

    Parameters
    ----------
    n : pypsa.Network
        Network instance containing the model and component data
    sns : pd.Index
        Set of snapshots for which to define the constraints
    component : str
        Name of the network component ("Generator" or "Link")

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
    [2] Y. Hua, C. Liu, J. Zhang, "Representing Operational
        Flexibility in Generation Expansion Planning Through Convex Relaxation
        of Unit Commitment," IEEE Transactions on Power Systems, vol. 32,
        no. 5, pp. 3854-3865, 2017, https://doi.org/10.1109/TPWRS.2017.2735026

    """
    c = as_components(n, component)
    com_i = c.committables.difference(c.inactive_assets)

    if com_i.empty:
        return

    status = n.model[f"{c.name}-status"]
    start_up = n.model[f"{c.name}-start_up"]
    shut_down = n.model[f"{c.name}-shut_down"]
    status_diff = status - status.shift(snapshot=1)
    p = n.model[f"{c.name}-p"].sel(name=com_i)
    active = c.da.active.sel(name=com_i, snapshot=sns)

    ext_i = c.extendables.difference(c.inactive_assets)
    com_ext_i = com_i.intersection(ext_i).difference(c.modulars)
    com_fix_i = com_i.difference(ext_i)

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
    ramp_start_up = nominal * c.da.ramp_limit_start_up.sel(name=com_i).fillna(1)
    ramp_shut_down = nominal * c.da.ramp_limit_shut_down.sel(name=com_i).fillna(1)
    up_time_before_set = c.da.up_time_before.sel(name=com_i)
    down_time_before_set = c.da.down_time_before.sel(name=com_i)
    initially_up = up_time_before_set.astype(bool)
    initially_down = down_time_before_set.astype(bool)

    # check if there are status calculated/fixed before given sns interval
    if sns[0] != n.snapshots[0]:
        start_i = n.snapshots.get_loc(sns[0])
        prev_sns = n.snapshots[:start_i][::-1]
        until_start_up = c.da.status.sel(name=com_i, snapshot=prev_sns)
        ref = DataArray(range(1, len(prev_sns) + 1), dims="snapshot")
        up_time_before = until_start_up.where(
            until_start_up.cumsum("snapshot") == ref
        ).sum("snapshot")
        up_time_before_set = up_time_before.clip(max=min_up_time_set)
        initially_up = up_time_before_set.astype(bool)
        until_start_down = ~until_start_up.astype(bool)
        down_time_before = until_start_down.where(
            until_start_down.cumsum("snapshot") == ref
        ).sum("snapshot")
        down_time_before_set = down_time_before.clip(max=min_down_time_set)
        initially_down = down_time_before_set.astype(bool)

    if not com_ext_i.empty:
        p_nom_var = n.model[f"{c.name}-{c._operational_attrs['nom']}"]
        M_values = c.get_committable_big_m_values(
            names=com_ext_i, max_pu=max_pu, committable_big_m=n._committable_big_m
        )
        p_ext = p.sel(name=com_ext_i)
        status_ext = status.sel(name=com_ext_i)
        p_nom_ext = p_nom_var.sel(name=com_ext_i)
        min_pu_ext = min_pu.sel(name=com_ext_i)
        max_pu_ext = max_pu.sel(name=com_ext_i)

        active_ext = active.sel(name=com_ext_i)
        lhs_lower_ext = (1, p_ext), (-min_pu_ext, p_nom_ext), (-M_values, status_ext)
        n.model.add_constraints(
            lhs_lower_ext,
            ">=",
            -M_values,
            name=f"{c.name}-com-ext-p-lower",
            mask=active_ext,
        )

        lhs_upper_ext = (1, p_ext), (-M_values, status_ext)
        n.model.add_constraints(
            lhs_upper_ext,
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

    # Operational constraints for modular committable components
    # For modular components, use p_nom_mod * status instead of p_nom * status
    com_mod_i = com_i.intersection(c.modulars)
    if not com_mod_i.empty:
        p_mod = p.sel(name=com_mod_i)
        status_mod = status.sel(name=com_mod_i)
        active_mod = active.sel(name=com_mod_i)

        # Get module size (p_nom_mod, s_nom_mod, e_nom_mod)
        mod_attr = c._operational_attrs["nom_mod"]
        nominal_mod = c.da[mod_attr].sel(name=com_mod_i)

        # Get min/max_pu for modular components
        min_pu_mod = min_pu.sel(name=com_mod_i)
        max_pu_mod = max_pu.sel(name=com_mod_i)

        # Calculate bounds using module size
        lower_p_mod = min_pu_mod * nominal_mod
        upper_p_mod = max_pu_mod * nominal_mod

        # Lower constraint: p >= min_pu * p_nom_mod * status
        lhs_lower_mod = (1, p_mod), (-lower_p_mod, status_mod)
        n.model.add_constraints(
            lhs_lower_mod,
            ">=",
            0,
            name=f"{c.name}-com-mod-p-lower",
            mask=active_mod,
        )

        # Upper constraint: p <= max_pu * p_nom_mod * status
        lhs_upper_mod = (1, p_mod), (-upper_p_mod, status_mod)
        n.model.add_constraints(
            lhs_upper_mod,
            "<=",
            0,
            name=f"{c.name}-com-mod-p-upper",
            mask=active_mod,
        )

    # state-transition constraint
    rhs = pd.DataFrame(0, sns, com_i)
    # Convert xarray boolean to list of indices for DataFrame indexing
    initially_up_indices = com_i[initially_up.values]
    if not initially_up_indices.empty:
        rhs.loc[sns[0], initially_up_indices] = -1

    lhs = start_up - status_diff
    n.model.add_constraints(
        lhs, ">=", rhs, name=f"{c.name}-com-transition-start-up", mask=active
    )

    rhs = pd.DataFrame(0, sns, com_i)
    if not initially_up_indices.empty:
        rhs.loc[sns[0], initially_up_indices] = 1

    lhs = shut_down + status_diff
    n.model.add_constraints(
        lhs, ">=", rhs, name=f"{c.name}-com-transition-shut-down", mask=active
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
        lhs = -status.loc[:, min_up_time_i] + merge(expr, dim=com_i.name)
        lhs = lhs.sel(snapshot=sns[1:])
        n.model.add_constraints(
            lhs,
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
        lhs = status.loc[:, min_down_time_i] + merge(expr, dim=com_i.name)
        lhs = lhs.sel(snapshot=sns[1:])
        n.model.add_constraints(
            lhs,
            "<=",
            1,
            name=f"{c.name}-com-down-time",
            mask=active.loc[sns[1:], min_down_time_i],
        )
    # up time before
    timesteps = xr.DataArray(
        [range(1, len(sns) + 1)] * len(com_i),
        coords=[com_i, sns],
        dims=["name", "snapshot"],
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

        lhs = (
            p_ce.shift(snapshot=1)
            - ramp_shut_down_ce * status_ce.shift(snapshot=1)
            - (upper_p_ce - ramp_shut_down_ce) * (status_ce - start_up_ce)
        )
        lhs = lhs.sel(snapshot=sns[1:])
        n.model.add_constraints(
            lhs,
            "<=",
            0,
            name=f"{c.name}-com-p-before",
            mask=active_ce,
        )

        # dispatch limit for partly start up/shut down for t
        lhs = (
            p_ce
            - upper_p_ce * status_ce
            + (upper_p_ce - ramp_start_up_ce) * start_up_ce
        )
        lhs = lhs.sel(snapshot=sns[1:])
        n.model.add_constraints(
            lhs,
            "<=",
            0,
            name=f"{c.name}-com-p-current",
            mask=active_ce,
        )

        # ramp up if committable is only partly active and some capacity is starting up
        lhs = (
            p_ce
            - p_ce.shift(snapshot=1)
            - (lower_p_ce + ramp_up_limit_ce) * status_ce
            + lower_p_ce * status_ce.shift(snapshot=1)
            + (lower_p_ce + ramp_up_limit_ce - ramp_start_up_ce) * start_up_ce
        )
        lhs = lhs.sel(snapshot=sns[1:])
        n.model.add_constraints(
            lhs,
            "<=",
            0,
            name=f"{c.name}-com-partly-start-up",
            mask=active_ce,
        )

        # ramp down if committable is only partly active and some capacity is shutting up
        lhs = (
            p_ce.shift(snapshot=1)
            - p_ce
            - ramp_shut_down_ce * status_ce.shift(snapshot=1)
            + (ramp_shut_down_ce - ramp_down_limit_ce) * status_ce
            - (lower_p_ce + ramp_down_limit_ce - ramp_shut_down_ce) * start_up_ce
        )
        lhs = lhs.sel(snapshot=sns[1:])
        n.model.add_constraints(
            lhs,
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

    Applies to Generator (p_nom), Line (s_nom), Transformer (s_nom), Link (p_nom),
    Store (e_nom), StorageUnit (p_nom).

    Parameters
    ----------
    n : pypsa.Network
        Network instance containing the model and component data
    component : str
        Name of the network component (e.g. "Generator", "StorageUnit")
    attr : str
        Name of the capacity attribute (e.g. "p_nom" for nominal power)

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


def _define_ramp_limit_big_m(
    n: Network,
    sns: pd.Index | pd.MultiIndex,
    c: Any,
    attr: str,
    idx: pd.Index,
    limit_up: DataArray,
    limit_down: DataArray,
    limit_start: DataArray,
    limit_shut: DataArray,
    no_up_limit: DataArray,
    no_down_limit: DataArray,
    mask: DataArray,
) -> None:
    """Add big-M ramp constraints for committable+extendable components."""
    m = n.model
    var_attr = "p"
    nom_attr = c._operational_attrs["nom"]
    hist_attr = "p0" if c.name in n.branch_components else "p"
    is_rolling_horizon = (sns[0] != n.snapshots[0]) & (not c.dynamic[hist_attr].empty)
    filter_first_sn = DataArray([1] + [0] * (len(sns) - 1), coords=[sns])

    M = c.get_committable_big_m_values(
        names=idx, committable_big_m=n._committable_big_m
    )

    p = m[f"{c.name}-{var_attr}"].sel(name=idx)
    p_nom = m[f"{c.name}-{nom_attr}"].sel(name=idx)
    status = m[f"{c.name}-status"].sel(name=idx)
    start_up = m[f"{c.name}-start_up"].sel(name=idx)
    shut_down = m[f"{c.name}-shut_down"].sel(name=idx)

    if is_rolling_horizon:
        start_i = n.snapshots.get_loc(sns[0]) - 1
        p_init = c.da[hist_attr][start_i].sel(name=idx)
        s_init = c.da.status[start_i].sel(name=idx).fillna(1)
    else:
        initially_up = c.da.up_time_before.sel(name=idx) > 0
        p_init = c.da.p_init.sel(name=idx).where(initially_up, 0)
        s_init = initially_up

    p_prev_ce = p.shift(snapshot=1) + p_init.fillna(0) * filter_first_sn
    status_prev_ce = status.shift(snapshot=1) + s_init.fillna(0) * filter_first_sn

    lhs_delta = p - p_prev_ce
    mask = mask.sel(name=idx)
    mask_up = mask & ~no_up_limit.sel(name=idx)
    mask_down = mask & ~no_down_limit.sel(name=idx)

    lu = limit_up.sel(name=idx)
    ld = limit_down.sel(name=idx)
    ls = limit_start.sel(name=idx)
    lsh = limit_shut.sel(name=idx)

    m.add_constraints(
        lhs_delta <= lu * p_nom + M * (1 - status_prev_ce),
        name=f"{c.name}-{attr}-ramp_limit_up-run-bigM",
        mask=mask_up,
    )

    m.add_constraints(
        lhs_delta <= ls * p_nom + M * (1 - start_up),
        name=f"{c.name}-{attr}-ramp_limit_up-start-bigM",
        mask=mask_up,
    )

    m.add_constraints(
        lhs_delta >= -ld * p_nom - M * (1 - status),
        name=f"{c.name}-{attr}-ramp_limit_down-run-bigM",
        mask=mask_down,
    )

    m.add_constraints(
        lhs_delta >= -lsh * p_nom - M * (1 - shut_down),
        name=f"{c.name}-{attr}-ramp_limit_down-shut-bigM",
        mask=mask_down,
    )


def define_ramp_limit_constraints(
    n: Network, sns: pd.Index, component: str, attr: str
) -> None:
    """Define ramp rate limit constraints for components.

    Sets ramp rate constraints to limit the change in output between
    consecutive time periods. The constraints are defined for fixed,
    extendable, and committable components, with different formulations
    for each case.

    Applies to Generator (p) and Link (p).

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
    c = n.c[component]
    var_attr = "p"
    nom_attr = c._operational_attrs["nom"]
    hist_attr = "p0" if component in n.branch_components else "p"

    if {"ramp_limit_up", "ramp_limit_down"}.isdisjoint(c.static.columns):
        return

    if c.static.empty:
        return

    idx = c.static.index
    kwargs = {"level": "name"} if isinstance(idx, pd.MultiIndex) else {}
    is_ext = idx.isin(c.extendables, **kwargs)
    is_com = idx.isin(c.committables, **kwargs)
    is_modular = idx.isin(c.modulars, **kwargs)
    is_com_ext = is_com & is_ext & ~is_modular
    is_com_ext_mod = is_com & is_ext & is_modular
    is_com_fix = is_com & ~is_com_ext

    limit_up = c.da.ramp_limit_up.sel(snapshot=sns)
    limit_down = c.da.ramp_limit_down.sel(snapshot=sns)
    limit_start = c.da.ramp_limit_start_up
    limit_shut = c.da.ramp_limit_shut_down
    mask = c.da.active.sel(snapshot=sns)

    no_up_limit = limit_up.isnull() & limit_start.isnull()
    no_down_limit = limit_down.isnull() & limit_shut.isnull()
    all_null = (no_up_limit & no_down_limit).all()
    if all_null:
        return

    limit_up = limit_up.fillna(1.0)
    limit_down = limit_down.fillna(1.0)
    limit_start = limit_start.fillna(1.0)
    limit_shut = limit_shut.fillna(1.0)

    is_rolling_horizon = (sns[0] != n.snapshots[0]) & (not c.dynamic[hist_attr].empty)
    filter_first_sn = DataArray([1] + [0] * (len(sns) - 1), coords=[sns])

    nom_mod_attr = c._operational_attrs["nom_mod"]
    p_nom = c.da[nom_attr].where(~is_ext, 0)
    if is_com_ext_mod.any():
        p_nom = p_nom.where(~is_com_ext_mod, c.da[nom_mod_attr])
    is_ext_main = is_ext & ~is_com_ext & ~is_com_ext_mod
    p_nom_ext_var = None
    if is_ext_main.any():
        ext_main_names = idx[is_ext_main]
        p_nom_ext_var = m[f"{c.name}-{nom_attr}"].sel(name=ext_main_names)

    status = DataArray(1, coords=[sns, idx])
    if is_com_fix.any():
        com_main_names = idx[is_com_fix]
        status = status.where(~is_com_fix, 0)
        status = linopy.LinearExpression(status, m)
        status_var = m[f"{c.name}-status"].sel(name=com_main_names)
        status = (status + status_var).loc[:, status.indexes["name"]]

    if is_rolling_horizon:
        start_i = n.snapshots.get_loc(sns[0]) - 1
        p_init = c.da[hist_attr][start_i]
        s_init = c.da.status[start_i].fillna(1)
    else:
        initially_up = c.da.up_time_before > 0
        p_init = c.da.p_init.where(initially_up, 0)
        s_init = initially_up
        mask[0] = p_init.notnull()

    p = m[f"{c.name}-{var_attr}"]
    p_prev = p.shift(snapshot=1) + p_init.fillna(0) * filter_first_sn
    status_shifted = status.shift(snapshot=1)
    if not is_com_fix.any():
        status_shifted = status_shifted.fillna(0)
    status_prev = status_shifted + s_init.fillna(0) * filter_first_sn

    non_com_ext = ~is_com_ext
    lhs = p - p_prev
    rhs = limit_up * p_nom * status_prev
    if is_com_fix.any():
        rhs = rhs + limit_start * p_nom * (status - status_prev)
    if p_nom_ext_var is not None:
        if is_rolling_horizon:
            s_init_ext = c.da.status[start_i].sel(name=ext_main_names).fillna(1)
        else:
            s_init_ext = (c.da.up_time_before.sel(name=ext_main_names) > 0) * 1.0
        sp_ext = (1 - filter_first_sn) + s_init_ext * filter_first_sn
        if not isinstance(rhs, linopy.LinearExpression):
            rhs = linopy.LinearExpression(rhs, m)
        rhs = rhs + limit_up.sel(name=ext_main_names) * p_nom_ext_var * sp_ext
        if is_com_fix.any():
            ds_ext = filter_first_sn * (1 - s_init_ext)
            rhs = rhs + limit_start.sel(name=ext_main_names) * p_nom_ext_var * ds_ext
    mask_up = mask & ~no_up_limit & non_com_ext
    m.add_constraints(lhs <= rhs, name=f"{c.name}-{attr}-ramp_limit_up", mask=mask_up)

    lhs = p - p_prev
    rhs = -limit_down * p_nom * status
    if is_com_fix.any():
        rhs = rhs - limit_shut * p_nom * (status_prev - status)
    if p_nom_ext_var is not None:
        if not isinstance(rhs, linopy.LinearExpression):
            rhs = linopy.LinearExpression(rhs, m)
        rhs = rhs - limit_down.sel(name=ext_main_names) * p_nom_ext_var
        if is_com_fix.any():
            ds_ext = filter_first_sn * (1 - s_init_ext)
            rhs = rhs + limit_shut.sel(name=ext_main_names) * p_nom_ext_var * ds_ext
    mask_down = mask & ~no_down_limit & non_com_ext
    m.add_constraints(
        lhs >= rhs, name=f"{c.name}-{attr}-ramp_limit_down", mask=mask_down
    )

    if is_com_ext.any():
        _define_ramp_limit_big_m(
            n,
            sns,
            c,
            attr,
            idx[is_com_ext],
            limit_up,
            limit_down,
            limit_start,
            limit_shut,
            no_up_limit,
            no_down_limit,
            mask,
        )


def define_nodal_balance_constraints(
    n: Network,
    sns: pd.Index,
    transmission_losses: bool | int | dict = False,
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

    Applies to Generator (p), Line (s), Transformer (s), Link (p), Store (p),
    Load (p), StorageUnit (p_dispatch, p_store).

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
    transmission_losses : int | dict, default 0
        If truthy, transmission losses are considered in the power balance.
    buses : Sequence | None, default None
        Subset of buses for which to define constraints; if None, all buses are used
    suffix : str, default ""
        Optional suffix to append to constraint names and dimensions

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

    links = n.components["Link"]

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
    ]

    # Group link ports by (delay, cyclic_delay)
    # - Non delayed (delay=0) go into the standard args list
    # - Delayed groups are collected separately and are time-shifted
    delayed_link_args: list[tuple[str, Any, pd.Index, int, bool]] = []
    if not links.empty:
        active = links.active_assets
        for i in ["1"] + n.c.links.additional_ports:
            i_suffix = "" if i == "1" else i
            eff = links.da[f"efficiency{i_suffix}"].sel(snapshot=sns)
            delay_col = f"delay{i_suffix}"
            cyclic_col = f"cyclic_delay{i_suffix}"

            if delay_col in links.static.columns:
                delays = links.static[delay_col]
                cyclics = links.static[cyclic_col]
            else:
                delays = 0
                cyclics = True

            # Group links sharing the same (delay, cyclic) configuration
            for (d, cyc), group in links.static.assign(
                _delay=delays, _cyclic=cyclics
            ).groupby(["_delay", "_cyclic"]):
                names = group.index
                # Manually handle stochastic dimension
                if isinstance(names, pd.MultiIndex):
                    names = names.get_level_values("name").unique()
                names = names.intersection(active)
                if names.empty:
                    continue
                if int(d) == 0:
                    args.append(["Link", "p", f"bus{i}", eff.sel(name=names)])
                else:
                    delayed_link_args.append(
                        (f"bus{i}", eff.sel(name=names), names, int(d), bool(cyc))
                    )

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

    # For delayed links, time shift the p variable so that output at
    # snapshot t uses the input from the source snapshot s(t)
    for bus_col, eff, names, delay, is_cyclic in delayed_link_args:
        active_names = names.intersection(active)
        if active_names.empty:
            continue

        # Map each target snapshot to its source snapshot position
        src_snapshot_pos, valid = Links.get_delay_source_indexer(
            sns,
            n.snapshot_weightings.generators.loc[sns],
            delay,
            is_cyclic,
        )
        # Zero out invalid positions (non-cyclic: before horizon start)
        sns_coords: xr.Coordinates | dict[str, Any]
        if isinstance(sns, pd.MultiIndex):
            sns_coords = xr.Coordinates.from_pandas_multiindex(sns, "snapshot")
        else:
            sns_coords = {"snapshot": sns}
        valid_mask = DataArray(
            valid.astype(float), dims=["snapshot"], coords=sns_coords
        )

        link_p = m["Link-p"].sel(name=active_names)
        shifted_p = link_p.isel(snapshot=src_snapshot_pos).assign_coords(sns_coords)
        shifted_p = shifted_p * valid_mask
        expr = eff.sel(name=active_names) * shifted_p
        cbuses = links._as_xarray(bus_col).sel(name=active_names)
        if n.has_scenarios:
            cbuses = cbuses.isel(scenario=0, drop=True)
        cbuses = cbuses[cbuses.isin(buses)].rename("Bus")
        cbuses = cbuses[cbuses != ""]
        if not cbuses.size:
            continue
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

    Applies to Line, Transformer, and Link (passive branch components).

    Parameters
    ----------
    n : pypsa.Network
        Network instance containing the model and component data
    sns : pd.Index
        Set of snapshots for which to define the constraints

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
    [3] J. Hörsch et al., "Linear optimal power flow using cycle flows,"
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
        con = lhs == 0
        mask = con.rhs.notnull()
        m.add_constraints(con, name="Kirchhoff-Voltage-Law", mask=mask)


def define_fixed_nominal_constraints(n: Network, component: str, attr: str) -> None:
    """Define constraints for fixing component capacities to specified values.

    Sets constraints to fix nominal (capacity) variables of components to values
    specified in the corresponding '_set' attribute.

    Applies to Generator (p_nom), Line (s_nom), Transformer (s_nom), Link (p_nom),
    Store (e_nom), StorageUnit (p_nom).

    Parameters
    ----------
    n : pypsa.Network
        Network instance containing the model and component data
    component : str
        Name of the network component (e.g. "Generator", "StorageUnit")
    attr : str
        Name of the capacity attribute (e.g. "p_nom" for nominal power)

    Notes
    -----
    The function only creates constraints for components that have non-NaN
    values in their '{attr}_set' attribute.

    """
    c = as_components(n, component)
    if attr + "_set" not in c.static:
        return

    fix = c.static[attr + "_set"].dropna()

    if fix.empty:
        return

    dim = f"{component}-{attr}_set_i"
    fix = fix.rename_axis(dim)

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

    Applies to Generator (p_nom), Line (s_nom), Transformer (s_nom), Link (p_nom),
    Store (e_nom), StorageUnit (p_nom).

    Parameters
    ----------
    n : pypsa.Network
        Network instance containing the model and component data
    component : str
        Name of the network component (e.g. "Generator", "StorageUnit")
    attr : str
        Name of the capacity attribute (e.g. "p_nom" for nominal power)

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

    # Get components that are both extendable and modular
    mod_i = c.extendables.intersection(c.modulars)

    # Unique component names for modular components (in absence of c.modulars helper)
    if isinstance(mod_i, pd.MultiIndex):
        mod_i = mod_i.unique(level="name")

    if mod_i.empty:
        return

    # Get modular capacity values
    mod_attr = c._operational_attrs["nom_mod"]
    modular_capacity = c.da[mod_attr].sel(name=mod_i)

    # Get variables
    modularity = m[f"{c.name}-n_mod"]
    capacity = m.variables[f"{c.name}-{attr}"].loc[mod_i]

    con = capacity - modularity * modular_capacity == 0
    n.model.add_constraints(con, name=f"{c.name}-{attr}_modularity", mask=None)


def define_committability_variables_constraints_with_fixed_upper_limit(
    n: Network, sns: pd.Index, component: str, attr: str
) -> None:
    """Define upper limit constraints for committable unit status variables with fixed limits.

    This function sets the upper limit of committable variables (status, start-up, shut-down)
    for components with fixed upper limits. The upper limit corresponds to:

    a) The installed number of modules for committable, non-extendable components with modularity.
       The number of modules is calculated as: nominal_capacity / module_size (e.g., p_nom / p_nom_mod)

    b) The value 1 for all committable components without modularity, regardless of whether
       they are extendable or not.

    For case a), if the number of modules is not an integer, the function raises a ValueError.

    Parameters
    ----------
    n : pypsa.Network
        Network instance containing the model and component data
    sns : pd.Index
        Set of snapshots for which to define the constraints
    component : str
        Name of the network component (e.g. "Generator", "Link")
    attr : str
        Name of the capacity attribute (e.g. "p_nom" for nominal power)

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If non-extendable modular components have nominal capacity that is not
        an integer multiple of their module size

    Notes
    -----
    This function is part of the modular unit commitment formulation that allows
    committable components to have discrete capacity expansion.

    """
    c = as_components(n, component)
    m = n.model

    # Get committable, modular, and non-extendable component indices
    com_i = c.committables
    mod_i = c.modulars
    fix_i = c.fixed

    if com_i.empty:
        return

    inter_i = com_i.intersection(mod_i).intersection(fix_i)

    if not inter_i.empty:
        # Get nominal capacity and module size
        nom_attr = c._operational_attrs["nom"]
        mod_attr = c._operational_attrs["nom_mod"]

        nom_values = c.static[nom_attr].loc[inter_i]
        mod_values = c.static[mod_attr].loc[inter_i]

        n_mod = nom_values / mod_values
        diff_n_mod = abs(n_mod - round(n_mod))
        non_integers_n_mod_i = diff_n_mod[diff_n_mod > 10**-6].index

        if not non_integers_n_mod_i.empty:
            msg = (
                f"For non-extendable but committable assets, if both {nom_attr} and {mod_attr} are declared, "
                f"{nom_attr} must be a multiple of {mod_attr}. Found assets in component {component} "
                f"that do not respect this criterion:\n\n\t{', '.join(non_integers_n_mod_i)}"
            )
            raise ValueError(msg)

        rhs = pd.DataFrame(0, sns, inter_i)
        rhs.loc[sns, inter_i] = n_mod.loc[inter_i].values

    inter_i2 = com_i.difference(mod_i)

    if not inter_i2.empty:
        if not inter_i.empty:
            rhs = rhs.reindex(columns=rhs.columns.union(inter_i2))
            rhs.loc[:, inter_i2] = 1
            inter_i = inter_i.union(inter_i2)
        else:
            rhs = pd.DataFrame(0, sns, inter_i2)
            rhs.loc[sns, inter_i2] = 1
            inter_i = inter_i2

    if inter_i.empty:
        return

    active = c.da.active.sel(snapshot=sns, name=inter_i) if n._multi_invest else None

    status = m.variables[f"{component}-status"].loc[sns, inter_i]
    m.add_constraints(
        status, "<=", rhs, name=f"{component}-status-{attr}-fixed-upper", mask=active
    )

    start_up = m.variables[f"{component}-start_up"].loc[sns, inter_i]
    m.add_constraints(
        start_up,
        "<=",
        rhs,
        name=f"{component}-start_up-{attr}-fixed-upper",
        mask=active,
    )

    shut_down = m.variables[f"{component}-shut_down"].loc[sns, inter_i]
    m.add_constraints(
        shut_down,
        "<=",
        rhs,
        name=f"{component}-shut_down-{attr}-fixed-upper",
        mask=active,
    )


def define_committability_variables_constraints_with_variable_upper_limit(
    n: Network, sns: pd.Index, component: str, attr: str
) -> None:
    """Define upper limit constraints for committable unit status variables with variable limits.

    This function sets the upper limit of committable variables (status, start-up, shut-down)
    to the variable n_mod for all committable, extendable, and modular components.

    The constraint enforces that the number of committed units cannot exceed the total
    number of installed modules, which is itself a decision variable in the optimization.

    Parameters
    ----------
    n : pypsa.Network
        Network instance containing the model and component data
    sns : pd.Index
        Set of snapshots for which to define the constraints
    component : str
        Name of the network component (e.g. "Generator", "Link")
    attr : str
        Name of the capacity attribute (e.g. "p_nom" for nominal power)

    Notes
    -----
    This function is part of the modular unit commitment formulation that allows
    committable and extendable components to have discrete capacity expansion.
    The number of committed units is constrained by the optimized number of modules.

    """
    c = as_components(n, component)
    m = n.model

    # Get committable, extendable, and modular component indices
    com_i = c.committables
    ext_i = c.extendables
    mod_i = c.modulars

    inter_i = com_i.intersection(mod_i).intersection(ext_i)

    if inter_i.empty:
        return

    active = c.da.active.sel(snapshot=sns, name=inter_i) if n._multi_invest else None

    n_mod = m[f"{component}-n_mod"].loc[inter_i]

    status = m.variables[f"{component}-status"].loc[sns, inter_i]
    lhs = ((1, status), (-1, n_mod))
    m.add_constraints(
        lhs, "<=", 0, name=f"{component}-status-{attr}-variable-upper", mask=active
    )

    start_up = m.variables[f"{component}-start_up"].loc[sns, inter_i]
    lhs = ((1, start_up), (-1, n_mod))
    m.add_constraints(
        lhs, "<=", 0, name=f"{component}-start_up-{attr}-variable-upper", mask=active
    )

    shut_down = m.variables[f"{component}-shut_down"].loc[sns, inter_i]
    lhs = ((1, shut_down), (-1, n_mod))
    m.add_constraints(
        lhs, "<=", 0, name=f"{component}-shut_down-{attr}-variable-upper", mask=active
    )


def define_fixed_operation_constraints(
    n: Network, sns: pd.Index, component: str, attr: str
) -> None:
    """Define constraints for fixing operational variables to specified values.

    Sets constraints to fix dispatch variables of components to values specified
    in the corresponding '_set' attribute.

    Applies to Generator (p), Line (s), Transformer (s), Link (p), Store (e),
    StorageUnit (p_dispatch, p_store, state_of_charge).

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

    Notes
    -----
    This function is useful for modeling must-run generators, fixed imports/exports,
    or pre-committed dispatch decisions.

    The function only creates constraints for snapshots and components where
    the '{attr}_set' values are not NaN and the component is active.

    For StorageUnit components, if `p_set` is specified (via attr="p"), the
    constraint fixes the net power (p_dispatch - p_store) to the given values.
    Positive p_set means net discharge, negative means net charge.

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

    if component == "StorageUnit" and attr == "p":
        p_dispatch = n.model["StorageUnit-p_dispatch"]
        p_store = n.model["StorageUnit-p_store"]
        lhs = p_dispatch - p_store
        n.model.add_constraints(lhs, "=", fix, name="StorageUnit-p_set", mask=mask)
    else:
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

    Applies to StorageUnit (p_dispatch, p_store, state_of_charge, spill).

    Parameters
    ----------
    n : pypsa.Network
        Network instance containing the model and component data
    sns : pd.Index
        Set of snapshots for which to define the constraints

    Notes
    -----
    The function handles different storage operating modes:
    - Cyclic storage (returning to initial state at the end of the period)
    - Non-cyclic storage (with specified initial state of charge)

    For multi-investment period models, the function supports both cycling
    within each period and carrying state of charge between periods.

    Three key flags control the behavior:

    - **C** (cyclic_state_of_charge): If True, globally cycle state of charge
      from the last snapshot back to the first snapshot across all periods.
    - **CP** (cyclic_state_of_charge_per_period): If True, cycle state of charge
      within each investment period (last snapshot of period wraps to first).
    - **IP** (state_of_charge_initial_per_period): If True, reset to initial
      state_of_charge_initial value at the start of each period.

    When CP=True and IP=True simultaneously, CP takes precedence (wrapping behavior).

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
    eff_stand = (1 - c.da.standing_loss.sel(snapshot=sns, name=c.active_assets)) ** eh
    eff_dispatch = c.da.efficiency_dispatch.sel(snapshot=sns, name=c.active_assets)
    eff_store = c.da.efficiency_store.sel(snapshot=sns, name=c.active_assets)

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
    noncyclic_b = ~c.da.cyclic_state_of_charge.sel(name=c.active_assets)
    include_previous_soc = (active.cumsum(dim) != 1).where(noncyclic_b, True)

    previous_soc = (
        soc.where(active)
        .ffill(dim)
        .roll(snapshot=1)
        .ffill(dim)
        .where(include_previous_soc)
    )

    # We add inflow and initial soc for noncyclic assets to rhs
    soc_init = c.da.state_of_charge_initial.sel(name=c.active_assets)
    rhs = -c.da.inflow.sel(snapshot=sns, name=c.active_assets) * eh

    if n._multi_invest:
        # If multi-horizon optimizing, we update the previous_soc and the rhs
        # for all assets which are cyclic/non-cyclic per period
        periods = soc.coords["period"]
        # An asset is treated as per-period if:
        # 1. It cycles per period (CP=cyclic_state_of_charge_per_period=True), OR
        # 2. It uses initial state per period (IP=state_of_charge_initial_per_period=True)
        per_period = c.da.cyclic_state_of_charge_per_period.sel(
            name=c.active_assets
        ) | c.da.state_of_charge_initial_per_period.sel(name=c.active_assets)

        # We calculate the previous soc per period while cycling within a period
        # Normally, we should use groupby, but is broken for multi-index
        # see https://github.com/pydata/xarray/issues/6836
        ps = sns.unique("period")
        sl = slice(None)
        previous_soc_pp_list = [
            soc.data.sel(snapshot=(p, sl)).roll(snapshot=1) for p in ps
        ]
        previous_soc_pp = concat(previous_soc_pp_list, dim="snapshot")

        # We create a mask `include_previous_soc_pp` which determines when to include
        # previous state of charge from within the period:
        # - Always include previous for snapshots within a period (periods == periods.shift())
        # - At period boundaries (first snapshot):
        #   * If CP=True AND IP=False: cycle to last snapshot of period (wrap)
        #   * If IP=True: use initial value instead (no wrap, handled via rhs)
        #   * If CP=True AND IP=True: CP takes precedence, wrap (IP ignored)
        include_previous_soc_pp = active & (
            (periods == periods.shift(snapshot=1))
            | c.da.cyclic_state_of_charge_per_period.sel(name=c.active_assets)
        )

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

    # Warn if cyclic overrides initial values (both global and per-period)
    has_initial = c.da.state_of_charge_initial.sel(name=c.active_assets) != 0
    global_conflict = (
        c.da.cyclic_state_of_charge.sel(name=c.active_assets) & has_initial
    )
    period_conflict = (
        (
            c.da.cyclic_state_of_charge_per_period.sel(name=c.active_assets)
            & c.da.state_of_charge_initial_per_period.sel(name=c.active_assets)
            & has_initial
        )
        if n._multi_invest
        else False
    )

    ignored = global_conflict | period_conflict
    if ignored.any():
        affected = c.active_assets[ignored.values].tolist()
        logger.warning(
            "StorageUnits %s: Cyclic state of charge constraint overrules initial storage level setting. "
            "User-defined state_of_charge_initial will be ignored.",
            affected,
        )

    # Warn if per-period cyclic overrides global cyclic
    if n._multi_invest:
        cp_overrides_c = c.da.cyclic_state_of_charge.sel(
            name=c.active_assets
        ) & c.da.cyclic_state_of_charge_per_period.sel(name=c.active_assets)
        if cp_overrides_c.any():
            affected = c.active_assets[cp_overrides_c.values].tolist()
            logger.warning(
                "StorageUnits %s: Per-period cyclic (cyclic_state_of_charge_per_period=True) "
                "overrides global cyclic (cyclic_state_of_charge=True). "
                "Storage will cycle within each investment period, not across the entire horizon.",
                affected,
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

    Applies to Store (e, p).

    Parameters
    ----------
    n : pypsa.Network
        Network instance containing the model and component data
    sns : pd.Index
        Set of snapshots for which to define the constraints

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

    Three key flags control the behavior:

    - **C** (e_cyclic): If True, globally cycle energy level
      from the last snapshot back to the first snapshot across all periods.
    - **CP** (e_cyclic_per_period): If True, cycle energy level
      within each investment period (last snapshot of period wraps to first).
    - **IP** (e_initial_per_period): If True, reset to initial
      e_initial value at the start of each period.

    When CP=True and IP=True simultaneously, CP takes precedence (wrapping behavior).

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

    if n._multi_invest:
        # If multi-horizon optimization, we update previous_e and the rhs
        # for all assets which are cyclic/non-cyclic per period
        periods = e.coords["period"]
        # An asset is treated as per-period if:
        # 1. It cycles per period (CP=e_cyclic_per_period=True), OR
        # 2. It uses initial energy per period (IP=e_initial_per_period=True)
        per_period = c.da.e_cyclic_per_period | c.da.e_initial_per_period
        per_period = per_period.sel(name=c.active_assets)

        # We calculate the previous e per period while cycling within a period
        # Normally, we should use groupby, but it's broken for multi-index
        # see https://github.com/pydata/xarray/issues/6836
        ps = sns.unique("period")
        sl = slice(None)
        previous_e_pp_list = [e.data.sel(snapshot=(p, sl)).roll(snapshot=1) for p in ps]
        previous_e_pp = concat(previous_e_pp_list, dim="snapshot")

        # We create a mask `include_previous_e_pp` which determines when to include
        # previous energy from within the period:
        # - Always include previous for snapshots within a period (periods == periods.shift())
        # - At period boundaries (first snapshot):
        #   * If CP=True AND IP=False: cycle to last snapshot of period (wrap)
        #   * If IP=True: use initial value instead (no wrap, handled via rhs)
        #   * If CP=True AND IP=True: CP takes precedence, wrap (IP ignored)
        include_previous_e_pp = active & (
            (periods == periods.shift(snapshot=1))
            | c.da.e_cyclic_per_period.sel(name=c.active_assets)
        )

        # We take values still to handle internal xarray multi-index difficulties
        previous_e_pp = previous_e_pp.where(
            include_previous_e_pp.values, linopy.variables.FILL_VALUE
        )

        # update previous_e variables and rhs
        previous_e = previous_e.where(~per_period, previous_e_pp)
        include_previous_e = include_previous_e_pp.where(per_period, include_previous_e)

    # Warn if cyclic overrides initial values (both global and per-period)
    has_initial = c.da.e_initial.sel(name=c.active_assets) != 0
    global_conflict = c.da.e_cyclic.sel(name=c.active_assets) & has_initial
    period_conflict = (
        (
            c.da.e_cyclic_per_period.sel(name=c.active_assets)
            & c.da.e_initial_per_period.sel(name=c.active_assets)
            & has_initial
        )
        if n._multi_invest
        else False
    )

    ignored = global_conflict | period_conflict
    if ignored.any():
        affected = c.active_assets[ignored.values].tolist()
        logger.warning(
            "Stores %s: Cyclic energy level constraint overrules initial value setting. "
            "User-defined e_initial will be ignored.",
            affected,
        )

    # Warn if per-period cyclic overrides global cyclic
    if n._multi_invest:
        cp_overrides_c = c.da.e_cyclic.sel(
            name=c.active_assets
        ) & c.da.e_cyclic_per_period.sel(name=c.active_assets)
        if cp_overrides_c.any():
            affected = c.active_assets[cp_overrides_c.values].tolist()
            logger.warning(
                "Stores %s: Per-period cyclic (e_cyclic_per_period=True) "
                "overrides global cyclic (e_cyclic=True). "
                "Storage will cycle within each investment period, not across the entire horizon.",
                affected,
            )

    # Add the previous energy term with standing efficiency factor
    lhs += [(eff_stand, previous_e)]

    # For snapshots where we don't include previous_e, we need to account for initial values
    rhs = -e_init.where(~include_previous_e, 0)

    m.add_constraints(lhs, "=", rhs, name=f"{component}-energy_balance", mask=active)


def define_tangent_loss_constraints(
    n: Network,
    sns: pd.Index,
    component: str,
    segments: int,
) -> None:
    """Approximate transmission losses using piecewise linear tangents.

    Applies to Line and Transformer (passive branch components when transmission_losses
    are used).

    The tangent-based approximation underestimates losses. See equations
    (39)-(46) in [1] for details.

    Called via ``n.optimize(transmission_losses={"mode": "tangents", "segments": N})``.

    Parameters
    ----------
    n : pypsa.Network
        Network instance containing the model and branch data
    sns : pd.Index
        Set of snapshots for which to define the constraints
    component : str
        Name of the passive branch component (e.g. "Line", "Transformer")
    segments : int
        Number of tangent segments to use in the piecewise linearization
        of the quadratic loss function. Higher values increase accuracy
        but also computational complexity.

    References
    ----------
    [1] F. Neumann, T. Brown, "Transmission losses in power system
        optimization models: A comparison of heuristic and exact solution
        methods," Applied Energy, 2022,
        https://doi.org/10.1016/j.apenergy.2022.118859

    """
    if not isinstance(segments, int) or segments < 1:
        msg = f"'segments' must be a positive integer, got {segments!r}"
        raise ValueError(msg)

    c = n.components[component]

    if c.static.empty or component not in n.passive_branch_components:
        return

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
    for k in range(1, segments + 1):
        # Calculate linearization parameters for segment k
        p_k = k / segments * s_max_pu * s_nom_max
        loss_k = r_pu_eff * p_k**2
        slope_k = 2 * r_pu_eff * p_k
        offset_k = loss_k - slope_k * p_k

        # Add constraints for both positive and negative flow
        for sign in [-1, 1]:
            lhs = n.model.linexpr((1, loss), (sign * slope_k, flow))
            n.model.add_constraints(
                lhs >= offset_k,
                name=f"{c.name}-loss_tangents-{k}-{sign}",
                mask=active,
            )


def define_secant_loss_constraints(
    n: Network,
    sns: pd.Index,
    component: str,
    atol: float = 1,
    rtol: float = 0.1,
    max_segments: int = 20,
) -> None:
    """Approximate transmission losses using piecewise linear secants.

    Applies to Line and Transformer (passive branch components when transmission_losses
    are used).

    Creates secant constraints to the quadratic loss curve ``L(p) = r * p^2``
    for passive branches. The secant-based approximation overestimates losses
    and is independent of ``s_nom_max``. The number of segments is determined
    automatically from the error tolerances. See [1] for details.

    Called via ``n.optimize(transmission_losses=True)`` or
    ``n.optimize(transmission_losses={"mode": "secants", ...})``.

    Parameters
    ----------
    n : pypsa.Network
        Network instance containing the model and branch data
    sns : pd.Index
        Set of snapshots for which to define the constraints
    component : str
        Name of the passive branch component (e.g. "Line", "Transformer")
    atol : float, default 1
        Absolute error tolerance between the quadratic loss curve and its
        piecewise linear approximation; controls segment density for small
        flows
    rtol : float, default 0.1
        Relative error tolerance; controls segment density for large flows
        where ``atol`` alone would produce too many segments
    max_segments : int, default 20
        Safety cap on the number of segments per direction. The total number
        of constraints may be at most ``2 * max_segments`` per branch.

    References
    ----------
    [1] https://github.com/PyPSA/PyPSA/pull/1495

    """
    if atol <= 0:
        msg = f"'atol' must be positive, got {atol}"
        raise ValueError(msg)
    if rtol < 0:
        msg = f"'rtol' must be non-negative, got {rtol}"
        raise ValueError(msg)
    if max_segments < 1:
        msg = f"'max_segments' must be >= 1, got {max_segments}"
        raise ValueError(msg)

    c = n.components[component]

    if c.static.empty or component not in n.passive_branch_components:
        return

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

    lossy = r_pu_eff > 0  # only for lines with losses
    target = (s_nom_max * s_max_pu).where(lossy, 0)

    # Step-by-step construct the breakpoints for the piecewise linear approximation
    # The first breakpoint p_0 is always at zero
    # The first step is always determined by atol, since the rtol step would be zero at p=0
    p_1 = where(lossy, 2 * sqrt(atol / r_pu_eff), 0)

    # Instead of building the full list of breakpoints, we just build the factors relative to p_1
    # This will allow for some algebraic simplifications later on
    breakpoint_factors_list: list[float] = [0.0, 1.0]  # factors for p_0 and p_1

    target_factors = where(
        lossy, target.max("snapshot") / p_1, 0
    )  # amounts to scaling p_1 to s_nom_max * s_max_pu

    while (breakpoint_factors_list[-1] < target_factors).any():
        k = len(breakpoint_factors_list)
        stepfactor_atol = k / (k - 1)
        stepfactor_rtol = 1 + 2 * (rtol + sqrt(rtol + rtol**2))
        stepfactor_k = maximum(stepfactor_atol, stepfactor_rtol)
        breakpoint_factors_list.append(breakpoint_factors_list[-1] * stepfactor_k)
        if k >= max_segments:
            msg = f"Secant loop hit max_segments; check atol/rtol or line parameters; current inputs would result in {2 * max_segments} additional constraints per line"
            raise RuntimeError(msg)

    # make a separate array of factors for every line
    factors_1d = DataArray(breakpoint_factors_list, dims=["secant"])
    breakpoint_factors = DataArray(
        tile(factors_1d.values[:, None], (1, p_1.sizes["name"])),
        dims=["secant", "name"],
        coords={"secant": factors_1d["secant"], "name": p_1["name"]},
    )
    # zero out factors for branches without losses
    breakpoint_factors = breakpoint_factors.where(lossy, 0)

    # Call the intersection points of a secant with the loss curve a and b, then we have:
    a_factors = breakpoint_factors.isel(secant=slice(None, -1))  # k segments: 0..K-1
    b_factors = breakpoint_factors.isel(secant=slice(1, None))  # k segments: 1..K
    b_factors["secant"] = b_factors["secant"] - 1  # align indices

    # The simplest form of the slope would be:
    # slope = r_pu_eff * (a + b)
    # with a=x_i, b=x_{i+1} we have:
    # x_i = breakpoint_factors[i] * x_1
    # ... = breakpoint_factors[i] * 2 * sqrt(atol / r_pu_eff)
    # Therefore:
    # slope = r_pu_eff * 2 * sqrt(atol / r_pu_eff) * (a_factors + b_factors) = ...
    slope = 2 * sqrt(atol * r_pu_eff) * (a_factors + b_factors)
    # offset = -r_pu_eff * a * b = ...
    offset = -4 * atol * (a_factors * b_factors)

    # Add constraints for both positive and negative flow
    for sign, s in [(-1, "neg"), (1, "pos")]:
        lhs = n.model.linexpr((1, loss), (sign * slope, flow))
        n.model.add_constraints(
            lhs >= offset,
            name=f"{c.name}-loss_secants-{s}",
            mask=active,
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

    Applies to Generator (by default, but component parameter can be changed).

    Parameters
    ----------
    n : pypsa.Network
        Network instance containing the model and component data
    sns : Sequence
        Set of snapshots for which to define the constraints
    component : str, default "Generator"
        Name of the network component to apply the constraints to

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
