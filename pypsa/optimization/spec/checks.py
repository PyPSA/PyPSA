# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Warnings and errors PyPSA raises on a network before its model is built."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from pypsa.components.common import invariant
from pypsa.descriptors import nominal_attrs
from pypsa.optimization.piecewise import get_piecewise_names

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    import xarray as xr

    from pypsa import Network
    from pypsa.optimization.piecewise import PiecewiseOptions

logger = logging.getLogger(__name__)

COMMITTABLE = ["Generator", "Link", "Process"]
OPERATIONAL_COSTS = (
    "marginal_cost",
    "marginal_cost_quadratic",
    "marginal_cost_dispatch",
    "marginal_cost_store",
    "marginal_cost_storage",
    "spill_cost",
)
COMMITMENT_COSTS = ("stand_by_cost", "start_up_cost", "shut_down_cost")
CYCLIC = {
    "StorageUnit": (
        "state_of_charge_initial",
        "cyclic_state_of_charge",
        "StorageUnits %s: Cyclic state of charge constraint overrules initial storage level setting. "
        "User-defined state_of_charge_initial will be ignored.",
    ),
    "Store": (
        "e_initial",
        "e_cyclic",
        "Stores %s: Cyclic energy level constraint overrules initial value setting. "
        "User-defined e_initial will be ignored.",
    ),
}


def _flagged(mask: xr.DataArray) -> list[str]:
    """Names of the assets for which `mask` holds in any scenario."""
    flagged = mask.any([d for d in mask.dims if d != "name"])
    return flagged.indexes["name"][flagged.values].tolist()


def _any_nonzero(da: xr.DataArray, names: pd.Index, sns: pd.Index) -> bool:
    """Whether `da` is non-zero for any of `names` in any of `sns`."""
    da = da.sel(name=names)
    if "snapshot" in da.dims:
        da = da.sel(snapshot=sns)
    return bool(da.size) and bool((da != 0).any())


def _check_invariants(n: Network) -> None:
    """Raise if data PyPSA reads as one value per asset differs across scenarios."""
    invariant(n.components.lines.static.carrier)
    for c in n.components[["Link", "Process"]]:
        for port in c._output_ports:
            suffix = c._port_suffix(port)
            for col in (f"delay{suffix}", f"cyclic_delay{suffix}"):
                if col in c.static:
                    invariant(c.static[col])


def _check_deprecations(n: Network) -> None:
    """Warn on `v_ang_min`."""
    for c in n.components[["Line", "Transformer"]]:
        if np.isfinite(c.static["v_ang_min"]).any():
            warnings.warn(
                f"The `v_ang_min` attribute of {c.name} is deprecated and "
                "ignored. `v_ang_max` applies symmetrically as an absolute "
                "voltage angle difference limit.",
                DeprecationWarning,
                stacklevel=4,
            )


def _check_modules(n: Network) -> None:
    """Raise if a fixed, modular committable's capacity is no multiple of its module."""
    for c in n.components[COMMITTABLE]:
        names = (
            c.committables.intersection(c.active_assets)
            .intersection(c.modulars)
            .intersection(c.fixed)
        )
        if names.empty:
            continue
        nom_attr = c._operational_attrs["nom"]
        mod_attr = c._operational_attrs["nom_mod"]
        n_mod = c.da[nom_attr].sel(name=names) / c.da[mod_attr].sel(name=names)
        bad = _flagged(abs(n_mod - n_mod.round()) > 10**-6)
        if bad:
            msg = (
                f"For non-extendable but committable assets, if both {nom_attr} and {mod_attr} are declared, "
                f"{nom_attr} must be a multiple of {mod_attr}. Found assets in component {c.name} "
                f"that do not respect this criterion:\n\n\t{', '.join(bad)}"
            )
            raise ValueError(msg)


def _check_tightening(n: Network) -> None:
    """Warn if the linearised unit commitment cannot be tightened for some units."""
    for c in n.components[COMMITTABLE]:
        names = c.committables.intersection(c.active_assets)
        if names.empty:
            continue
        equal = c.da.start_up_cost.sel(name=names) == c.da.shut_down_cost.sel(
            name=names
        )
        if not equal.all():
            logger.warning(
                "The linear relaxation of the unit commitment cannot be "
                "tightened for all generators since the start up costs "
                "are not equal to the shut down costs in all snapshots. "
                "Proceed with the linear relaxation without the tightening by "
                "additional constraints for these. This might result in a longer "
                "solving time."
            )


def _check_cyclic(n: Network, multi_invest: bool) -> None:
    """Warn where a cyclic storage level overrules an initial one or a global cycle."""
    for c in n.components[list(CYCLIC)]:
        if c.static.empty:
            continue
        initial, cyclic, message = CYCLIC[c.name]
        da = {
            attr: c.da[attr].sel(name=c.active_assets)
            for attr in (
                initial,
                cyclic,
                f"{cyclic}_per_period",
                f"{initial}_per_period",
            )
        }
        has_initial = da[initial] != 0
        ignored = da[cyclic] & has_initial
        if multi_invest:
            ignored = ignored | (
                da[f"{cyclic}_per_period"] & da[f"{initial}_per_period"] & has_initial
            )
        if ignored.any():
            logger.warning(message, _flagged(ignored))
        overridden = da[cyclic] & da[f"{cyclic}_per_period"]
        if multi_invest and overridden.any():
            logger.warning(
                "%ss %s: Per-period cyclic (%s_per_period=True) overrides global cyclic (%s=True). "
                "Storage will cycle within each investment period, not across the entire horizon.",
                c.name,
                _flagged(overridden),
                cyclic,
                cyclic,
            )


def _check_energy_limits(n: Network) -> None:
    """Raise if an energy floor is infinity or an energy cap minus infinity."""
    for c in n.components[list(nominal_attrs)]:
        if "e_sum_min" not in c.static:
            continue
        for attr, wrong in (("e_sum_min", np.inf), ("e_sum_max", -np.inf)):
            if (c.static[attr] == wrong).any():
                msg = f"Constraint {c.name}-{attr} contains incorrect infinite values."
                raise ValueError(msg)


def _check_piecewise(n: Network, options: Sequence[PiecewiseOptions]) -> None:
    """Raise if an `lp` piecewise option asks for an equality or meets a committable curve."""
    for option in options:
        c = n.components[option.component]
        if option.method != "lp" or not c.has_piecewise(option.attribute):
            continue
        if option.sign == "==":
            msg = "method='lp' requires exactly one tuple with sign='<=' or '>='."
            raise ValueError(msg)
        gated = get_piecewise_names(c, option.attribute, c.committables)
        if option.attribute != "capital_cost" and not gated.empty:
            msg = (
                f"method='lp' is not applicable to the committable {list(gated)}: "
                "a status gate is not supported by LP. Use method='incremental'."
            )
            raise ValueError(msg)


def _has_costs(n: Network, sns: pd.Index) -> bool:
    """Whether any active asset carries a cost the objective prices."""
    for c in n.components[list(nominal_attrs)]:
        active = c.active_assets
        for attr in OPERATIONAL_COSTS:
            if attr in c.static and (
                c.has_piecewise(attr) or _any_nonzero(c.da[attr], active, sns)
            ):
                return True
        if c.name in COMMITTABLE:
            committables = c.committables.intersection(active)
            if any(_any_nonzero(c.da[a], committables, sns) for a in COMMITMENT_COSTS):
                return True
        extendables = c.extendables.intersection(active)
        if not c._piecewise_schema("capital_cost").empty and not extendables.empty:
            return True
        if _any_nonzero(c.periodized_cost, extendables, sns):
            return True
    return False


def _check_objective(n: Network, sns: pd.Index) -> None:
    """Raise if no asset carries a cost, or a risk preference meets a quadratic cost."""
    if not _has_costs(n, sns):
        msg = (
            "Objective function could not be created. "
            "Please make sure the components have assigned costs."
        )
        raise ValueError(msg)
    quadratic = (
        _any_nonzero(c.da.marginal_cost_quadratic, c.active_assets, sns)
        for c in n.components[list(nominal_attrs)]
        if "marginal_cost_quadratic" in c.static
    )
    if n.has_risk_preference and any(quadratic):
        msg = (
            "CVaR with quadratic operational costs yields quadratic constraints. "
            "So a(s) >= OPEX(s) - theta becomes a quadratic inequality. "
            "Remove/approximate quadratic costs (e.g. set 'marginal_cost_quadratic=0' "
            "or use a piecewise-linear approximation)."
        )
        raise ValueError(msg)


CARRIED_OVER = (
    "Found non-cyclic {kind} with {what} and continuous depletion over multiple "
    "investment periods combined with investment period year weightings != 1. "
    "The {constraint} constraint will be inconsistent. Please consider setting "
    "`{per_period}` to True, using equal period weightings or a cyclic {one} instead."
)


def _check_carried_over_storage(n: Network, sns: pd.Index) -> None:
    """Refuse storage carried over periods of other than one year into an energy limit."""
    years = n.investment_period_weightings.years[sns.unique("period")]
    glcs = n.c.global_constraints.static
    if years.eq(1).all() or glcs.empty:
        return
    carriers = n.c.carriers.static
    storages = (
        (
            "storage units",
            "storage unit",
            n.c.storage_units,
            "cyclic_state_of_charge",
            "state_of_charge_initial_per_period",
        ),
        ("stores", "store", n.c.stores, "e_cyclic", "e_initial_per_period"),
    )
    for glc in glcs.itertuples():
        if glc.type == "primary_energy":
            found = carriers.index[carriers[glc.carrier_attribute] != 0]
            what, constraint = "associated carrier emissions", "primary energy"
        elif glc.type == "operational_limit":
            found = [glc.carrier_attribute]
            what, constraint = "the limited carrier", "operational"
        else:
            continue
        for kind, one, c, cyclic, per_period in storages:
            static = c.static[
                c.static.index.get_level_values("name").isin(c.active_assets)
            ]
            hit = static.carrier.isin(found) & ~static[cyclic] & ~static[per_period]
            if hit.any():
                msg = CARRIED_OVER.format(
                    kind=kind,
                    what=what,
                    constraint=constraint,
                    per_period=per_period,
                    one=one,
                )
                raise NotImplementedError(msg)


def check(
    n: Network,
    sns: pd.Index,
    *,
    multi_invest: bool = False,
    linearized_unit_commitment: bool = False,
    piecewise_options: Sequence[PiecewiseOptions] = (),
    **_: Any,
) -> None:
    """Warn and raise on network data as PyPSA's code-built model did, before the build."""
    _check_invariants(n)
    _check_deprecations(n)
    _check_modules(n)
    if linearized_unit_commitment:
        _check_tightening(n)
    _check_cyclic(n, multi_invest)
    _check_energy_limits(n)
    _check_piecewise(n, piecewise_options)
    _check_objective(n, sns)
    if multi_invest:
        _check_carried_over_storage(n, sns)
