# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Define global constraints for optimisation problems with Linopy."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import pandas as pd
from linopy import merge

from pypsa.descriptors import nominal_attrs

if TYPE_CHECKING:
    from pypsa import Network
logger = logging.getLogger(__name__)


def define_nominal_constraints_per_bus_carrier(n: Network, sns: pd.Index) -> None:
    """Set an capacity expansion limit for assets of the same carrier at the same bus.

    The function searches for columns in the `buses` dataframe matching the pattern
    "nom_{min/max}_{carrier}". In case the constraint should only be defined for one
    investment period, the column name can be constructed according to
    "nom_{min/max}_{carrier}_{period}" where period must be in `n.investment_periods`.

    Parameters
    ----------
    n : pypsa.Network
        The network to apply constraints to.
    sns : list-like
        Set of snapshots to which the constraint should be applied.

    """
    cols = n.c.buses.static.columns[n.c.buses.static.columns.str.startswith("nom_")]
    if cols.empty:
        return

    m = n.model
    warnings.warn(
        "Nominal constraints per bus carrier are deprecated and will be removed in the future. "
        "Use global constraint of type 'define_tech_capacity_expansion_limit' instead."
        "Deprecated in PyPSA 1.0 and will be removed in PyPSA 2.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    buses = n.c.buses.static.index[n.c.buses.static[cols].notnull().any(axis=1)]
    if n.has_scenarios and not buses.empty:
        msg = "Nominal constraints per bus carrier are not implemented for stochastic networks."
        raise NotImplementedError(msg)

    multi_invest = isinstance(n.snapshots, pd.MultiIndex)
    periods = n.optimize._window.subset(sns).periods if multi_invest else pd.Index([])

    for col in cols:
        msg = (
            f"Bus column '{col}' has invalid specification and cannot be "
            "interpreted as constraint, must match the pattern "
            "`nom_{min/max}_{carrier}` or `nom_{min/max}_{carrier}_{period}`"
        )
        if col.startswith("nom_min_"):
            sign = ">="
        elif col.startswith("nom_max_"):
            sign = "<="
        else:
            logger.warning(msg)
            continue
        remainder = col[len("nom_max_") :]
        if remainder in n.c.carriers.static.index:
            carrier = remainder
            period = None
        elif multi_invest:
            carrier, period = remainder.rsplit("_", 1)
            period = int(period)
            if carrier not in n.c.carriers.static.index or period not in periods:
                logger.warning(msg)
                continue
        else:
            logger.warning(msg)
            continue

        lhs = []

        for c, attr in nominal_attrs.items():
            var = f"{c}-{attr}"
            static = n.c[c].static

            if c not in n.one_port_components or "carrier" not in static:
                continue

            ext_i = n.c[c].extendables.intersection(
                static.index[static.carrier == carrier]
            )
            ext_i = n.c[c].filter_by_active_assets(ext_i, period)

            if ext_i.empty:
                continue

            busmap = static.loc[ext_i, "bus"].rename(buses.name).to_xarray()
            expr = m[var].loc[ext_i].groupby(busmap).sum().reindex({buses.name: buses})
            lhs.append(expr)

        if not lhs:
            continue

        lhs = merge(lhs)
        rhs = n.c.buses.static.loc[buses, col]
        mask = rhs.notnull()
        n.model.add_constraints(
            lhs,
            sign,
            rhs,
            name=f"Bus-{col}",
            mask=mask,
            scaling=n.c.buses._scaling(buses),
        )
