# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Define optimisation variables from PyPSA networks with Linopy."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pypsa import Network

logger = logging.getLogger(__name__)


def define_operational_variables(
    n: Network, sns: Sequence, c_name: str, attr: str
) -> None:
    """Initialize variables for power dispatch for a given component and a given attribute.

    Parameters
    ----------
    n : pypsa.Network
        Network instance
    sns : Sequence
        Snapshots
    c_name : str
        name of the network component
    attr : str
        name of the attribute, e.g. 'p'

    """
    c = n.components[c_name]
    if c.empty:
        return

    active = c.da.active.sel(name=c.active_assets, snapshot=sns)
    coords = active.coords
    n.model.add_variables(coords=coords, name=f"{c.name}-{attr}", mask=active)


def define_status_variables(
    n: Network, sns: Sequence, c_name: str, is_linearized: bool = False
) -> None:
    """Initialize variables for unit status decisions.

    Uses integer variables if there are modular committables (status represents
    number of modules), otherwise uses binary variables for efficiency.
    Upper bounds are set by constraint functions.

    Parameters
    ----------
    n : pypsa.Network
        Network instance
    sns : Sequence
        Snapshots
    c_name : str
        name of the network component
    is_linearized : bool, default False
        Whether the unit commitment should be linearized

    """
    c = n.components[c_name]
    com_i = c.committables.difference(c.inactive_assets)

    if com_i.empty:
        return

    active = c.da.active.sel(name=com_i, snapshot=sns)
    coords = active.coords

    # Use integer variables if there are modular committables, binary otherwise
    has_modular = not com_i.intersection(c.modulars).empty
    is_integer = has_modular and not is_linearized
    is_binary = not has_modular and not is_linearized

    if has_modular:
        kwargs = {"lower": 0}  # Upper bound set by constraint
    elif is_linearized:
        kwargs = {"upper": 1, "lower": 0}  # Explicit bounds for LP relaxation
    else:
        kwargs = {}  # Binary variables handle bounds internally
    n.model.add_variables(
        coords=coords,
        name=f"{c.name}-status",
        mask=active,
        integer=is_integer,
        binary=is_binary,
        **kwargs,
    )


def define_start_up_variables(
    n: Network, sns: Sequence, c_name: str, is_linearized: bool = False
) -> None:
    """Initialize variables for unit start-up decisions.

    Uses integer variables if there are modular committables (start-up represents
    number of modules), otherwise uses binary variables for efficiency.
    Upper bounds are set by constraint functions.

    Parameters
    ----------
    n : pypsa.Network
        Network instance
    sns : Sequence
        Snapshots
    c_name : str
        name of the network component
    is_linearized : bool, default False
        Whether the unit commitment should be linearized

    """
    c = n.components[c_name]
    com_i = c.committables.difference(c.inactive_assets)

    if com_i.empty:
        return

    active = c.da.active.sel(name=com_i, snapshot=sns)
    coords = active.coords

    # Use integer variables if there are modular committables, binary otherwise
    has_modular = not com_i.intersection(c.modulars).empty
    is_integer = has_modular and not is_linearized
    is_binary = not has_modular and not is_linearized

    if has_modular:
        kwargs = {"lower": 0}
    elif is_linearized:
        kwargs = {"upper": 1, "lower": 0}
    else:
        kwargs = {}
    n.model.add_variables(
        coords=coords,
        name=f"{c.name}-start_up",
        mask=active,
        integer=is_integer,
        binary=is_binary,
        **kwargs,
    )


def define_shut_down_variables(
    n: Network, sns: Sequence, c_name: str, is_linearized: bool = False
) -> None:
    """Initialize variables for unit shut-down decisions.

    Uses integer variables if there are modular committables (shut-down represents
    number of modules), otherwise uses binary variables for efficiency.
    Upper bounds are set by constraint functions.

    Parameters
    ----------
    n : pypsa.Network
        Network instance
    sns : Sequence
        Snapshots
    c_name : str
        name of the network component
    is_linearized : bool, default False
        Whether the unit commitment should be linearized

    """
    c = n.components[c_name]
    com_i = c.committables.difference(c.inactive_assets)

    if com_i.empty:
        return

    active = c.da.active.sel(name=com_i, snapshot=sns)
    coords = active.coords

    # Use integer variables if there are modular committables, binary otherwise
    has_modular = not com_i.intersection(c.modulars).empty
    is_integer = has_modular and not is_linearized
    is_binary = not has_modular and not is_linearized

    if has_modular:
        kwargs = {"lower": 0}
    elif is_linearized:
        kwargs = {"upper": 1, "lower": 0}
    else:
        kwargs = {}
    n.model.add_variables(
        coords=coords,
        name=f"{c.name}-shut_down",
        mask=active,
        integer=is_integer,
        binary=is_binary,
        **kwargs,
    )


def define_nominal_variables(n: Network, c_name: str, attr: str) -> None:
    """Initialize variables for nominal capacities.

    Parameters
    ----------
    n : pypsa.Network
        Network instance
    c_name : str
        name of network component of which the nominal capacity should be defined
    attr : str
        name of the variable, e.g. 'p_nom'

    """
    c = n.components[c_name]
    ext_i = c.extendables.difference(c.inactive_assets)
    if ext_i.empty:
        return
    if isinstance(ext_i, pd.MultiIndex):
        ext_i = ext_i.unique(level="name")

    n.model.add_variables(coords=[ext_i], name=f"{c.name}-{attr}")


def define_modular_variables(n: Network, c_name: str, attr: str) -> None:
    """Initialize variables 'attr' for a given component c to allow a modular expansion of the attribute 'attr_nom'.

    It allows to define 'n_opt', the optimal number of installed modules.

    Parameters
    ----------
    n : pypsa.Network
        Network instance
    c_name : str
        name of network component of which the nominal capacity should be defined
    attr : str
        name of the variable to be handled attached to modular constraints, e.g. 'p_nom'

    """
    c = n.components[c_name]
    mod_i = c.extendables.intersection(c.modulars).difference(c.inactive_assets)

    if mod_i.empty:
        return

    n.model.add_variables(lower=0, coords=[mod_i], name=f"{c.name}-n_mod", integer=True)


def define_spillage_variables(n: Network, sns: Sequence) -> None:
    """Define the spillage variables for storage units."""
    c_name = "StorageUnit"
    c = n.components[c_name]

    if c.empty:
        return

    upper = c.da.inflow.sel(name=c.active_assets, snapshot=sns)
    if upper.size == 0 or (upper.max() <= 0).all():
        return

    active = c.da.active.sel(snapshot=sns, name=c.active_assets)

    active_aligned, upper_aligned = xr.align(active, upper, join="inner")
    active = active_aligned.where(upper_aligned > 0, False)

    n.model.add_variables(0, upper_aligned, name=f"{c.name}-spill", mask=active)


def define_loss_variables(n: Network, sns: Sequence, c_name: str) -> None:
    """Initialize variables for transmission losses.

    Parameters
    ----------
    n : pypsa.Network
        Network instance
    sns : Sequence
        Snapshots
    c_name : str
        name of the network component

    """
    c = n.components[c_name]
    if c.empty or c.name not in n.passive_branch_components:
        return

    active = c.da.active.sel(name=c.active_assets, snapshot=sns)
    coords = active.coords
    n.model.add_variables(0, coords=coords, name=f"{c.name}-loss", mask=active)


def define_cvar_variables(n: Network) -> None:
    """Define auxiliary variables used in the CVaR (Conditional Value-at-Risk) formulation.

    This helper adds three auxiliary variables to the model when
    stochastic optimisation with risk preference is enabled.

    * `CVaR-a` (per-scenario, non-negative): auxiliary excess loss variables `a_s`.
      They linearise the tail expectation: `a_s >= OPEX_s - theta`.
    * `CVaR-theta` (scalar): the Value-at-Risk (VaR) level `theta` at confidence `alpha`.
    * `CVaR` (scalar): the Conditional Value-at-Risk (Expected Shortfall) objective term.

    These variables are linked by constraints (added in the objective construction)
    to implement the linear CVaR formulation.

    Parameters
    ----------
    n : pypsa.Network
        Network instance

    """
    if n.has_scenarios and n.has_risk_preference is False:
        return

    # Per-scenario auxiliary variables a[s]
    scenarios = n.scenarios
    if scenarios is None or len(scenarios) == 0:
        return

    # Non-negative excess loss variables per scenario
    n.model.add_variables(lower=0, coords=[scenarios], name="CVaR-a")
    # Scalar theta (VaR) and CVaR
    n.model.add_variables(name="CVaR-theta")
    n.model.add_variables(name="CVaR")
