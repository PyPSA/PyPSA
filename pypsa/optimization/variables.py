#!/usr/bin/env python3
"""
Define optimisation variables from PyPSA networks with Linopy.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pypsa import Network

logger = logging.getLogger(__name__)


def define_operational_variables(
    n: Network, sns: Sequence, c_name: str, attr: str
) -> None:
    """
    Initializes variables for power dispatch for a given component and a given
    attribute.

    Parameters
    ----------
    n : pypsa.Network
    c : str
        name of the network component
    attr : str
        name of the attribute, e.g. 'p'
    """
    c = n.components[c_name]
    if c.empty:
        return

    active = c.as_xarray("active", sns)
    coords = active.coords
    n.model.add_variables(coords=coords, name=f"{c.name}-{attr}", mask=active)


def define_status_variables(
    n: Network, sns: Sequence, c_name: str, is_linearized: bool = False
) -> None:
    c = n.components[c_name]
    com_i = c.get_committable_i()

    if com_i.empty:
        return

    active = c.as_xarray("active", sns, com_i)
    coords = active.coords
    is_binary = not is_linearized
    kwargs = dict(upper=1, lower=0) if not is_binary else {}
    n.model.add_variables(
        coords=coords, name=f"{c.name}-status", mask=active, binary=is_binary, **kwargs
    )


def define_start_up_variables(
    n: Network, sns: Sequence, c_name: str, is_linearized: bool = False
) -> None:
    """
    Initializes variables for unit start-up decisions.

    Parameters
    ----------
    n : pypsa.Network
    sns : Sequence
        Snapshots
    c_name : str
        name of the network component
    is_linearized : bool, default False
        Whether the unit commitment should be linearized
    """
    c = n.components[c_name]
    com_i = c.get_committable_i()

    if com_i.empty:
        return

    active = c.as_xarray("active", sns, com_i)
    coords = active.coords
    is_binary = not is_linearized
    kwargs = dict(upper=1, lower=0) if not is_binary else {}
    n.model.add_variables(
        coords=coords,
        name=f"{c.name}-start_up",
        mask=active,
        binary=is_binary,
        **kwargs,
    )


def define_shut_down_variables(
    n: Network, sns: Sequence, c_name: str, is_linearized: bool = False
) -> None:
    """
    Initializes variables for unit shut-down decisions.

    Parameters
    ----------
    n : pypsa.Network
    sns : Sequence
        Snapshots
    c_name : str
        name of the network component
    is_linearized : bool, default False
        Whether the unit commitment should be linearized
    """
    c = n.components[c_name]
    com_i = c.get_committable_i()

    if com_i.empty:
        return

    active = c.as_xarray("active", sns, com_i)
    coords = active.coords
    is_binary = not is_linearized
    kwargs = dict(upper=1, lower=0) if not is_binary else {}
    n.model.add_variables(
        coords=coords,
        name=f"{c.name}-shut_down",
        binary=is_binary,
        **kwargs,
        mask=active,
    )


def define_nominal_variables(n: Network, c_name: str, attr: str) -> None:
    """
    Initializes variables for nominal capacities for a given component and a
    given attribute.

    Parameters
    ----------
    n : pypsa.Network
    c_name : str
        name of network component of which the nominal capacity should be defined
    attr : str
        name of the variable, e.g. 'p_nom'
    """
    c = n.components[c_name]
    ext_i = c.get_extendable_i()
    if ext_i.empty:
        return

    n.model.add_variables(coords=[ext_i], name=f"{c.name}-{attr}")


def define_modular_variables(n: Network, c_name: str, attr: str) -> None:
    """
    Initializes variables 'attr' for a given component c to allow a modular
    expansion of the attribute 'attr_nom' It allows to define 'n_opt', the
    optimal number of installed modules.

    Parameters
    ----------
    n : pypsa.Network
    c_name : str
        name of network component of which the nominal capacity should be defined
    attr : str
        name of the variable to be handled attached to modular constraints, e.g. 'p_nom'
    """
    c = n.components[c_name]
    mod_i = c.static.query(f"{attr}_extendable and ({attr}_mod>0)").index
    mod_i = mod_i.rename(f"{c.name}-ext")

    if mod_i.empty:
        return

    n.model.add_variables(lower=0, coords=[mod_i], name=f"{c.name}-n_mod", integer=True)


def define_spillage_variables(n: Network, sns: Sequence) -> None:
    """
    Defines the spillage variables for storage units.
    """
    c_name = "StorageUnit"
    c = n.components[c_name]

    if c.empty:
        return

    upper = c.as_xarray("inflow", sns)
    if (upper.max() <= 0).all():
        return

    active = c.as_xarray("active", sns).where(upper > 0, False)
    n.model.add_variables(0, upper, name=f"{c.name}-spill", mask=active)


def define_loss_variables(n: Network, sns: Sequence, c_name: str) -> None:
    """
    Initializes variables for transmission losses.

    Parameters
    ----------
    n : pypsa.Network
    sns : Sequence
        Snapshots
    c_name : str
        name of the network component
    """
    c = n.components[c_name]
    if c.empty or c.name not in n.passive_branch_components:
        return

    active = c.as_xarray("active", sns)
    coords = active.coords
    n.model.add_variables(0, coords=coords, name=f"{c.name}-loss", mask=active)
