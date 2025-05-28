"""Define optimisation variables from PyPSA networks with Linopy."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pypsa.descriptors import get_activity_mask
from pypsa.descriptors import get_switchable_as_dense as get_as_dense

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pypsa import Network

logger = logging.getLogger(__name__)


def define_operational_variables(n: Network, sns: Sequence, c: str, attr: str) -> None:
    """Initialize variables for power dispatch for a given component and a given
    attribute.

    Parameters
    ----------
    n : pypsa.Network
    c : str
        name of the network component
    attr : str
        name of the attribute, e.g. 'p'

    """
    if n.static(c).empty:
        return

    active = get_activity_mask(n, c, sns)
    coords = [sns, n.static(c).index.rename(c)]
    n.model.add_variables(coords=coords, name=f"{c}-{attr}", mask=active)


def define_status_variables(n: Network, sns: Sequence, c: str) -> None:
    com_i = n.get_committable_i(c)

    if com_i.empty:
        return

    active = get_activity_mask(n, c, sns, com_i)
    coords = (sns, com_i)
    is_binary = not n._linearized_uc
    kwargs = {"upper": 1, "lower": 0} if not is_binary else {}
    n.model.add_variables(
        coords=coords, name=f"{c}-status", mask=active, binary=is_binary, **kwargs
    )


def define_start_up_variables(n: Network, sns: Sequence, c: str) -> None:
    com_i = n.get_committable_i(c)

    if com_i.empty:
        return

    active = get_activity_mask(n, c, sns, com_i)
    coords = (sns, com_i)
    is_binary = not n._linearized_uc
    kwargs = {"upper": 1, "lower": 0} if not is_binary else {}
    n.model.add_variables(
        coords=coords, name=f"{c}-start_up", mask=active, binary=is_binary, **kwargs
    )


def define_shut_down_variables(n: Network, sns: Sequence, c: str) -> None:
    com_i = n.get_committable_i(c)

    if com_i.empty:
        return

    active = get_activity_mask(n, c, sns, com_i)
    coords = (sns, com_i)
    is_binary = not n._linearized_uc
    kwargs = {"upper": 1, "lower": 0} if not is_binary else {}
    n.model.add_variables(
        coords=coords, name=f"{c}-shut_down", binary=is_binary, **kwargs, mask=active
    )


def define_nominal_variables(n: Network, c: str, attr: str) -> None:
    """Initialize variables for nominal capacities for a given component and a
    given attribute.

    Parameters
    ----------
    n : pypsa.Network
    c : str
        network component of which the nominal capacity should be defined
    attr : str
        name of the variable, e.g. 'p_nom'

    """
    ext_i = n.get_extendable_i(c)
    if ext_i.empty:
        return

    n.model.add_variables(coords=[ext_i], name=f"{c}-{attr}")


def define_modular_variables(n: Network, c: str, attr: str) -> None:
    """Initialize variables 'attr' for a given component c to allow a modular
    expansion of the attribute 'attr_nom' It allows to define 'n_opt', the
    optimal number of installed modules.

    Parameters
    ----------
    n : pypsa.Network
    c : str
        network component of which the nominal capacity should be defined
    attr : str
        name of the variable to be handled attached to modular constraints, e.g. 'p_nom'

    """
    mod_i = n.static(c).query(f"{attr}_extendable and ({attr}_mod>0)").index
    mod_i = mod_i.rename(f"{c}-ext")

    if (mod_i).empty:
        return

    n.model.add_variables(lower=0, coords=[mod_i], name=f"{c}-n_mod", integer=True)


def define_spillage_variables(n: Network, sns: Sequence) -> None:
    """Define the spillage variables for storage units."""
    c = "StorageUnit"
    if n.static(c).empty:
        return

    upper = get_as_dense(n, c, "inflow", sns)
    if (upper.max() <= 0).all():
        return

    active = get_activity_mask(n, c, sns).where(upper > 0, False)
    n.model.add_variables(0, upper, name="StorageUnit-spill", mask=active)


def define_loss_variables(n: Network, sns: Sequence, c: str) -> None:
    """Initialize variables for transmission losses."""
    if n.static(c).empty or c not in n.passive_branch_components:
        return

    active = get_activity_mask(n, c, sns)
    coords = [sns, n.static(c).index.rename(c)]
    n.model.add_variables(0, coords=coords, name=f"{c}-loss", mask=active)
