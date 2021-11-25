#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 10:29:16 2021

@author: fabian
"""
import logging

from ..descriptors import get_activity_mask

logger = logging.getLogger(__name__)


def define_operational_variables(n, sns, c, attr):
    """
    Initializes variables for power dispatch for a given component and a
    given attribute.

    Parameters
    ----------
    n : pypsa.Network
    c : str
        name of the network component
    attr : str
        name of the attribute, e.g. 'p'

    """
    if n.df(c).empty:
        return

    active = get_activity_mask(n, c, sns) if n._multi_invest else None
    coords = [sns, n.df(c).index]
    n.model.add_variables(coords=coords, name=f"{c}-{attr}", mask=active)


def define_status_variables(n, sns, c):
    com_i = n.get_committable_i(c)

    if com_i.empty:
        return

    active = get_activity_mask(n, c, sns, com_i) if n._multi_invest else None
    coords = (sns, com_i)
    n.model.add_variables(coords=coords, name=f"{c}-status", mask=active, binary=True)


def define_nominal_variables(n, c, attr):
    """
    Initializes variables for nominal capacities for a given component and a
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
