# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""General utility functions for PyPSA components."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from pypsa.components.components import Components
from pypsa.deprecations import COMPONENT_ALIAS_DICT

if TYPE_CHECKING:
    from pypsa.type_utils import NetworkType


def as_components(n: NetworkType, value: str | Components) -> Components:
    """Get component instance from string.

    <!-- md:badge-version v0.33.0 -->

    E.g. pass 'Generator', 'generators' or Components class instance to get the
    corresponding Components class instance.

    Parameters
    ----------
    value : str | Components
        String or Components class instance.
    n : pypsa.Network
        Network instance to which the components are attached.

    Returns
    -------
    Components
        Components class instance.

    Examples
    --------
    >>> # Get generators component from string
    >>> generators = pypsa.components.common.as_components(n, 'generators')
    >>> generators.name
    'Generator'

    >>> # Also works with singular form
    >>> gen = pypsa.components.common.as_components(n, 'Generator')
    >>> gen.name
    'Generator'

    """
    if isinstance(value, str):
        if value in COMPONENT_ALIAS_DICT:
            value = COMPONENT_ALIAS_DICT[value]
        return getattr(n.components, value)
    if isinstance(value, Components):
        if value.n is None:
            msg = "Passed component must be attached to the same network."
            raise ValueError(msg)
        if value.n is not n:
            msg = "Passed component is attached to a different network."
            raise ValueError(msg)
        return value
    msg = "Value must be a string or Components class instance."
    raise TypeError(msg)


def invariant(data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Index static data by name, requiring identical values across scenarios.

    Parameters
    ----------
    data : pd.Series | pd.DataFrame
        Static data, optionally with a `(scenario, name)` index.

    Returns
    -------
    pd.Series | pd.DataFrame
        `data` unchanged if it has no `scenario` level, otherwise reduced to
        the `name` index.

    Raises
    ------
    ValueError
        If values differ across scenarios.

    Examples
    --------
    >>> carrier = n_stochastic.c.generators.static.carrier
    >>> pypsa.components.common.invariant(carrier)
    name
    solar        solar
    wind          wind
    gas            gas
    lignite    lignite
    Name: carrier, dtype: object

    """
    if "scenario" not in data.index.names:
        return data
    varies = data.groupby(level="name").nunique(dropna=False).gt(1)
    where = ""
    if isinstance(varies, pd.DataFrame):
        where = f" in columns {varies.columns[varies.any()].tolist()}"
        varies = varies.any(axis=1)
    if varies.any():
        msg = (
            "Expected static data which is identical across scenarios, got "
            f"differing values for {varies.index[varies].tolist()}{where}."
        )
        raise ValueError(msg)
    return data.groupby(level="name", sort=False).first()
