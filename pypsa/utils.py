from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
from pandas.api.types import is_list_like

if TYPE_CHECKING:
    from pypsa import Network


def as_index(
    network: Network,
    values: Any,
    network_attribute: str,
    index_name: str | None,
) -> pd.Index:
    """
    Returns a pd.Index object from a list-like or scalar object.

    Also checks if the values are a subset of the corresponding attribute of the
    network object. If values is None, it is also used as the default.

    Parameters
    ----------
    network : pypsa.Network
        Network object from which to extract the default values.
    values : Any
        List-like or scalar object or None.
    network_attribute : str
        Name of the network attribute to be used as the default values. Only used if
        values is None.
    index_name : str, optional
        Name of the index. Will overwrite the name of the default attribute or passed
        values.

    Returns
    -------
    pd.Index: values as a pd.Index object.
    """

    if values is None:
        values_ = getattr(network, network_attribute)
    elif isinstance(values, (pd.Index, pd.MultiIndex)):
        values_ = values
    elif not is_list_like(values):
        values_ = pd.Index([values])
    else:
        values_ = pd.Index(values)
    values_.name = index_name

    assert values_.isin(getattr(network, network_attribute)).all()
    assert isinstance(values_, pd.Index)

    return values_
