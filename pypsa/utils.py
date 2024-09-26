from __future__ import annotations

import functools
import warnings
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd
from pandas.api.types import is_list_like

if TYPE_CHECKING:
    from pypsa import Network


def as_index(
    n: Network,
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
    n : pypsa.Network
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
        values_ = getattr(n, network_attribute)
    elif isinstance(values, (pd.Index, pd.MultiIndex)):
        values_ = values
    elif not is_list_like(values):
        values_ = pd.Index([values])
    else:
        values_ = pd.Index(values)
    values_.name = index_name

    assert values_.isin(getattr(n, network_attribute)).all()
    assert isinstance(values_, pd.Index)

    return values_


def deprecated_kwargs(**aliases: str) -> Callable:
    """
    Decorator for deprecated function and method arguments.
    Based on solution from [here](https://stackoverflow.com/questions/49802412).

    Parameters
    ----------
    aliases : dict
        A mapping of old argument names to new argument names.

    Returns
    -------
    Callable
        A decorator that renames the old arguments to the new arguments.

    Examples
    --------
    >>> @deprecated_alias(object_id="id_object")
    ... def __init__(self, id_object):
    """

    def deco(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            rename_kwargs(f.__name__, kwargs, aliases)
            return f(*args, **kwargs)

        return wrapper

    return deco


def rename_kwargs(
    func_name: str, kwargs: dict[str, Any], aliases: dict[str, str]
) -> None:
    """
    Helper function for deprecating function arguments.

    Based on solution from [here](https://stackoverflow.com/questions/49802412).

    Parameters
    ----------
    func_name : str
        The name of the function.
    kwargs : dict
        The keyword arguments of the function.
    aliases : dict
        A mapping of old argument names to new argument names.

    """

    for alias, new in aliases.items():
        if alias in kwargs:
            if new in kwargs:
                raise TypeError(
                    f"{func_name} received both {alias} and {new} as arguments!"
                    f" {alias} is deprecated, use {new} instead."
                )
            warnings.warn(
                message=(
                    f"`{alias}` is deprecated as an argument to `{func_name}`; use"
                    f" `{new}` instead."
                ),
                category=DeprecationWarning,
                stacklevel=3,
            )
            kwargs[new] = kwargs.pop(alias)


def deprecated_common_kwargs(f: Callable) -> Callable:
    """
    Decorator that predefines the 'a' keyword to be renamed to 'b'.
    This allows its usage as `@deprecated_ab` without parentheses.

    Parameters
    ----------
    f : Callable
        The function we are decorating.

    Returns
    -------
    Callable
        A decorated function that renames 'a' to 'b'.
    """
    return deprecated_kwargs(network="n")(f)
