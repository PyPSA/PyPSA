import functools
import warnings
from typing import Any, Callable

warnings.simplefilter("always", DeprecationWarning)


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

    def deco(f: Callable):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            rename_kwargs(f.__name__, kwargs, aliases)
            return f(*args, **kwargs)

        return wrapper

    return deco


def rename_kwargs(func_name: str, kwargs: dict[str, Any], aliases: dict[str, str]):
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
