"""
General utility functions for PyPSA.
"""

from __future__ import annotations

import functools
import logging
import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from deprecation import deprecated
from pandas.api.types import is_list_like

from pypsa.definitions.structures import Dict

if TYPE_CHECKING:
    from pypsa import Network

logger = logging.getLogger(__name__)


def as_index(
    n: Network, values: Any, network_attribute: str, force_subset: bool = True
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
    force_subset : bool, optional
        If True, the values must be a subset of the network attribute. Otherwise this
        is not checked, by default True.

    Returns
    -------
    pd.Index: values as a pd.Index object.
    """
    n_attr = getattr(n, network_attribute)

    if values is None:
        values_ = n_attr
    elif isinstance(values, pd.MultiIndex):
        values_ = values
        values_.names = n_attr.names
        values_.name = n_attr.name
    elif isinstance(values, pd.Index):
        values_ = values
        # If only timestep level is given for multiindex snapshots
        # TODO: This ambiguity should be resolved
        values_.names = n_attr.names[:1]
    elif not is_list_like(values):
        values_ = pd.Index([values], name=n_attr.names[0])
    else:
        values_ = pd.Index(values, name=n_attr.names[0])

    # if n_attr.nlevels != values_.nlevels:
    #     raise ValueError(
    #         f"Number of levels of the given MultiIndex does not match the number"
    #         f" of levels of the network attribute '{network_attribute}'. Please"
    #         f" set them for the network first."
    #     )

    if force_subset:
        if not values_.isin(n_attr).all():
            msg = (
                f"Values must be a subset of the network attribute "
                f"'{network_attribute}'. Pass force_subset=False to disable this check."
            )
            raise ValueError(msg)
    assert isinstance(values_, pd.Index)

    return values_


def equals(a: Any, b: Any, ignored_classes: Any = None) -> bool:
    assert isinstance(a, type(b)), f"Type mismatch: {type(a)} != {type(b)}"

    if ignored_classes is not None:
        if isinstance(a, tuple(ignored_classes)):
            return True

    # Classes with equality methods
    if isinstance(a, np.ndarray):
        if not np.array_equal(a, b):
            return False
    elif isinstance(a, (pd.DataFrame | pd.Series | pd.Index)):
        if not a.equals(b):
            return False
    # Iterators
    elif isinstance(a, (dict | Dict)):
        for k, v in a.items():
            if not equals(v, b[k]):
                return False
    elif isinstance(a, (list | tuple)):
        for i, v in enumerate(a):
            if not equals(v, b[i]):
                return False
    # Nans
    elif pd.isna(a) and pd.isna(b):
        pass
    else:
        if a != b:
            return False

    return True


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
    >>> @deprecated_kwargs(object_id="id_object")
    ... def some_func(id_object):
    ...     print(id_object)
    >>> some_func(object_id=1) # doctest: +SKIP
    1

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


def future_deprecation(*args: Any, activate: bool = False, **kwargs: Any) -> Callable:
    """
    Decorator factory which conditionally applies a deprecation warning.

    This way future deprecations can be marked without already raising the warning.
    """

    def custom_decorator(func: Callable) -> Callable:
        if activate:
            # Apply the deprecated decorator conditionally
            decorated_func = deprecated(*args, **kwargs)(func)
        else:
            decorated_func = func

        @functools.wraps(decorated_func)
        def wrapper(*func_args: Any, **func_kwargs: Any) -> Any:
            return decorated_func(*func_args, **func_kwargs)

        return wrapper

    return custom_decorator


def deprecated_namespace(func: Callable, previous_module: str) -> Callable:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        warnings.warn(
            f"The namespace `{previous_module}.{func.__name__}` is deprecated and will be removed in a future version. "
            f"Please use the new namespace `{func.__module__}.{func.__name__}` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    return wrapper


def list_as_string(
    list_: Sequence | dict, prefix: str = "", style: str = "comma-seperated"
) -> str:
    """
    Convert a list to a formatted string.

    Parameters
    ----------
    list_ : Sequence
        The input sequence to be converted.
    prefix : str, optional
        String to prepend to each line, by default "".
    style : {'same-line', 'bullet-list'}, optional
        Output format style, by default "same-line".

    Returns
    -------
    str
        Formatted string representation of the input sequence.

    Raises
    ------
    ValueError
        If an invalid style is provided.

    Examples
    --------
    >>> list_as_string(['a', 'b', 'c'])
    'a, b, c'
    """
    if isinstance(list_, dict):
        list_ = list(list_.keys())
    if len(list_) == 0:
        return ""

    if style == "comma-seperated":
        return prefix + ", ".join(list_)
    elif style == "bullet-list":
        return prefix + "- " + f"\n{prefix}- ".join(list_)
    else:
        raise ValueError(
            f"Style '{style}' not recognized. Use 'comma-seperated' or 'bullet-list'."
        )


def pass_none_if_keyerror(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except (KeyError, AttributeError):
            return None

    return wrapper


def pass_empty_series_if_keyerror(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> pd.Series:
        try:
            return func(*args, **kwargs)
        except (KeyError, AttributeError):
            return pd.Series([], dtype=float)

    return wrapper


def check_optional_dependency(module_name: str, install_message: str) -> None:
    """
    Check if an optional dependency is installed.

    If not, raise an ImportError with an install message.
    """
    try:
        __import__(module_name)
    except ImportError:
        raise ImportError(install_message)


def _convert_to_series(
    variable: dict | Sequence | float | int, index: pd.Index
) -> pd.Series:
    if isinstance(variable, dict):
        return pd.Series(variable)
    elif not isinstance(variable, pd.Series):
        return pd.Series(variable, index=index)
    return variable


def resample_timeseries(
    df: pd.DataFrame, freq: str, numeric_columns: list[str] | None = None
) -> pd.DataFrame:
    """
    Resample a DataFrame with proper handling of numeric and non-numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to resample, must have a datetime index
    freq : str
        Frequency string for resampling (e.g. 'H' for hourly)
    numeric_columns : list[str] | None
        List of numeric column names to resample. If None, auto-detected.

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame with numeric columns aggregated by mean
        and non-numeric columns forward-filled
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(pd.to_datetime(df.index))

    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns

    # Handle duplicate indices by aggregating first
    if not df.index.is_unique:
        numeric_df = df[numeric_columns].groupby(level=0).mean()
        non_numeric_df = df.drop(columns=numeric_columns).groupby(level=0).first()
        df = pd.concat([numeric_df, non_numeric_df], axis=1)[df.columns]

    # Split into numeric and non-numeric columns
    numeric_df = df[numeric_columns].resample(freq).mean()
    non_numeric_df = df.drop(columns=numeric_columns).resample(freq).ffill()

    # Combine the results
    return pd.concat([numeric_df, non_numeric_df], axis=1)[df.columns]


def check_pypsa_version(version_string: str) -> None:
    """
    Check if the installed PyPSA version was resolved correctly.
    """
    if version_string.startswith("0.0"):
        logger.warning(
            "The correct version of PyPSA could not be resolved. This is likely due to "
            "a local clone without pulling tags. Please run `git fetch --tags`."
        )
