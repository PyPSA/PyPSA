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
import pandas.testing as pd_testing
from deprecation import deprecated
from packaging import version
from pandas.api.types import is_list_like

from pypsa.definitions.components import ComponentsStore
from pypsa.definitions.structures import Dict
from pypsa.version import __version_semver__

if TYPE_CHECKING:
    from pypsa import Network, Networks

logger = logging.getLogger(__name__)


class MethodHandlerWrapper:
    """
    Decorator to wrap a method with a handler class.

    This decorator wraps any method with a handler class that is used to
    process the method's return value. The handler class must be a callable with
    the same signature as the method to guarantee compatibility. It also must be
    initialized with the method as its first argument. If so, the API is only extended
    and not changed. The handler class can be used as a drop-in replacement for the
    method.

    Needs to be used as a callable decorator, i.e. with parentheses:
    >>> class MyHandlerClass:
    ...     def __init__(self, bound_method: Callable) -> None:
    ...         self.bound_method = bound_method
    ...     def __call__(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
    ...         return self.bound_method(*args, **kwargs)

    >>> @MethodHandlerWrapper(handler_class=MyHandlerClass)
    ... def my_method(self, *args, **kwargs):
    ...     pass

    """

    def __init__(
        self,
        func: Callable | None = None,
        *,
        handler_class: Any = None,
        inject_attrs: dict[str, str] | None = None,
    ) -> None:
        """
        Initialize the decorator.

        Parameters
        ----------
        func : Callable, optional
            The statistic method to wrap.
        handler_class : Type, optional
            The handler class to use for wrapping the method. It should be callable
            with same signature as the method to guarantee compatibility.
        inject_attrs : dict[str, str], optional
            A mapping of instance attributes to be passed to the handler class
            as keyword arguments. The keys are the names of the attributes to be
            passed, and the values are the names of the attributes in the handler
            class. If None, no attributes are passed. Pass only strings, not
            attributes of the instance.
        """
        self.func = func
        self.handler_class = handler_class
        self.inject_attrs = inject_attrs or {}

    def __call__(self, func: Callable | None = None) -> MethodHandlerWrapper:
        """Call the decorator with the function to wrap."""
        if func is not None:
            self.func = func
            return self
        return self

    def __get__(self, obj: Any, objtype: Any = None) -> Any:
        """Bind method to an object instance."""
        if obj is None:
            return self

        if self.func is None:
            raise TypeError("Method has not been set correctly in MethodHandlerWrapper")

        # Create a bound method wrapper
        bound_method = self.func.__get__(obj, objtype)

        # Prepare additional arguments from instance attributes, if any
        handler_kwargs = {}
        for key, value in self.inject_attrs.items():
            if hasattr(obj, value):
                handler_kwargs[key] = getattr(obj, value)
            else:
                msg = (
                    f"Attribute '{key}' not found in the object instance. "
                    f"Please ensure it is set before using the decorator."
                )
                raise AttributeError(msg)

        wrapper = self.handler_class(bound_method, **handler_kwargs)

        return wrapper


def network_method_wrapper(func: Callable) -> Callable:
    """
    Decorator that allows a Network method to be applied to Networks containers.

    This decorator checks if the first argument is a Networks container instead of a
    Network. If it is, it applies the function to each network in the container.

    Parameters
    ----------
    func : Callable
        Network method to wrap.

    Returns
    -------
    Callable
        Wrapped function that can handle both Network and Networks objects.

    Examples
    --------
    >>> @network_method_wrapper
    ... def my_network_function(n, other_arg):
    ...     # do something with network n
    ...     return result
    """

    @functools.wraps(func)
    def wrapper(n: Network | Networks, *args: Any, **kwargs: Any) -> Any:
        from pypsa.networks import Networks

        if isinstance(n, Networks):
            return n.apply(lambda net: func(net, *args, **kwargs))
        return func(n, *args, **kwargs)

    return wrapper


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


def equals(
    a: Any,
    b: Any,
    ignored_classes: Any = None,
    log_mode: str = "silent",
    path: str = "",
) -> bool:
    """
    Check if two objects are equal and track the location of differences.

    Parameters
    ----------
    a : Any
        First object to compare.
    b : Any
        Second object to compare.
    ignored_classes : Any, default=None
        Classes to ignore during comparison. If None, no classes are ignored.
    log_mode: str, default="silent"
        Controls how differences are reported:
        - 'silent': No logging, just returns True/False
        - 'verbose': Prints differences but doesn't raise errors
        - 'strict': Raises ValueError on first difference
    path: str, default=""
        Current path in the object structure (used for tracking differences).

    Raises
    ------
    ValueError
        If log_mode is 'strict' and components are not equal.

    Returns
    -------
    bool
        True if the objects are equal, False otherwise.
    """
    if not isinstance(log_mode, str):
        msg = "'log_mode' must be a string, not {type(log_mode)}."
    if log_mode not in ["silent", "verbose", "strict"]:
        msg = (
            f"'log_mode' must be one of 'silent', 'verbose', 'strict'], not {log_mode}."
        )
        raise ValueError(msg)

    current_path = path or "root"

    def handle_diff(message: str) -> bool:
        if log_mode == "strict":
            raise ValueError(message)
        elif log_mode == "verbose":
            logger.warning(message)
        return False

    if not isinstance(a, type(b)):
        # Ignore if they are subtypes # TODO: remove with data validation PR
        if np.issubdtype(type(a), type(b)) or np.issubdtype(type(b), type(a)):
            pass
        else:
            msg = f"Types differ at '{current_path}'\n\n{a} ({type(a)})\n\n!=\n\n{b} ({type(b)})\n"
            return handle_diff(msg)

    if ignored_classes is not None:
        if isinstance(a, tuple(ignored_classes)):
            return True

    # Classes with equality methods
    if isinstance(a, np.ndarray):
        if not np.array_equal(a, b, equal_nan=True):
            msg = f"numpy arrays differ at '{current_path}'\n\n{a}\n\n!=\n\n{b}\n"
            return handle_diff(msg)

    elif isinstance(a, (pd.DataFrame | pd.Series | pd.Index)):
        if a.empty and b.empty:
            return True
        if not a.equals(b):
            # TODO: Resolve with data validation PR
            # Check if dtypes are equal
            try:
                pd_testing.assert_frame_equal(a, b, check_dtype=False)
            except AssertionError:
                msg = f"pandas objects differ at '{current_path}'\n\n{a}\n\n!=\n\n{b}\n"
                return handle_diff(msg)
    # Custom classes
    elif isinstance(a, ComponentsStore):
        if a.keys() != b.keys():
            msg = (
                f"ComponentsStore keys differ at '{current_path}'\n\n{a}\n\n!=\n\n{b}\n"
            )
            return handle_diff(msg)
        for k in a.keys():
            if not equals(a[k], b[k], ignored_classes, log_mode, f"{current_path}.{k}"):
                return False
    # Iterators
    elif isinstance(a, (dict | Dict)):
        for k, v in a.items():
            if k not in b:
                msg = f"Key '{k}' missing from second dict at '{current_path}'"
                return handle_diff(msg)
            if not equals(v, b[k], ignored_classes, log_mode, f"{current_path}.{k}"):
                return False
        # Check for extra keys in b
        for k in b:
            if k not in a:
                msg = f"Key '{k}' missing from first dict at '{current_path}'"
                return handle_diff(msg)

    elif isinstance(a, (list | tuple)):
        if len(a) != len(b):
            msg = f"Collections have different lengths at '{current_path}': {len(a)} != {len(b)}"
            return handle_diff(msg)

        for i, v in enumerate(a):
            if not equals(v, b[i], ignored_classes, log_mode, f"{current_path}[{i}]"):
                return False

    # Nans
    elif pd.isna(a) and pd.isna(b):
        pass

    # Other objects
    else:
        if a != b:
            msg = f"Objects differ at '{current_path}'\n\n{a}\n\n!=\n\n{b}\n"
            return handle_diff(msg)

    return True


def rename_deprecated_kwargs(
    func_name: str,
    kwargs: dict[str, Any],
    aliases: dict[str, str],
    deprecated_in: str,
    removed_in: str,
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
    deprecated_in : str
        Version in which the argument was deprecated.
    removed_in : str
        Version in which the argument will be removed.
    """
    if version.parse(deprecated_in) > version.parse(__version_semver__):
        msg = (
            "'rename_deprecated_kwargs' can only be used in a version >= deprecated_in "
            f"(current version: {__version_semver__}, deprecated_in: {deprecated_in})."
        )
        raise ValueError(msg)

    for alias, new in aliases.items():
        if alias in kwargs:
            if new in kwargs:
                raise DeprecationWarning(
                    f"{func_name} received both {alias} and {new} as arguments!"
                    f" {alias} is deprecated, use {new} instead."
                )

            message = f"`{alias}` is deprecated as an argument to `{func_name}`; use `{new}` instead."
            if deprecated_in:
                message += f" Deprecated in version {deprecated_in}."
            if removed_in:
                message += f" Will be removed in version {removed_in}."

            warnings.warn(
                message=message,
                category=DeprecationWarning,
                stacklevel=3,
            )
            kwargs[new] = kwargs.pop(alias)


def deprecated_kwargs(deprecated_in: str, removed_in: str, **aliases: str) -> Callable:
    """
    Decorator for deprecated function and method arguments.
    Based on solution from [here](https://stackoverflow.com/questions/49802412).

    Parameters
    ----------
    deprecated_in : str
        Version in which the argument was deprecated.
    removed_in : str
        Version in which the argument will be removed.
    aliases : dict
        A mapping of old argument names to new argument names.

    Returns
    -------
    Callable
        A decorator that renames the old arguments to the new arguments.

    Examples
    --------
    >>> @deprecated_kwargs(deprecated_in="0.32.0", removed_in="1.0", object_id="id_object")
    ... def some_func(id_object):
    ...     print(id_object)
    >>> some_func(object_id=1) # doctest: +SKIP
    1
    """

    def deco(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            rename_deprecated_kwargs(
                f.__name__, kwargs, aliases, deprecated_in, removed_in
            )
            return f(*args, **kwargs)

        return wrapper

    return deco


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
    return deprecated_kwargs(network="n", deprecated_in="0.31", removed_in="1.0")(f)


def deprecated_in_next_major(details: str) -> Callable:
    """
    A wrapper for the @deprecated decorator that only requires specifying the details.

    Deprecates the function in the next major version and removes it in the
    following major version. Currently set to deprecate in version 1.0 and remove
    in version 2.0.

    Parameters
    ----------
    details : str
        Details about the deprecation.

    Returns
    -------
    Callable
        A decorator that marks the function as deprecated.

    """

    def decorator(func: Callable) -> Callable:
        return deprecated(
            deprecated_in="1.0",
            removed_in="2.0",
            current_version=__version_semver__,
            details=details,
        )(func)

    return decorator


def deprecated_namespace(
    func: Callable,
    previous_module: str,
    deprecated_in: str,
    removed_in: str,
) -> Callable:
    """
    Decorator for functions that have been moved from one namespace to another.

    Parameters
    ----------
    func : Callable
        The function that has been moved.
    previous_module : str
        The previous module path where the function was located.
    deprecated_in : str
        Version in which the namespace was deprecated.
    removed_in : str
        Version in which the namespace will be removed.

    Returns
    -------
    Callable
        A wrapper function that warns about the deprecated namespace.
    """
    if version.parse(deprecated_in) > version.parse(__version_semver__):
        msg = (
            "'deprecated_namespace' can only be used in a version >= deprecated_in "
            f"(current version: {__version_semver__}, deprecated_in: {deprecated_in})."
        )
        raise ValueError(msg)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Build the warning message with version information
        message = (
            f"`{previous_module}.{func.__name__}` is deprecated and will be removed in a future version. "
            f"Please use `{func.__module__}.{func.__name__}` instead."
        )

        if deprecated_in:
            message += f" Deprecated since version {deprecated_in}."
        if removed_in:
            message += f" Will be removed in version {removed_in}."

        warnings.warn(
            message,
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
    """
    Convert a variable to a pandas Series with the given index.

    Parameters
    ----------
    variable : dict | Sequence | float | int
        The variable to convert.
    index : pd.Index
        The index to use for the Series.

    Examples
    --------
    >>> _convert_to_series([1, 2, 3], pd.Index(['a', 'b', 'c']))
    a    1
    b    2
    c    3
    dtype: int64

    """
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
