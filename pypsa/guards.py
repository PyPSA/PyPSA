"""Assertion guards for runtime verification of PyPSA.

Methods of this module should only be called when
pypsa.options.debug.runtime_verification is True. By default and in production,
this is False to avoid overhead. In development and testing, it can be enabled
to catch errors early.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable

from pypsa.common import UnexpectedError

if TYPE_CHECKING:
    from collections.abc import Callable

    import xarray

    from pypsa import Network


def _guard_error_handler(func: Callable) -> Callable:
    """Decorate guard functions to handle unexpected errors."""

    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> None:
        try:
            return func(*args, **kwargs)
        except UnexpectedError:
            raise  # Re-raise UnexpectedError as is
        except Exception as e:
            msg = f"Unexpected error in guard function {func.__name__}: {e}. "
            raise UnexpectedError(msg) from e

    return _wrapper


# Sub guards called by other guards


@_guard_error_handler
def _network_components_data_verification(n: Network) -> None:
    """Assert that for all components, dynamic attribute columns are subsets of static index.

    Internal guard function - should only be called by other guard functions.

    Parameters
    ----------
    n : Network
        The PyPSA Network instance to validate.

    Raises
    ------
    UnexpectedError
        If any dynamic attribute columns are not a subset of the static component index.

    """
    for c in n.components:
        for attr_name, dynamic_df in c.dynamic.items():
            if not dynamic_df.empty:
                # Check if all dynamic columns exist in static index
                if not isinstance(dynamic_df, pd.DataFrame):
                    msg = (
                        f"Dynamic attribute '{attr_name}' of component '{c.name}' "
                        f"is not a DataFrame. Found type: {type(dynamic_df)}."
                    )
                    raise UnexpectedError(msg)
                missing_columns = dynamic_df.columns.difference(c.static.index)
                if not missing_columns.empty:
                    msg = (
                        f"Dynamic attribute '{attr_name}' of component '{c.name}' "
                        f"has columns {list(missing_columns)} that are not in the static index. "
                        f"Static index: {list(c.static.index)}"
                    )


# Guards to be used in runtime verification


@_guard_error_handler
def _as_xarray_guard(component: Any, res: xarray.DataArray) -> None:
    if component.has_scenarios and list(res.scenario.values) != list(
        component.scenarios
    ):
        msg = f"Scenario order mismatch: {list(res.scenario.values)} != {list(component.scenarios)}"
        raise UnexpectedError(msg)

    if list(res.coords["name"].values) != list(component.component_names):
        msg = f"Component order mismatch: {list(res.coords['name'].values)} != {list(component.component_names)}"
        raise UnexpectedError(msg)


@_guard_error_handler
def _consistency_check_guard(n: Network) -> None:
    _network_components_data_verification(n)


@_guard_error_handler
def _optimize_guard(n: Network) -> None:
    _network_components_data_verification(n)
