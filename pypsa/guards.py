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
        if c.static.index.name != "name" if not c.has_scenarios else None:
            msg = f"Unexpected static index name for component '{c.name}': {c.static.index.name}"
            raise UnexpectedError(msg)
        # if not c.static.index.names == (
        #     ["name"] if not c.has_scenarios else ["scenario", "name"]
        # ):
        #     msg = f"Unexpected static index names for component '{c.name}': {c.static.index.names}"
        #     raise UnexpectedError(msg)
        # TODO n.sub_networks needs a fix
        # if not c.static.columns.name == None:
        #     msg = f"Unexpected static columns name for component '{c.name}': {c.static.columns.name}"
        #     raise UnexpectedError(msg)
        for attr_name, dynamic_df in c.dynamic.items():
            # if not dynamic_df.index.equals(c.snapshots):
            #     msg = f"`n.c.dynamic['{attr_name}']` of component '{c.name}' has index that does not match snapshots. Expected: {c.snapshots}, Found: {dynamic_df.index}"
            #     raise UnexpectedError(msg)
            # if not dynamic_df.columns.isin(c.static.index).all():
            #     msg = f"`n.c.dynamic['{attr_name}']` of component '{c.name}' has columns that are not in the static index. Found columns: {dynamic_df.columns[~dynamic_df.columns.isin(c.static.index)]}"
            #     raise UnexpectedError(msg)
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


@_guard_error_handler
def _network_index_data_verification(n: Network) -> None:
    """Assert that network index data is consistent and valid.

    Internal guard function - should only be called by other guard functions.

    Parameters
    ----------
    n : Network
        The PyPSA Network instance to validate.

    Raises
    ------
    UnexpectedError
        If any network index data is inconsistent or invalid.

    """
    # Verify snapshots index consistency
    if len(n.snapshots) == 0:
        msg = "Network snapshots must not be empty."
        raise UnexpectedError(msg)

    if n.snapshots.name != "snapshot":
        msg = f"Snapshots index must be named 'snapshot', found: {n.snapshots.name}"
        raise UnexpectedError(msg)

    # Verify MultiIndex structure for investment periods
    if isinstance(n.snapshots, pd.MultiIndex):
        if n.snapshots.nlevels != 2:
            msg = f"Snapshots MultiIndex must have exactly 2 levels, found: {n.snapshots.nlevels}"
            raise UnexpectedError(msg)

        expected_names = ["period", "timestep"]
        if list(n.snapshots.names) != expected_names:
            msg = f"Snapshots MultiIndex must have names {expected_names}, found: {list(n.snapshots.names)}"
            raise UnexpectedError(msg)

    required_weighting_cols = ["objective", "stores", "generators"]
    missing_cols = set(required_weighting_cols) - set(n._snapshots_data.columns)
    if missing_cols:
        msg = f"Snapshot weightings missing required columns: {missing_cols}"
        raise UnexpectedError(msg)

    # Verify investment period weightings if periods exist
    if n.has_periods:
        required_period_cols = ["objective", "years"]
        missing_period_cols = set(required_period_cols) - set(
            n._investment_periods_data.columns
        )
        if missing_period_cols:
            msg = f"Investment period weightings missing required columns: {missing_period_cols}"
            raise UnexpectedError(msg)

    # Verify scenarios consistency if they exist
    if n.has_scenarios:
        if n._scenarios_data.index.name != "scenario":
            msg = f"Scenarios index must be named 'scenario', found: {n._scenarios_data.index.name}"
            raise UnexpectedError(msg)

        if "weight" not in n._scenarios_data.columns:
            msg = "Scenarios data must have 'weight' column."
            raise UnexpectedError(msg)


# Guards to be used in runtime verification


@_guard_error_handler
def _as_xarray_guard(component: Any, res: xarray.DataArray) -> None:
    if component.has_scenarios and list(res.scenario.values) != list(
        component.scenarios
    ):
        msg = f"Scenario order mismatch: {list(res.scenario.values)} != {list(component.scenarios)}"
        raise UnexpectedError(msg)

    if list(res.coords["name"].values) != list(component.names):
        msg = f"Component order mismatch: {list(res.coords['name'].values)} != {list(component.names)}"
        raise UnexpectedError(msg)


@_guard_error_handler
def _consistency_check_guard(n: Network) -> None:
    _network_components_data_verification(n)
    _network_index_data_verification(n)


@_guard_error_handler
def _optimize_guard(n: Network) -> None:
    _network_components_data_verification(n)
    _network_index_data_verification(n)
