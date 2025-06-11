"""Consistency check functions for PyPSA networks.

Mainly used in the `Network.consistency_check()` method.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from deprecation import deprecated

from pypsa.common import deprecated_common_kwargs
from pypsa.constants import RE_PORTS_FILTER
from pypsa.network.abstract import _NetworkABC

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pypsa import Network
    from pypsa.components import Components

logger = logging.getLogger(__name__)


class ConsistencyError(ValueError):
    """Error raised when a consistency check fails."""


def _bus_columns(df: pd.DataFrame) -> pd.Index:
    return df.columns[df.columns.str.contains(RE_PORTS_FILTER)]


def _log_or_raise(strict: bool, message: str, *args: Any) -> None:
    formatted_message = message % args if args else message
    if strict:
        raise ConsistencyError(formatted_message)
    logger.warning(message, *args)


@deprecated_common_kwargs
def check_for_unknown_buses(
    n: Network, component: Components, strict: bool = False
) -> None:
    """Check if buses are attached to component but are not defined in the network.

    Activate strict mode in general consistency check by passing `['unknown_buses']` to
    the `strict` argument.

    Parameters
    ----------
    n : pypsa.Network
        The network to check.
    component : pypsa.Component
        The component to check.
    strict : bool, optional
        If True, raise an error instead of logging a warning.

    See Also
    --------
    [pypsa.Network.consistency_check][] : General consistency check method, which runs
    all consistency checks.

    """
    for attr in _bus_columns(component.static):
        missing = ~component.static[attr].astype(str).isin(n.buses.index)
        # if bus2, bus3... contain empty strings do not warn
        if component.name in n.branch_components and int(attr[-1]) > 1:
            missing &= component.static[attr] != ""
        if missing.any():
            _log_or_raise(
                strict,
                "The following %s have buses which are not defined:\n%s",
                component.list_name,
                component.static.index[missing],
            )


@deprecated_common_kwargs
def check_for_disconnected_buses(n: Network, strict: bool = False) -> None:
    """Check if network has buses that are not connected to any component.

    Activate strict mode in general consistency check by passing `['disconnected_buses']`
    the `strict` argument.

    Parameters
    ----------
    n : pypsa.Network
        The network to check.
    strict : bool, optional
        If True, raise an error instead of logging a warning.

    See Also
    --------
    [pypsa.Network.consistency_check][] : General consistency check method, which runs
    all consistency checks.

    """
    connected_buses = set()
    for component in n.iterate_components():
        for attr in _bus_columns(component.static):
            connected_buses.update(component.static[attr])

    disconnected_buses = set(n.buses.index) - connected_buses
    if disconnected_buses:
        _log_or_raise(
            strict,
            "The following buses have no attached components, which can break the lopf: %s",
            disconnected_buses,
        )


@deprecated_common_kwargs
def check_for_unknown_carriers(
    n: Network, component: Components, strict: bool = False
) -> None:
    """Check if carriers are attached to component but are not defined in the network.

    Activate strict mode in general consistency check by passing `['unknown_carriers']`
    the `strict` argument.

    Parameters
    ----------
    n : pypsa.Network
        The network to check.
    component : pypsa.Component
        The component to check.
    strict : bool, optional
        If True, raise an error instead of logging a warning.

    See Also
    --------
    [pypsa.Network.consistency_check][] : General consistency check method, which runs
    all consistency checks.


    """
    if "carrier" in component.static.columns:
        missing = (
            ~component.static["carrier"].isin(n.carriers.index)
            & component.static["carrier"].notna()
            & (component.static["carrier"] != "")
        )
        if missing.any():
            _log_or_raise(
                strict,
                "The following %s have carriers which are not defined:\n%s",
                component.list_name,
                component.static.index[missing],
            )


@deprecated_common_kwargs
def check_for_zero_impedances(
    n: Network, component: Components, strict: bool = False
) -> None:
    """Check if component has zero impedances. Only checks passive branch components.

    Activate strict mode in general consistency check by passing `['zero_impedances']`
    the `strict` argument.

    Parameters
    ----------
    n : pypsa.Network
        The network to check.
    component : pypsa.Component
        The component to check.
    strict : bool, optional
        If True, raise an error instead of logging a warning.


    See Also
    --------
    [pypsa.Network.consistency_check][] : General consistency check method, which runs
    all consistency checks.


    """
    if component.name in n.passive_branch_components:
        for attr in ["x", "r"]:
            bad = component.static[attr] == 0
            if bad.any():
                _log_or_raise(
                    strict,
                    "The following %s have zero %s, which could break the linear load flow:\n%s",
                    component.list_name,
                    attr,
                    component.static.index[bad],
                )


@deprecated_common_kwargs
def check_for_zero_s_nom(component: Components, strict: bool = False) -> None:
    """Check if component has zero s_nom. Only checks transformers.

    Activate strict mode in general consistency check by passing `['zero_s_nom']` to
    the `strict` argument.

    Parameters
    ----------
    component : pypsa.Component
        The component to check.
    strict : bool, optional
        If True, raise an error instead of logging a warning.

    See Also
    --------
    [pypsa.Network.consistency_check][] : General consistency check method, which runs
    all consistency checks.

    """
    if component.name in {"Transformer"}:
        bad = component.static["s_nom"] == 0
        if bad.any():
            _log_or_raise(
                strict,
                "The following %s have zero s_nom, which is used to define the "
                "impedance and will thus break the load flow:\n%s",
                component.list_name,
                component.static.index[bad],
            )


@deprecated_common_kwargs
def check_time_series(n: Network, component: Components, strict: bool = False) -> None:
    """Check if time series of component are aligned with network snapshots.

    Activate strict mode in general consistency check by passing `['time_series']` to
    the `strict` argument.

    Parameters
    ----------
    n : pypsa.Network
        The network to check.
    component : pypsa.Component
        The component to check.
    strict : bool, optional
        If True, raise an error instead of logging a warning.

    See Also
    --------
    [pypsa.Network.consistency_check][] : General consistency check method, which runs
    all consistency checks.


    """
    for attr in component.attrs.index[component.attrs.varying & component.attrs.static]:
        attr_df = component.dynamic[attr]

        diff = attr_df.columns.difference(component.static.index)
        if len(diff):
            _log_or_raise(
                strict,
                "The following %s have time series defined for attribute %s in n.%s_t, "
                "but are not defined in n.%s:\n%s",
                component.list_name,
                attr,
                component.list_name,
                component.list_name,
                diff,
            )

        if not n.snapshots.equals(attr_df.index):
            _log_or_raise(
                strict,
                "The index of the time-dependent Dataframe for attribute %s of n.%s_t "
                "is not aligned with network snapshots",
                attr,
                component.list_name,
            )


@deprecated_common_kwargs
def check_static_power_attributes(
    n: Network, component: Components, strict: bool = False
) -> None:
    """Check static attrs p_now, s_nom, e_nom in any component.

    Activate strict mode in general consistency check by passing `['static_power_attrs']`
    the `strict` argument.

    Parameters
    ----------
    n : pypsa.Network
        The network to check.
    component : pypsa.Component
        The component to check.
    strict : bool, optional
        If True, raise an error instead of logging a warning.

    See Also
    --------
    [pypsa.Network.consistency_check][] : General consistency check method, which runs
    all consistency checks.


    """
    static_attrs = ["p_nom", "s_nom", "e_nom"]
    if component.name in n.all_components - {"TransformerType"}:
        static_attr = component.attrs.query("static").index.intersection(static_attrs)
        if len(static_attr):
            attr = static_attr[0]
            bad = component.static[attr + "_max"] < component.static[attr + "_min"]
            if bad.any():
                _log_or_raise(
                    strict,
                    "The following %s have smaller maximum than minimum expansion "
                    "limit which can lead to infeasibility:\n%s",
                    component.list_name,
                    component.static.index[bad],
                )

            attr = static_attr[0]
            for col in [attr + "_min", attr + "_max"]:
                if (
                    component.static[col][component.static[attr + "_extendable"]]
                    .isna()
                    .any()
                ):
                    _log_or_raise(
                        strict,
                        "Encountered nan's in column %s of component '%s'.",
                        col,
                        component.name,
                    )


@deprecated_common_kwargs
def check_time_series_power_attributes(
    n: Network, component: Components, strict: bool = False
) -> None:
    """Check `p_max_pu` and `e_max_pu` nan and infinite values in time series.

    Activate strict mode in general consistency check by passing `['time_series_power_attrs']`
    the `strict` argument.

    Parameters
    ----------
    n : pypsa.Network
        The network to check.
    component : pypsa.Component
        The component to check.
    strict : bool, optional
        If True, raise an error instead of logging a warning.

    See Also
    --------
    [pypsa.Network.consistency_check][] : General consistency check method, which runs
    all consistency checks.

    """
    varying_attrs = ["p_max_pu", "e_max_pu"]
    if component.name in n.all_components - {"TransformerType"}:
        varying_attr = component.attrs.query("varying").index.intersection(
            varying_attrs
        )

        if len(varying_attr):
            attr = varying_attr[0][0]
            max_pu = n.get_switchable_as_dense(component.name, attr + "_max_pu")
            min_pu = n.get_switchable_as_dense(component.name, attr + "_min_pu")

            # check for NaN values:
            if max_pu.isna().to_numpy().any():
                for col in max_pu.columns[max_pu.isna().any()]:
                    _log_or_raise(
                        strict,
                        "The attribute %s_max_pu of element %s of %s has NaN "
                        "values for the following snapshots:\n%s",
                        attr,
                        col,
                        component.list_name,
                        max_pu.index[max_pu[col].isna()],
                    )
            if min_pu.isna().to_numpy().any():
                for col in min_pu.columns[min_pu.isna().any()]:
                    _log_or_raise(
                        strict,
                        "The attribute %s_min_pu of element %s of %s has NaN "
                        "values for the following snapshots:\n%s",
                        attr,
                        col,
                        component.list_name,
                        min_pu.index[min_pu[col].isna()],
                    )

            # check for infinite values
            if np.isinf(max_pu).to_numpy().any():
                for col in max_pu.columns[np.isinf(max_pu).any()]:
                    _log_or_raise(
                        strict,
                        "The attribute %s_max_pu of element %s of %s has infinite"
                        " values for the following snapshots:\n%s",
                        attr,
                        col,
                        component.list_name,
                        max_pu.index[np.isinf(max_pu[col])],
                    )
            if np.isinf(min_pu).to_numpy().any():
                for col in min_pu.columns[np.isinf(min_pu).any()]:
                    _log_or_raise(
                        strict,
                        "The attribute %s_min_pu of element %s of %s has infinite"
                        " values for the following snapshots:\n%s",
                        attr,
                        col,
                        component.list_name,
                        min_pu.index[np.isinf(min_pu[col])],
                    )

            diff = max_pu - min_pu
            diff = diff[diff < 0].dropna(axis=1, how="all")
            for col in diff.columns:
                _log_or_raise(
                    strict,
                    "The element %s of %s has a smaller maximum than minimum operational"
                    " limit which can lead to infeasibility for the following snapshots:\n%s",
                    col,
                    component.list_name,
                    diff[col].dropna().index,
                )


@deprecated_common_kwargs
def check_assets(n: Network, component: Components, strict: bool = False) -> None:
    """Check if assets are only committable or extendable, but not both.

    Activate strict mode in general consistency check by passing `['assets']` to the
    `strict` argument.

    Parameters
    ----------
    n : pypsa.Network
        The network to check.
    component : pypsa.Component
        The component to check.
    strict : bool, optional
        If True, raise an error instead of logging a warning.

    See Also
    --------
    [pypsa.Network.consistency_check][] : General consistency check method, which runs
    all consistency checks.


    """
    if component.name in {"Generator", "Link"}:
        committables = n.get_committable_i(component.name)
        extendables = n.get_extendable_i(component.name)
        intersection = committables.intersection(extendables)
        if not intersection.empty:
            _log_or_raise(
                strict,
                "Assets can only be committable or extendable."
                " Found assets in component %s which are both:\n\n\t%s",
                component.name,
                ", ".join(intersection),
            )


@deprecated_common_kwargs
def check_generators(component: Components, strict: bool = False) -> None:
    """Check the consistency of generator attributes before the simulation.

    This function performs the following checks on generator components:
    1. Ensures that committable generators are not both up and down before the simulation.
    2. Verifies that the minimum total energy to be produced (e_sum_min) is not greater than the maximum total energy to be produced (e_sum_max).

    Activate strict mode in general consistency check by passing `['generators']` to the
    the `strict` argument.

    Parameters
    ----------
    component : Component
        The generator component to be checked.
    strict : bool, optional
        If True, raise an error instead of logging a warning.



    See Also
    --------
    [pypsa.Network.consistency_check][] : General consistency check method, which runs
    all consistency checks.


    """
    if component.name in {"Generator"}:
        bad_uc_gens = component.static.index[
            component.static.committable
            & (component.static.up_time_before > 0)
            & (component.static.down_time_before > 0)
        ]
        if not bad_uc_gens.empty:
            _log_or_raise(
                strict,
                "The following committable generators were both"
                " up and down before the simulation: %s. This could cause an infeasibility.",
                bad_uc_gens,
            )

        bad_e_sum_gens = component.static.index[
            component.static.e_sum_min > component.static.e_sum_max
        ]
        if not bad_e_sum_gens.empty:
            _log_or_raise(
                strict,
                "The following generators have e_sum_min > e_sum_max,"
                " which can lead to infeasibility:\n%s.",
                bad_e_sum_gens,
            )


@deprecated_common_kwargs
def check_dtypes_(component: Components, strict: bool = False) -> None:
    """Check if the dtypes of the attributes in the component are as expected.

    Activate strict mode in general consistency check by passing `['dtypes']` to the
    `strict` argument.

    Parameters
    ----------
    n : pypsa.Network
        The network to check.
    component : pypsa.Component
        The component to check.
    strict : bool, optional
        If True, raise an error instead of logging a warning.

    See Also
    --------
    [pypsa.Network.consistency_check][] : General consistency check method, which runs
    all consistency checks.


    """
    dtypes_soll = component.attrs.loc[component.attrs["static"], "dtype"].drop("name")
    unmatched = component.static.dtypes[dtypes_soll.index] != dtypes_soll

    if unmatched.any():
        _log_or_raise(
            strict,
            "The following attributes of the dataframe %s have"
            " the wrong dtype:\n%s\nThey are:\n%s\nbut should be:\n%s",
            component.list_name,
            unmatched.index[unmatched],
            component.static.dtypes[dtypes_soll.index[unmatched]],
            dtypes_soll[unmatched],
        )

    # now check varying attributes

    types_soll = component.attrs.loc[component.attrs["varying"], ["typ", "dtype"]]

    for attr, typ, dtype in types_soll.itertuples():
        if component.dynamic[attr].empty:
            continue

        unmatched = component.dynamic[attr].dtypes != dtype

        if unmatched.any():
            _log_or_raise(
                strict,
                "The following columns of time-varying attribute %s in %s_t"
                " have the wrong dtype:\n%s\nThey are:\n%s\nbut should be:\n%s",
                attr,
                component.list_name,
                unmatched.index[unmatched],
                component.dynamic[attr].dtypes[unmatched],
                typ,
            )


@deprecated_common_kwargs
def check_investment_periods(n: Network, strict: bool = False) -> None:
    """Check if investment periods are aligned with snapshots.

    Activate strict mode in general consistency check by passing `['investment_periods']`
    to the `strict` argument.

    Parameters
    ----------
    n : pypsa.Network
        The network to check.
    strict : bool, optional
        If True, raise an error instead of logging a warning.

    See Also
    --------
    [pypsa.Network.consistency_check][] : General consistency check method, which can be
                                      runs all consistency checks.


    """
    constraint_periods = set(n.global_constraints.investment_period.dropna().unique())
    if isinstance(n.snapshots, pd.MultiIndex):
        if not constraint_periods.issubset(n.snapshots.unique("period")):
            msg = (
                "The global constraints contain investment periods which "
                "are not in the set of optimized snapshots."
            )
            if strict:
                raise ValueError(msg)
            _log_or_raise(strict, msg)
    elif constraint_periods:
        msg = (
            "The global constraints contain investment periods but "
            "snapshots are not multi-indexed."
        )
        if strict:
            raise ValueError(msg)
        _log_or_raise(strict, msg)


@deprecated_common_kwargs
def check_shapes(n: Network, strict: bool = False) -> None:
    """Check if shapes are aligned with related components.

    Activate strict mode in general consistency check by passing `['shapes']` to the
    `strict` argument.

    Parameters
    ----------
    n : pypsa.Network
        The network to check.
    strict : bool, optional
        If True, raise an error instead of logging a warning.

    See Also
    --------
    [pypsa.Network.consistency_check][] : General consistency check method, which runs
    all consistency checks.


    """
    shape_components = n.shapes.component.unique()
    for c in set(shape_components) & set(n.all_components):
        geos = n.shapes.query("component == @c")
        not_included = geos.index[~geos.idx.isin(n.static(c).index)]

        if not not_included.empty:
            _log_or_raise(
                strict,
                "The following shapes are related to component %s and"
                " have idx values that are not included in the component's index:\n%s",
                c,
                not_included,
            )


@deprecated_common_kwargs
def check_nans_for_component_default_attrs(
    n: Network, component: Components, strict: bool = False
) -> None:
    """Check for missing values in component attributes.

    Activate strict mode in general consistency check by passing `['nans_for_component_default_attrs']`
    the `strict` argument.

    Checks for all attributes if they are nan but have a default value, which is not
     nan.

    Parameters
    ----------
    n : pypsa.Network
        The network to check.
    component : pypsa.Component
        The component to check.
    strict : bool, optional
        If True, raise an error instead of logging a warning.

    See Also
    --------
    [pypsa.Network.consistency_check][] : General consistency check method, which runs
    all consistency checks.

    """
    # Get non-NA and not-empty default attributes for the current component
    default = n.component_attrs[component.name]["default"]
    not_null_component_attrs = n.component_attrs[component.name][
        default.notna() & default.ne("")
    ].index

    # Remove attributes that are not in the component's static data
    relevant_static_df = component.static[
        list(set(component.static.columns).intersection(not_null_component_attrs))
    ]

    # Run the check for nan values on relevant static data
    if (isna := relevant_static_df.isna().any()).any():
        nan_cols = relevant_static_df.columns[isna]
        _log_or_raise(
            strict,
            "Encountered nan's in static data for columns %s of component '%s'.",
            nan_cols.to_list(),
            component.name,
        )

    # Remove attributes that are not in the component's time series data (if
    # there is any)
    relevant_series_dfs = {
        key: value
        for key, value in component.dynamic.items()
        if key in not_null_component_attrs and not value.empty
    }

    # Run the check for nan values on relevant data
    for key, values_df in relevant_series_dfs.items():
        if (isna := values_df.isna().any()).any():
            nan_cols = values_df.columns[isna]
            _log_or_raise(
                strict,
                "Encountered nan's in varying data '%s' for columns %s of component '%s'.",
                key,
                nan_cols.to_list(),
                component.name,
            )


def check_for_missing_carrier_colors(n: Network, strict: bool = False) -> None:
    """Check if carriers are missing colors.

    Parameters
    ----------
    n : pypsa.Network
        The network to check.
    strict : bool, optional
        If True, raise an error instead of logging a warning.

    """
    missing_colors = n.carriers[n.carriers.color.isna() | n.carriers.color.eq("")]
    if not missing_colors.empty:
        _log_or_raise(
            strict,
            "The following carriers are missing colors:\n%s",
            missing_colors.index,
        )


@deprecated(
    deprecated_in="0.35",
    removed_in="1.0",
    details="Use `n.consistency_check` instead.",
)
def consistency_check(
    n: Network, check_dtypes: bool = False, strict: Sequence | None = None
) -> None:
    """Use `n.consistency_check` instead."""
    return n.consistency_check(check_dtypes=check_dtypes, strict=strict)


@deprecated(
    deprecated_in="0.35",
    removed_in="1.0",
    details="Use `n.consistency_check_plots` instead.",
)
def plotting_consistency_check(n: Network, strict: Sequence | None = None) -> None:
    """Use `n.consistency_check_plots` instead."""
    return n.consistency_check_plots(strict=strict)


class NetworkConsistencyMixin(_NetworkABC):
    """Mixin class for network consistency checks.

    Class only inherits to [pypsa.Network][] and should not be used directly.
    All attributes and methods can be used within any Network instance.
    """

    calculate_dependent_values: Callable
    iterate_components: Callable

    def consistency_check(
        self, check_dtypes: bool = False, strict: Sequence | None = None
    ) -> None:
        """Check network for consistency.

        Runs a series of checks on the network to ensure that it is consistent, e.g. that
        all components are connected to existing buses and that no impedances are singular.

        Parameters
        ----------
        check_dtypes : bool, optional
            If True, check the dtypes of the attributes in the components.
        strict : list, optional
            If some checks should raise an error instead of logging a warning, pass a list
            of strings with the names of the checks to be strict about. If 'all' is passed,
            all checks will be strict. The default is no strict checks.

        Raises
        ------
        ConsistencyError : If any of the checks fail and strict mode is activated.

        See Also
        --------
        [pypsa.consistency.check_for_unknown_buses][] : Check if buses are attached to component but are not defined in the network.
        [pypsa.consistency.check_for_disconnected_buses][] : Check if network has buses that are not connected to any component.
        [pypsa.consistency.check_for_unknown_carriers][] : Check if carriers are attached to
            component but are not defined in the network.
        [pypsa.consistency.check_for_zero_impedances][] : Check if component has zero
            impedances. Only checks passive branch components.
        [pypsa.consistency.check_for_zero_s_nom][] : Check if component has zero s_nom. Only
            checks transformers.
        [pypsa.consistency.check_time_series][] : Check if time series of component are
            aligned with network snapshots.
        [pypsa.consistency.check_static_power_attributes][] : Check static attrs p_now, s_nom,
            e_nom in any component.
        [pypsa.consistency.check_time_series_power_attributes][] : Check `p_max_pu` and
            `e_max_pu` nan and infinite values in time series.
        [pypsa.consistency.check_assets][] : Check if assets are only committable or
            extendable, but not both.
        [pypsa.consistency.check_generators][] : Check the consistency of generator attributes
            before the simulation.
        [pypsa.consistency.check_dtypes_][] : Check if the dtypes of the attributes in the
            component are as expected.
        [pypsa.consistency.check_investment_periods][] : Check if investment periods are aligned
            with snapshots.
        [pypsa.consistency.check_shapes][] : Check if shapes are aligned with related components.
        [pypsa.consistency.check_nans_for_component_default_attrs][] : Check for missing values
            in component attributes.

        """
        if strict is None:
            strict = []

        strict_options = [
            "unknown_buses",
            "unknown_carriers",
            "time_series",
            "static_power_attrs",
            "time_series_power_attrs",
            "nans_for_component_default_attrs",
            "zero_impedances",
            "zero_s_nom",
            "assets",
            "generators",
            "disconnected_buses",
            "investment_periods",
            "shapes",
            "dtypes",
        ]

        if "all" in strict:
            strict = strict_options
        if not all(s in strict_options for s in strict):
            msg = (
                f"Invalid strict option(s) {set(strict) - set(strict_options)}. "
                f"Valid options are {strict_options}. Please check the documentation for "
                "more details on them."
            )
            raise ValueError(msg)

        self.calculate_dependent_values()

        # TODO: Check for bidirectional links with efficiency < 1.
        # TODO: Warn if any ramp limits are 0.

        # Per component checks
        for c in self.iterate_components():
            # Checks all components
            check_for_unknown_buses(self, c, "unknown_buses" in strict)
            check_for_unknown_carriers(self, c, "unkown_carriers" in strict)
            check_time_series(self, c, "time_series" in strict)
            check_static_power_attributes(self, c, "static_power_attrs" in strict)
            check_time_series_power_attributes(
                self, c, "time_series_power_attrs" in strict
            )
            check_nans_for_component_default_attrs(
                self, c, "nans_for_component_default_attrs" in strict
            )
            # Checks passive_branch_components
            check_for_zero_impedances(self, c, "zero_impedances" in strict)
            # Checks transformers
            check_for_zero_s_nom(c, "zero_s_nom" in strict)
            # Checks generators and links
            check_assets(self, c, "assets" in strict)
            # Checks generators
            check_generators(c, "generators" in strict)

            if check_dtypes:
                check_dtypes_(c, "dtypes" in strict)

        # Combined checks
        check_for_disconnected_buses(self, "disconnected_buses" in strict)
        check_investment_periods(self, "investment_periods" in strict)
        check_shapes(self, "shapes" in strict)

    def consistency_check_plots(self, strict: Sequence | None = None) -> None:
        """Check network for consistency for plotting functions.

        Parameters
        ----------
        strict : list, optional
            If some checks should raise an error instead of logging a warning, pass a list
            of strings with the names of the checks to be strict about. If 'all' is passed,
            all checks will be strict. The default is no strict checks.


        See Also
        --------
        [pypsa.consistency.consistency_check][] : General consistency check method, which can be
            runs all consistency checks.
        [pypsa.consistency.check_for_unknown_buses][] : Check if buses are attached to
            component but are not defined in the network.
        [pypsa.consistency.check_for_unknown_carriers][] : Check if carriers are attached to
            component but are not defined in the network.

        """
        if strict is None:
            strict = []

        strict_options = ["unknown_carriers", "missing_carrier_colors"]

        if "all" in strict:
            strict = strict_options

        if not all(s in strict_options for s in strict):
            msg = (
                f"Invalid strict option(s) {set(strict) - set(strict_options)}. "
                f"Valid options are {strict_options}. Please check the documentation for "
                "more details on them."
            )
            raise ValueError(msg)

        for c in self.iterate_components():
            check_for_unknown_carriers(self, c, strict="unknown_carriers" in strict)
        check_for_missing_carrier_colors(
            self,  # type: ignore
            strict="missing_carrier_colors" in strict,
        )
