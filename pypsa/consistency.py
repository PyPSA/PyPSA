# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Consistency check functions for PyPSA networks.

Mainly used in the `Network.consistency_check()` method.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pypsa._options import options
from pypsa.constants import RE_PORTS_FILTER
from pypsa.guards import _assert_data_integrity
from pypsa.network.abstract import _NetworkABC

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pypsa import Network
    from pypsa.components import Components
    from pypsa.type_utils import NetworkType

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


def check_for_unknown_buses(
    n: NetworkType, component: Components, strict: bool = False
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
    [pypsa.Network.consistency_check][], [pypsa.Network.sanitize][]

    """
    for attr in _bus_columns(component.static):
        missing = ~component.static[attr].astype(str).isin(n.c.buses.names)
        # if bus2, bus3... contain empty strings do not warn
        if component.name in n.branch_components and int(attr[-1]) > 1:
            missing &= component.static[attr] != ""
        # if bus contains empty strings for global constraints do not warn
        if component.name == "GlobalConstraint":
            missing &= component.static[attr] != ""
        if missing.any():
            _log_or_raise(
                strict,
                "The following %s have buses which are not defined. Add them using "
                "n.add() or run n.sanitize() to add them automatically. Components "
                "with undefined buses:\n%s",
                component.list_name,
                component.static.index[missing],
            )


def check_for_disconnected_buses(n: NetworkType, strict: bool = False) -> None:
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
    [pypsa.Network.consistency_check][]

    """
    connected_buses = set()
    for component in n.components:
        for attr in _bus_columns(component.static):
            connected_buses.update(component.static[attr])

    disconnected_buses = set(n.c.buses.names) - connected_buses
    if disconnected_buses:
        _log_or_raise(
            strict,
            "The following buses have no attached components, which can break the lopf: %s",
            disconnected_buses,
        )


def check_for_unknown_carriers(
    n: NetworkType, component: Components, strict: bool = False
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
    [pypsa.Network.consistency_check][], [pypsa.Network.sanitize][]


    """
    if "carrier" in component.static.columns:
        missing = (
            ~component.static["carrier"].isin(n.c.carriers.names)
            & component.static["carrier"].notna()
            & (component.static["carrier"] != "")
        )
        if missing.any():
            _log_or_raise(
                strict,
                "The following %s have carriers which are not defined. Run n.sanitize()"
                " to add them. Components with undefined carriers:\n%s",
                component.list_name,
                component.static.index[missing],
            )


def check_for_zero_impedances(
    n: NetworkType, component: Components, strict: bool = False
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
    [pypsa.Network.consistency_check][]


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
    [pypsa.Network.consistency_check][]

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


def check_time_series(
    n: NetworkType, component: Components, strict: bool = False
) -> None:
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
    [pypsa.Network.consistency_check][]


    """
    for attr in component.defaults.index[
        component.defaults.varying & component.defaults.static
    ]:
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


def check_static_power_attributes(
    n: NetworkType, component: Components, strict: bool = False
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
    [pypsa.Network.consistency_check][]


    """
    static_attrs = ["p_nom", "s_nom", "e_nom"]
    if component.name in n.all_components - {"TransformerType"}:
        static_attr = component.defaults.query("static").index.intersection(
            static_attrs
        )
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


def check_time_series_power_attributes(
    n: NetworkType, component: Components, strict: bool = False
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
    [pypsa.Network.consistency_check][]

    """
    varying_attrs = ["p_max_pu", "e_max_pu"]
    if component.name in n.all_components - {"TransformerType"}:
        varying_attr = component.defaults.query("varying").index.intersection(
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


def check_assets(n: NetworkType, component: Components, strict: bool = False) -> None:
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
    [pypsa.Network.consistency_check][]

    """
    if component.name in {"Generator", "Link"}:
        committables = component.committables
        extendables = component.extendables
        intersection = committables.intersection(extendables)
        if not intersection.empty:
            _log_or_raise(
                strict,
                "Assets can only be committable or extendable."
                " Found assets in component %s which are both:\n\n\t%s",
                component.name,
                ", ".join(intersection),
            )


def check_cost_consistency(component: Components, strict: bool = False) -> None:
    """Check if both overnight_cost and capital_cost are set for the same asset.

    When both are specified, overnight_cost takes precedence and capital_cost is
    ignored.

    Activate strict mode in general consistency check by passing `['cost_consistency']`
    to the `strict` argument.

    Parameters
    ----------
    component : pypsa.Component
        The component to check.
    strict : bool, optional
        If True, raise an error instead of logging a warning.

    See Also
    --------
    `pypsa.Network.consistency_check`

    """
    static = component.static
    if not {"capital_cost", "overnight_cost"}.issubset(static.columns):
        return
    has_overnight = static["overnight_cost"].notna()
    has_capital = static["capital_cost"] != 0

    both_set = has_overnight & has_capital
    if both_set.any():
        assets = static.index[both_set].tolist()
        _log_or_raise(
            strict,
            "Component %s has assets with both 'overnight_cost' and 'capital_cost' "
            "set: %s. When 'overnight_cost' is provided, it takes precedence and "
            "'capital_cost' is ignored. Consider setting capital_cost=0 for these assets.",
            component.name,
            ", ".join(assets[:5]) + ("..." if len(assets) > 5 else ""),
        )

    if "discount_rate" in static.columns:
        missing_discount_rate = has_overnight & static["discount_rate"].isna()
        if missing_discount_rate.any():
            assets = static.index[missing_discount_rate].tolist()
            _log_or_raise(
                True,
                "Component %s has assets with 'overnight_cost' set but missing "
                "'discount_rate': %s. Provide discount_rate for annuitization.",
                component.name,
                ", ".join(assets[:5]) + ("..." if len(assets) > 5 else ""),
            )

    if "lifetime" in static.columns:
        missing_lifetime_rate = has_overnight & static["lifetime"].isna()
        if missing_lifetime_rate.any():
            assets = static.index[missing_lifetime_rate].tolist()
            _log_or_raise(
                True,
                "Component %s has assets with 'overnight_cost' set but missing "
                "'lifetime': %s. Provide lifetime for annuitization.",
                component.name,
                ", ".join(assets[:5]) + ("..." if len(assets) > 5 else ""),
            )


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
    [pypsa.Network.consistency_check][]

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
        bad_uc_gens = component.static.index[
            component.static.committable
            & (component.static.up_time_before == 0)
            & (component.static.p_init.notnull())
        ]
        if not bad_uc_gens.empty:
            _log_or_raise(
                strict,
                "The following committable generators were down "
                "before the simulation and have a p_init value. The latter will be ignored: %s.",
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


def check_dtypes_(component: Components, strict: bool = False) -> None:
    """Check if the dtypes of the attributes in the component are as expected.

    Activate strict mode in general consistency check by passing `['dtypes']` to the
    `strict` argument.

    Parameters
    ----------
    component : pypsa.Component
        The component to check.
    strict : bool, optional
        If True, raise an error instead of logging a warning.

    See Also
    --------
    [pypsa.Network.consistency_check][]

    """
    dtypes_soll = component.defaults.loc[component.defaults["static"], "dtype"].drop(
        "name"
    )
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

    types_soll = component.defaults.loc[component.defaults["varying"], ["typ", "dtype"]]

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


def check_investment_periods(n: NetworkType, strict: bool = False) -> None:
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
    [pypsa.Network.consistency_check][]

    """
    constraint_periods = set(
        n.c.global_constraints.static.investment_period.dropna().unique()
    )
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


def check_shapes(n: NetworkType, strict: bool = False) -> None:
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
    [pypsa.Network.consistency_check][]

    """
    shape_components = n.c.shapes.static.component.unique()
    for c in set(shape_components) & set(n.all_components):
        geos = n.c.shapes.static.query("component == @c")
        not_included = geos.index[~geos.idx.isin(n.c[c].static.index)]

        if not not_included.empty:
            _log_or_raise(
                strict,
                "The following shapes are related to component %s and"
                " have idx values that are not included in the component's index:\n%s",
                c,
                not_included,
            )


def check_nans_for_component_default_attrs(
    n: NetworkType, component: Components, strict: bool = False
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
    [pypsa.Network.consistency_check][]

    """
    # Get non-NA and not-empty default attributes for the current component
    default = component.defaults["default"]
    not_null_component_attrs = component.defaults[
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

    See Also
    --------
    [pypsa.Network.consistency_check][], [pypsa.Network.sanitize][]

    """
    missing_colors = n.c.carriers.static[
        n.c.carriers.static.color.isna() | n.c.carriers.static.color.eq("")
    ]
    if not missing_colors.empty:
        _log_or_raise(
            strict,
            "The following carriers are missing colors. Run n.sanitize()"
            " to assign them. Carriers missing colors:\n%s",
            missing_colors.index,
        )


class NetworkConsistencyMixin(_NetworkABC):
    """Mixin class for network consistency checks.

    Class inherits to [pypsa.Network][]. All attributes and methods can be used
    within any Network instance.
    """

    def consistency_check(
        self, check_dtypes: bool = False, strict: Sequence | None = None
    ) -> None:
        """Check network for consistency.

        <!-- md:badge-version v0.7.0 -->

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
        ConsistencyError
            If any of the checks fail and strict mode is activated.

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
            "cost_consistency",
            "disconnected_buses",
            "investment_periods",
            "shapes",
            "dtypes",
            "scenarios_sum",
            "scenario_invariant_attrs",
            "line_types",
            "transformer_types",
            "slack_bus_consistency",
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
        for c in self.components:
            # Checks all components
            check_for_unknown_buses(self, c, "unknown_buses" in strict)
            check_for_unknown_carriers(self, c, "unknown_carriers" in strict)
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
            # Checks cost attributes consistency
            check_cost_consistency(c)

            if check_dtypes:
                check_dtypes_(c, "dtypes" in strict)

        # Combined checks
        check_for_disconnected_buses(self, "disconnected_buses" in strict)
        check_investment_periods(self, "investment_periods" in strict)
        check_shapes(self, "shapes" in strict)
        check_scenarios_sum_to_one(self, "scenarios_sum" in strict)
        check_scenario_invariant_attributes(self, "scenario_invariant_attrs" in strict)
        check_line_types_consistency(self, "line_types" in strict)
        check_transformer_types_consistency(self, "transformer_types" in strict)
        check_stochastic_slack_bus_consistency(self, "slack_bus_consistency" in strict)

        # Optional runtime verification
        if options.debug.runtime_verification:
            _assert_data_integrity(self)

    def consistency_check_plots(self, strict: Sequence | None = None) -> None:
        """Check network for consistency for plotting functions.

        <!-- md:badge-version v0.34.0 -->

        Parameters
        ----------
        strict : list, optional
            If some checks should raise an error instead of logging a warning, pass a list
            of strings with the names of the checks to be strict about. If 'all' is passed,
            all checks will be strict. The default is no strict checks.

        Raises
        ------
        ConsistencyError
            If any of the checks fail and strict mode is activated.

        See Also
        --------
        [pypsa.Network.consistency_check][], [pypsa.consistency.check_for_unknown_buses][],
        [pypsa.consistency.check_for_unknown_carriers][]

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

        for c in self.components:
            check_for_unknown_carriers(self, c, strict="unknown_carriers" in strict)
        check_for_missing_carrier_colors(
            self,  # type: ignore
            strict="missing_carrier_colors" in strict,
        )

    def sanitize(self) -> None:
        """Sanitize the network to ensure data integrity.

        <!-- md:badge-version v1.1.0 -->

        This method performs a set of operations to heal the networks data integrity.
        For a full list of operations which are done, check the See Also section below.

        See Also
        --------
        [pypsa.components.Buses.add_missing_buses][],
        [pypsa.components.Carriers.add_missing_carriers][],
        [pypsa.components.Carriers.assign_colors][]

        """
        logger.info("Sanitizing network...")

        self.c.buses.add_missing_buses()

        self.c.carriers.add_missing_carriers()
        self.c.carriers.assign_colors()

        logger.info("Network sanitization complete.")


def check_scenarios_sum_to_one(n: NetworkType, strict: bool = False) -> None:
    """Check if scenarios probabilities sum to 1.

    This check verifies that scenario probabilities have not been modified after
    initialization to break the constraint that they must sum to 1.

    Activate strict mode in general consistency check by passing `['scenarios_sum']`
    to the `strict` argument.

    Parameters
    ----------
    n : pypsa.Network
        The network to check.
    strict : bool, optional
        If True, raise an error instead of logging a warning.

    See Also
    --------
    [pypsa.Network.consistency_check][]

    """
    if n.has_scenarios:
        total_weight = n.scenario_weightings["weight"].sum()

        if not np.isclose(total_weight, 1.0, rtol=1e-10, atol=1e-10):
            _log_or_raise(
                strict,
                "Scenario probabilities must sum to 1.0 (got %.10g). "
                "This may indicate scenarios were modified after initialization.",
                total_weight,
            )


def check_scenario_invariant_attributes(n: NetworkType, strict: bool = False) -> None:
    """Check if invariant component attributes are not changed across scenarios.

    There are some component attributes that must remain the same across scenarios.
    These attributes define the topology of the network or the mathematical structure.
    We raise an error if user attemps to modify them across scenarios.
    Any difference in values (including NaN vs non-NaN) will trigger an error.

    Parameters
    ----------
    n : pypsa.Network
        The network to check.
    strict : bool, optional
        If True, raise an error instead of logging a warning.

    See Also
    --------
    [pypsa.Network.consistency_check][]

    """
    # This test is for stochastic networks only
    if not n.has_scenarios:
        return

    # Attributes that must be identical across all scenarios
    INVARIANT_ATTRS = {
        "name",
        "bus",
        # "control" is excluded - different buses can have different control types across scenarios
        # but we ensure consistent slack bus selection separately
        "type",
        "p_nom_extendable",  # changes mathematical problem
        "s_nom_extendable",
        "e_nom_extendable",
        "p_nom_mod",  # modular investment is first-stage decision
        "s_nom_mod",
        "e_nom_mod",
        "committable",  # changes mathematical problem
        "sign",
        "carrier",
        "weight",
        "p_nom_opt",  # optimization result
        "s_nom_opt",
        "e_nom_opt",
        "build_year",
        "lifetime",
        "active",  # theoretically can be different, but problematic with "Line"
    }

    for component in n.components:
        if component.static.index.nlevels < 2:
            continue  # No scenario dimension

        # Get attributes that exist for this component and are in invariant list
        component_invariant_attrs = INVARIANT_ATTRS.intersection(
            component.static.columns
        )

        if not component_invariant_attrs:
            continue

        # Group by component name (second level of MultiIndex) and check for differences
        grouped = component.static.groupby(level=1)  # Group by component name

        for attr in component_invariant_attrs:
            for comp_name, group in grouped:
                # Check if all scenarios have the same value for this attribute
                unique_values = group[attr].unique()

                # If there's more than one unique value, it's an error - no exceptions
                if len(unique_values) > 1:
                    scenarios_with_diff = (
                        group[group[attr] != group[attr].iloc[0]]
                        .index.get_level_values(0)
                        .tolist()
                    )
                    _log_or_raise(
                        True,
                        "Component '%s' of type '%s' has attribute '%s' that varies across scenarios. "
                        "This attribute must be identical across all scenarios. "
                        "Scenarios with different values: %s. Values: %s",
                        comp_name,
                        component.name,
                        attr,
                        scenarios_with_diff,
                        group[attr].to_dict(),
                    )


def check_line_types_consistency(n: NetworkType, strict: bool = False) -> None:
    """Check that line_types are identical across all scenarios.

    In stochastic networks, line_types must be identical across all scenarios

    Parameters
    ----------
    n : pypsa.Network
        The network to check.
    strict : bool, optional
        If True, raise an error instead of logging a warning.

    See Also
    --------
    [pypsa.Network.consistency_check][]

    """
    if not n.has_scenarios:
        return

    # Check line_types consistency across scenarios
    if not n.c.line_types.static.empty and len(n.scenarios) > 1:
        # Get reference line_types from first scenario
        reference_scenario = n.scenarios[0]
        reference_line_types = n.c.line_types.static.xs(
            reference_scenario, level="scenario"
        )

        # Check each other scenario
        for scenario in n.scenarios[1:]:
            scenario_line_types = n.c.line_types.static.xs(scenario, level="scenario")

            # Check if DataFrames are equal
            if not reference_line_types.equals(scenario_line_types):
                _log_or_raise(
                    strict,
                    "line_types must be identical across all scenarios. "
                    "Found differences between scenario '%s' and '%s'. "
                    "line_types define physical characteristics and cannot vary across scenarios.",
                    reference_scenario,
                    scenario,
                )


def check_transformer_types_consistency(n: NetworkType, strict: bool = False) -> None:
    """Check that transformer_types are identical across all scenarios.

    In stochastic networks, transformer_types must be identical across all scenarios
    since they define physical characteristics of transformers.

    Parameters
    ----------
    n : pypsa.Network
        The network to check.
    strict : bool, optional
        If True, raise an error instead of logging a warning.

    See Also
    --------
    [pypsa.Network.consistency_check][]

    """
    if not n.has_scenarios:
        return

    # Check transformer_types consistency across scenarios
    if not n.c.transformer_types.static.empty and len(n.scenarios) > 1:
        # Get reference transformer_types from first scenario
        reference_scenario = n.scenarios[0]
        reference_transformer_types = n.c.transformer_types.static.xs(
            reference_scenario, level="scenario"
        )

        # Check each other scenario
        for scenario in n.scenarios[1:]:
            scenario_transformer_types = n.c.transformer_types.static.xs(
                scenario, level="scenario"
            )

            # Check if DataFrames are equal
            if not reference_transformer_types.equals(scenario_transformer_types):
                _log_or_raise(
                    strict,
                    "transformer_types must be identical across all scenarios. "
                    "Found differences between scenario '%s' and '%s'. "
                    "transformer_types define physical characteristics and cannot vary across scenarios.",
                    reference_scenario,
                    scenario,
                )


def check_stochastic_slack_bus_consistency(
    n: NetworkType, strict: bool = False
) -> None:
    """Check that the same bus is chosen as slack across all scenarios in stochastic networks.

    Ensure that the same bus is consistently chosen as the slack bus
    to maintain mathematical consistency of the optimization problem.

    Parameters
    ----------
    n : pypsa.Network
        The network to check.
    strict : bool, optional
        If True, raise an error instead of logging a warning.

    See Also
    --------
    [pypsa.Network.consistency_check][]

    """
    # This test is for stochastic networks only
    if not n.has_scenarios:
        return

    # Check that each sub-network has the same slack bus across all scenarios
    if n.has_scenarios and "control" in n.c.buses.static.columns:
        # Extract slack buses for each scenario
        slack_buses_by_scenario = {}

        for scenario in n.scenarios:
            if n.c.buses.static.index.nlevels > 1:
                # MultiIndex case (stochastic network)
                scenario_buses = n.c.buses.static.xs(scenario, level="scenario")
                slack_buses = scenario_buses[scenario_buses.control == "Slack"]
                slack_buses_by_scenario[scenario] = set(slack_buses.index)
            else:
                # Single scenario case, shouldn't reach here for stochastic networks
                slack_buses = n.c.buses.static[n.c.buses.static.control == "Slack"]
                slack_buses_by_scenario[scenario] = set(slack_buses.index)

        # Compare slack buses across scenarios
        if len(slack_buses_by_scenario) > 1:
            scenarios = list(slack_buses_by_scenario.keys())
            reference_slack_buses = slack_buses_by_scenario[scenarios[0]]

            for scenario in scenarios[1:]:
                current_slack_buses = slack_buses_by_scenario[scenario]
                if reference_slack_buses != current_slack_buses:
                    _log_or_raise(
                        strict,
                        "Different slack buses found across scenarios. "
                        "This can cause mathematical inconsistency in stochastic optimization. "
                        "Reference scenario '%s' has slack buses %s, "
                        "but scenario '%s' has slack buses %s",
                        scenarios[0],
                        reference_slack_buses,
                        scenario,
                        current_slack_buses,
                    )
