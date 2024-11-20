"""
Consistency check functions for PyPSA networks.

Mainly used in the `Network.consistency_check()` method.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pypsa.utils import deprecated_common_kwargs

if TYPE_CHECKING:
    from pypsa.components import Component, Network

logger = logging.getLogger(__name__)


def _bus_columns(df: pd.DataFrame) -> pd.Index:
    return df.columns[df.columns.str.startswith("bus")]


@deprecated_common_kwargs
def check_for_unknown_buses(n: Network, component: Component) -> None:
    """
    Check if buses are attached to component but are not defined in the network.

    Parameters
    ----------
    n : pypsa.Network
        The network to check.
    component : pypsa.Component
        The component to check.

    """
    for attr in _bus_columns(component.static):
        missing = ~component.static[attr].isin(n.buses.index)
        # if bus2, bus3... contain empty strings do not warn
        if component.name in n.branch_components and int(attr[-1]) > 1:
            missing &= component.static[attr] != ""
        if missing.any():
            logger.warning(
                "The following %s have buses which are not defined:\n%s",
                component.list_name,
                component.static.index[missing],
            )


@deprecated_common_kwargs
def check_for_disconnected_buses(n: Network) -> None:
    """
    Check if network has buses that are not connected to any component.

    Parameters
    ----------
    n : pypsa.Network
        The network to check.

    """
    connected_buses = set()
    for component in n.iterate_components():
        for attr in _bus_columns(component.static):
            connected_buses.update(component.static[attr])

    disconnected_buses = set(n.buses.index) - connected_buses
    if disconnected_buses:
        logger.warning(
            "The following buses have no attached components, which can break the "
            "lopf:\n%s",
            disconnected_buses,
        )


@deprecated_common_kwargs
def check_for_unknown_carriers(n: Network, component: Component) -> None:
    """
    Check if carriers are attached to component but are not defined in the network.

    Parameters
    ----------
    n : pypsa.Network
        The network to check.
    component : pypsa.Component
        The component to check.

    """
    if "carrier" in component.static.columns:
        missing = (
            ~component.static["carrier"].isin(n.carriers.index)
            & component.static["carrier"].notna()
            & (component.static["carrier"] != "")
        )
        if missing.any():
            logger.warning(
                "The following %s have carriers which are not defined:\n%s",
                component.list_name,
                component.static.index[missing],
            )


@deprecated_common_kwargs
def check_for_zero_impedances(n: Network, component: Component) -> None:
    """
    Check if component has zero impedances. Only checks passive branch components.

    Parameters
    ----------
    n : pypsa.Network
        The network to check.
    component : pypsa.Component
        The component to check.

    """
    if component.name in n.passive_branch_components:
        for attr in ["x", "r"]:
            bad = component.static[attr] == 0
            if bad.any():
                logger.warning(
                    "The following %s have zero %s, which "
                    "could break the linear load flow:\n%s",
                    component.list_name,
                    attr,
                    component.static.index[bad],
                )


@deprecated_common_kwargs
def check_for_zero_s_nom(component: Component) -> None:
    """
    Check if component has zero s_nom. Only checks transformers.

    Parameters
    ----------
    component : pypsa.Component
        The component to check.

    """
    if component.name in {"Transformer"}:
        bad = component.static["s_nom"] == 0
        if bad.any():
            logger.warning(
                "The following %s have zero s_nom, which is used "
                "to define the impedance and will thus break "
                "the load flow:\n%s",
                component.list_name,
                component.static.index[bad],
            )


@deprecated_common_kwargs
def check_time_series(n: Network, component: Component) -> None:
    """
    Check if time series of component are aligned with network snapshots.

    Parameters
    ----------
    n : pypsa.Network
        The network to check.
    component : pypsa.Component
        The component to check.

    """
    for attr in component.attrs.index[component.attrs.varying & component.attrs.static]:
        attr_df = component.dynamic[attr]

        diff = attr_df.columns.difference(component.static.index)
        if len(diff):
            logger.warning(
                "The following %s have time series defined "
                "for attribute %s in n.%s_t, but are "
                "not defined in n.%s:\n%s",
                component.list_name,
                attr,
                component.list_name,
                component.list_name,
                diff,
            )

        if not n.snapshots.equals(attr_df.index):
            logger.warning(
                "The index of the time-dependent Dataframe for attribute "
                "%s of n.%s_t is not aligned with network snapshots",
                attr,
                component.list_name,
            )


@deprecated_common_kwargs
def check_static_power_attributes(n: Network, component: Component) -> None:
    """
    Check static attrs p_now, s_nom, e_nom in any component.

    Parameters
    ----------
    n : pypsa.Network
        The network to check.
    component : pypsa.Component
        The component to check.

    """
    static_attrs = ["p_nom", "s_nom", "e_nom"]
    if component.name in n.all_components - {"TransformerType"}:
        static_attr = component.attrs.query("static").index.intersection(static_attrs)
        if len(static_attr):
            attr = static_attr[0]
            bad = component.static[attr + "_max"] < component.static[attr + "_min"]
            if bad.any():
                logger.warning(
                    "The following %s have smaller maximum than "
                    "minimum expansion limit which can lead to "
                    "infeasibilty:\n%s",
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
                    logger.warning(
                        "Encountered nan's in column %s of component '%s'.",
                        col,
                        component.name,
                    )


@deprecated_common_kwargs
def check_time_series_power_attributes(n: Network, component: Component) -> None:
    """
    Check `p_max_pu` and `e_max_pu` nan and infinite values in time series.

    Parameters
    ----------
    n : pypsa.Network
        The network to check.
    component : pypsa.Component
        The component to check.

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
                    logger.warning(
                        "The attribute %s of element %s of %s has "
                        "NaN values for the following snapshots:\n%s",
                        attr + "_max_pu",
                        col,
                        component.list_name,
                        max_pu.index[max_pu[col].isna()],
                    )
            if min_pu.isna().to_numpy().any():
                for col in min_pu.columns[min_pu.isna().any()]:
                    logger.warning(
                        "The attribute %s of element %s of %s has "
                        "NaN values for the following snapshots:\n%s",
                        attr + "_min_pu",
                        col,
                        component.list_name,
                        min_pu.index[min_pu[col].isna()],
                    )

            # check for infinite values
            if np.isinf(max_pu).to_numpy().any():
                for col in max_pu.columns[np.isinf(max_pu).any()]:
                    logger.warning(
                        "The attribute %s of element %s of %s has "
                        "infinite values for the following snapshots:\n%s",
                        attr + "_max_pu",
                        col,
                        component.list_name,
                        max_pu.index[np.isinf(max_pu[col])],
                    )
            if np.isinf(min_pu).to_numpy().any():
                for col in min_pu.columns[np.isinf(min_pu).any()]:
                    logger.warning(
                        "The attribute %s of element %s of %s has "
                        "infinite values for the following snapshots:\n%s",
                        attr + "_min_pu",
                        col,
                        component.list_name,
                        min_pu.index[np.isinf(min_pu[col])],
                    )

            diff = max_pu - min_pu
            diff = diff[diff < 0].dropna(axis=1, how="all")
            for col in diff.columns:
                logger.warning(
                    "The element %s of %s has a smaller maximum "
                    "than minimum operational limit which can "
                    "lead to infeasibility for the following snapshots:\n%s",
                    col,
                    component.list_name,
                    diff[col].dropna().index,
                )


@deprecated_common_kwargs
def check_assets(n: Network, component: Component) -> None:
    """
    Check if assets are only committable or extendable, but not both.

    Parameters
    ----------
    n : pypsa.Network
        The network to check.
    component : pypsa.Component
        The component to check.

    """
    if component.name in {"Generator", "Link"}:
        committables = n.get_committable_i(component.name)
        extendables = n.get_extendable_i(component.name)
        intersection = committables.intersection(extendables)
        if not intersection.empty:
            logger.warning(
                "Assets can only be committable or extendable. Found "
                f"assets in component {component.name} which are both:"
                f"\n\n\t{', '.join(intersection)}"
            )


@deprecated_common_kwargs
def check_generators(component: Component) -> None:
    """
    Check the consistency of generator attributes before the simulation.

    This function performs the following checks on generator components:
    1. Ensures that committable generators are not both up and down before the simulation.
    2. Verifies that the minimum total energy to be produced (e_sum_min) is not greater than the maximum total energy to be produced (e_sum_max).

    Parameters
    ----------
    component : Component
        The generator component to be checked.

    Returns
    -------
    None

    """
    if component.name in {"Generator"}:
        bad_uc_gens = component.static.index[
            component.static.committable
            & (component.static.up_time_before > 0)
            & (component.static.down_time_before > 0)
        ]
        if not bad_uc_gens.empty:
            logger.warning(
                "The following committable generators were both up and down"
                f" before the simulation: {bad_uc_gens}."
                " This could cause an infeasibility."
            )

        bad_e_sum_gens = component.static.index[
            component.static.e_sum_min > component.static.e_sum_max
        ]
        if not bad_e_sum_gens.empty:
            logger.warning(
                "The following generators have e_sum_min > e_sum_max, "
                "which can lead to infeasibility:\n"
                f"{bad_e_sum_gens}.",
            )


@deprecated_common_kwargs
def check_dtypes_(component: Component) -> None:
    """
    Check if the dtypes of the attributes in the component are as expected.

    Parameters
    ----------
    n : pypsa.Network
        The network to check.
    component : pypsa.Component
        The component to check.

    """
    dtypes_soll = component.attrs.loc[component.attrs["static"], "dtype"].drop("name")
    unmatched = component.static.dtypes[dtypes_soll.index] != dtypes_soll

    if unmatched.any():
        logger.warning(
            "The following attributes of the dataframe %s "
            "have the wrong dtype:\n%s\n"
            "They are:\n%s\nbut should be:\n%s",
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
            logger.warning(
                "The following columns of time-varying attribute "
                "%s in %s_t have the wrong dtype:\n%s\n"
                "They are:\n%s\nbut should be:\n%s",
                attr,
                component.list_name,
                unmatched.index[unmatched],
                component.dynamic[attr].dtypes[unmatched],
                typ,
            )


@deprecated_common_kwargs
def check_investment_periods(n: Network) -> None:
    """
    Check if investment periods are aligned with snapshots.

    Parameters
    ----------
    n : pypsa.Network
        The network to check.

    """
    constraint_periods = set(n.global_constraints.investment_period.dropna().unique())
    if isinstance(n.snapshots, pd.MultiIndex):
        if not constraint_periods.issubset(n.snapshots.unique("period")):
            msg = (
                "The global constraints contain investment periods which "
                "are not in the set of optimized snapshots."
            )
            raise ValueError(msg)
    else:
        if constraint_periods:
            msg = (
                "The global constraints contain investment periods but "
                "snapshots are not multi-indexed."
            )
            raise ValueError(msg)


@deprecated_common_kwargs
def check_shapes(n: Network) -> None:
    """
    Check if shapes are aligned with related components.

    Parameters
    ----------
    n : pypsa.Network
        The network to check.
    component : pypsa.Component
        The component to check.

    """
    shape_components = n.shapes.component.unique()
    for c in set(shape_components) & set(n.all_components):
        geos = n.shapes.query("component == @c")
        not_included = geos.index[~geos.idx.isin(n.static(c).index)]

        if not not_included.empty:
            logger.warning(
                f"The following shapes are related to component {c} and have"
                f" idx values that are not included in the component's index:\n"
                f"{not_included}"
            )


@deprecated_common_kwargs
def check_nans_for_component_default_attrs(n: Network, component: Component) -> None:
    """
    Check for missing values in component attributes.

    Checks for all attributes if they are nan but have a default value, which is not
     nan.

    Parameters
    ----------
    n : pypsa.Network
        The network to check.
    component : pypsa.Component
        The component to check.

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
        logger.warning(
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
            logger.warning(
                "Encountered nan's in varying data '%s' for columns %s of component '%s'.",
                key,
                nan_cols.to_list(),
                component.name,
            )
