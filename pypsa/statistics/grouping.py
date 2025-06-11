"""Groupers for PyPSA statistics.

Use them via the groupers instance via `pypsa.statistics.groupers`. Do not use the
grouping module directly.
"""

from __future__ import annotations

import logging
from inspect import signature
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pypsa import Network

import warnings

import pandas as pd

logger = logging.getLogger(__name__)


class Groupers:
    """Container for all the get_ methods."""

    def _map_with_multiindex(
        self, component_series: pd.Series, mapping_series: pd.Series
    ) -> pd.Series:
        """Map values across MultiIndex DataFrames.

        This handles the case where both series have MultiIndex (NetworkCollection case)
        by dropping scenario levels from the mapping series and removing duplicates.

        Parameters
        ----------
        component_series : pd.Series
            Series from component (e.g., generator buses)
        mapping_series : pd.Series
            Series to map to (e.g., bus carriers)

        Returns
        -------
        pd.Series
            Mapped series with same index as component_series

        """
        # Handle MultiIndex case (NetworkCollection)
        if isinstance(component_series.index, pd.MultiIndex) and isinstance(
            mapping_series.index, pd.MultiIndex
        ):
            # Drop all levels except the last one from mapping series
            n_levels_to_drop = len(mapping_series.index.names) - 1
            if n_levels_to_drop > 0:
                simplified_mapping = mapping_series.droplevel(
                    list(range(n_levels_to_drop))
                )
                # Remove duplicates (keep first occurrence)
                simplified_mapping = simplified_mapping[
                    ~simplified_mapping.index.duplicated(keep="first")
                ]
            else:
                simplified_mapping = mapping_series
            # Do the mapping
            return component_series.map(simplified_mapping)
        else:
            # Original behavior for single networks
            return component_series.map(mapping_series)

    def __repr__(self) -> str:
        """Return a string representation of the grouper container.

        Returns
        -------
        str
            String representation.

        Examples
        --------
        >>> pypsa.statistics.groupers # doctest: +ELLIPSIS
        Grouper container with the following groupers: bus, bus_carrier, ...

        """
        return (
            f"Grouper container with the following groupers: "
            f"{', '.join(self.list_groupers())}"
        )

    def __getitem__(self, keys: str | Callable | Sequence[str | Callable]) -> Callable:
        """Get a single or multi-indexed grouper method.

        Parameters
        ----------
        keys : str | Callable | Sequence[str | Callable]
            Single or multiple keys to get the grouper method.

        Returns
        -------
        Callable
            Grouper method.

        Examples
        --------
        Single indexed grouper

        >>> pypsa.statistics.groupers["carrier"] # doctest: +ELLIPSIS
        <function Groupers._multi_grouper.<locals>.multi_grouper at 0x...>

        Multi-indexed grouper

        >>> pypsa.statistics.groupers[["carrier", "bus"]] # doctest: +ELLIPSIS
        <function Groupers._multi_grouper.<locals>.multi_grouper at 0x...>

        """
        return self._multi_grouper(keys)

    def __setitem__(self, key: str, value: Callable) -> None:
        """Set a custom grouper method.

        Parameters
        ----------
        key : str
            Name of the custom grouper.

        value : Callable
            Custom grouper function.



        """
        raise NotImplementedError()

    def _get_generic_grouper(self, n: Network, c: str, key: str) -> pd.Series:
        try:
            return n.static(c)[key].rename(key)
        except KeyError as e:
            msg = f"Unknown grouper {key}."
            raise KeyError(msg) from e

    def list_groupers(self) -> dict:
        """List all available groupers which are avaliable on the module level.

        Returns
        -------
        dict
            Dictionary with all available groupers. The keys are the grouper names and
            the values are the grouper methods. The keys can be used to directly
            access the grouper in any `groupby` argument.

        Examples
        --------
        >>> pypsa.statistics.groupers.list_groupers().keys() # doctest: +ELLIPSIS
        dict_keys(['bus', 'bus_carrier', ...

        """
        no_groupers = ["add_grouper", "list_groupers"]
        return {
            key: getattr(self, key)
            for key in dir(self)
            if not key.startswith("_") and key not in no_groupers
        }

    def _multi_grouper(
        self, keys: str | Callable | Sequence[str | Callable]
    ) -> Callable:
        """Get a single or multi-indexed grouper method.

        Should be used via groupers __getitem__ method and not directly.

        Parameters
        ----------
        keys : str | Callable | Sequence[str | Callable]
            Single or multiple keys to get the grouper method.

        Returns
        -------
        Callable
            Grouper method.

        Examples
        --------
        Single indexed grouper

        >>> pypsa.statistics.groupers["carrier"] # doctest: +ELLIPSIS
        <function Groupers._multi_grouper.<locals>.multi_grouper at 0x...>

        Multi-indexed grouper

        >>> pypsa.statistics.groupers[["carrier", "bus"]] # doctest: +ELLIPSIS
        <function Groupers._multi_grouper.<locals>.multi_grouper at 0x...>

        """
        keys_: Sequence[str | Callable]
        if scalar_passed := isinstance(keys, str) or callable(keys):
            keys_ = (keys,)
        else:
            keys_ = keys

        def multi_grouper(
            n: Network, c: str, port: str = "", nice_names: bool = False
        ) -> list:
            grouped_data = []
            for key in keys_:
                if isinstance(key, str):
                    if key not in self.list_groupers():
                        grouped_data.append(self._get_generic_grouper(n, c, key))
                        continue
                    method = self.list_groupers()[key]
                else:
                    method = key

                kwargs: dict[str, str | bool] = {}
                if "port" in signature(method).parameters:
                    kwargs["port"] = port
                if "nice_names" in signature(method).parameters:
                    kwargs["nice_names"] = nice_names
                grouped_data.append(method(n, c, **kwargs))

            return grouped_data[0] if scalar_passed else grouped_data

        return multi_grouper

    def add_grouper(self, name: str, func: Callable) -> None:
        """Add a custom grouper to groupers on module level.

        After registering a custom grouper, it can be accessed via the groupers module
        level object and used in the statistics methods or as a groupers method.

        Parameters
        ----------
        name : str
            Name of the custom grouper. This will be used as the key in the groupby
            argument.
        func : Callable
            Custom grouper function, which must return a pandas Series with the same
            length as the component index and accept as arguments:
            * n (Network): The PyPSA network instance
            * c (str): Component name
            * port (str): Component port as integer string
            * nice_names (bool, optional): Whether to use nice carrier names


        """
        setattr(self, name, func)

    def carrier(self, n: Network, c: str, nice_names: bool = True) -> pd.Series:
        """Grouper method to group by the carrier of the components.

        Parameters
        ----------
        n : Network
            PyPSA network instance.
        c : str
            Components type name. E.g. "Generator", "StorageUnit", etc.
        nice_names : bool, optional
            Whether to use nice carrier names.

        Returns
        -------
        pd.Series
            Series with the carrier of the components.

        """
        static = n.static(c)
        fall_back = pd.Series("", index=static.index)
        carrier_series = static.get("carrier", fall_back).rename("carrier")
        if nice_names:
            carrier_series = carrier_series.replace(
                n.carriers.nice_name[lambda ds: ds != ""]
            ).replace("", "-")
        return carrier_series

    def bus_carrier(
        self, n: Network, c: str, port: str = "", nice_names: bool = True
    ) -> pd.Series:
        """Grouper method to group by the carrier of the attached bus of a component.

        Parameters
        ----------
        n : Network
            PyPSA network instance.
        c : str
            Components type name. E.g. "Generator", "StorageUnit", etc.
        port : str, optional
            Port of corresponding bus, which should be used.
        nice_names : bool, optional
            Whether to use nice carrier names.

        Returns
        -------
        pd.Series
            Series with the bus and carrier of the components.

        """
        bus = f"bus{port}"
        buses_carrier = self.carrier(n, "Bus", nice_names=nice_names)
        component_buses = n.static(c)[bus]

        return self._map_with_multiindex(component_buses, buses_carrier).rename(
            "bus_carrier"
        )

    def bus(self, n: Network, c: str, port: str = "") -> pd.Series:
        """Grouper method to group by the attached bus of the components.

        Parameters
        ----------
        n : Network
            PyPSA network instance.
        c : str
            Components type name. E.g. "Generator", "StorageUnit", etc.
        port : str, optional
            Port of corresponding bus, which should be used.

        Returns
        -------
        pd.Series
            Series with the bus of the components.

        """
        bus = f"bus{port}"
        return n.static(c)[bus].rename("bus")

    def country(self, n: Network, c: str, port: str = "") -> pd.Series:
        """Grouper method to group by the country of the components corresponding bus.

        Parameters
        ----------
        n : Network
            PyPSA network instance.
        c : str
            Components type name. E.g. "Generator", "StorageUnit", etc.
        port : str, optional
            Port of corresponding bus, which should be used.

        Returns
        -------
        pd.Series
            Series with the country of the components corresponding bus.

        """
        bus = f"bus{port}"
        component_buses = n.static(c)[bus]
        buses_country = n.buses.country
        return self._map_with_multiindex(component_buses, buses_country).rename(
            "country"
        )

    def location(self, n: Network, c: str, port: str = "") -> pd.Series:
        """Grouper method to group by the location of the components corresponding bus.

        Parameters
        ----------
        n : Network
            PyPSA network instance.
        c : str
            Components type name. E.g. "Generator", "StorageUnit", etc.
        port : str, optional
            Port of corresponding bus, which should be used.

        Returns
        -------
        pd.Series
            Series with the location of the components corresponding bus.

        """
        bus = f"bus{port}"
        component_buses = n.static(c)[bus]
        buses_location = n.buses.location
        return self._map_with_multiindex(component_buses, buses_location).rename(
            "location"
        )

    def unit(self, n: Network, c: str, port: str = "") -> pd.Series:
        """Grouper method to group by the unit of the components corresponding bus.

        Parameters
        ----------
        n : Network
            PyPSA network instance.
        c : str
            Components type name. E.g. "Generator", "StorageUnit", etc.
        port : str, optional
            Port of corresponding bus, which should be used.

        Returns
        -------
        pd.Series
            Series with the unit of the components corresponding bus.

        """
        bus = f"bus{port}"
        component_buses = n.static(c)[bus]
        buses_unit = n.buses.unit
        return self._map_with_multiindex(component_buses, buses_unit).rename("unit")

    def name(self, n: Network, c: str) -> pd.Series:
        """Grouper method to group by the name of components.

        Parameters
        ----------
        n : Network
            PyPSA network instance.
        c : str
            Components type name. E.g. "Generator", "StorageUnit", etc.

        Returns
        -------
        pd.Series
            Series with the component names.

        """
        return n.static(c).index.to_series().rename("name")


groupers = Groupers()

new_grouper_access = {
    "get_carrier": ".carrier",
    "get_bus_carrier": ".bus_carrier",
    "get_bus": ".bus",
    "get_country": ".country",
    "get_location": ".location",
    "get_unit": ".unit",
    "get_name": ".name",
    "get_bus_and_carrier": '["bus", "carrier"]',
    "get_bus_unit_and_carrier": '["bus", "unit", "carrier"]',
    "get_name_bus_and_carrier": '["name", "bus", "carrier"]',
    "get_country_and_carrier": '["country", "carrier"]',
    "get_location_and_carrier": '["location", "carrier"]',
    "get_bus_and_carrier_and_bus_carrier": '["bus", "carrier", "bus_carrier"]',
    "get_carrier_and_bus_carrier": '["carrier", "bus_carrier"]',
}


def deprecated_grouper(func: Callable) -> Callable:
    """Deprecate old grouper methods with custom deprecation warning.

    Parameters
    ----------
    func : Callable
        Function to deprecate.

    Returns
    -------
    Callable
        Same function wrapped with deprecation warning.

    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        msg = (
            f"`n.statistics.{func.__name__}` and `pypsa.statistics.{func.__name__}` "
            f"are deprecated. Use "
            f"`pypsa.statistics.groupers{new_grouper_access[func.__name__]}` instead."
            "Deprecated in version 0.34 and will be removed in version 1.0."
        )
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)

    return wrapper


class DeprecatedGroupers:
    """Grouper class to allow full backwards compatiblity with old grouper methods.

    Allows access to the old grouper methods, points them to new structure on
    module level and raises a DeprecationWarning.
    """

    @deprecated_grouper
    def get_carrier(self, *args: Any, **kwargs: Any) -> pd.Series:
        """Use `pypsa.statistics.groupers.carrier` instead."""
        return groupers.carrier(*args, **kwargs)

    @deprecated_grouper
    def get_bus_carrier(self, *args: Any, **kwargs: Any) -> pd.Series:
        """Use `pypsa.statistics.groupers.bus_carrier` instead."""
        return groupers.bus_carrier(*args, **kwargs)

    @deprecated_grouper
    def get_bus(self, *args: Any, **kwargs: Any) -> pd.Series:
        """Use `pypsa.statistics.groupers.bus` instead."""
        return groupers.bus(*args, **kwargs)

    @deprecated_grouper
    def get_country(self, *args: Any, **kwargs: Any) -> pd.Series:
        """Use `pypsa.statistics.groupers.country` instead."""
        return groupers.country(*args, **kwargs)

    @deprecated_grouper
    def get_unit(self, *args: Any, **kwargs: Any) -> pd.Series:
        """Use `pypsa.statistics.groupers.unit` instead."""
        return groupers.unit(*args, **kwargs)

    @deprecated_grouper
    def get_name(self, *args: Any, **kwargs: Any) -> pd.Series:
        """Use `pypsa.statistics.groupers.name` instead."""
        return groupers.name(*args, **kwargs)

    @deprecated_grouper
    def get_bus_and_carrier(self, *args: Any, **kwargs: Any) -> list:
        """Use `pypsa.statistics.groupers["bus", "carrier"]` instead."""
        return groupers["bus", "carrier"](*args, **kwargs)

    @deprecated_grouper
    def get_bus_unit_and_carrier(self, *args: Any, **kwargs: Any) -> list:
        """Use `pypsa.statistics.groupers["bus", "unit", "carrier"]` instead."""
        return groupers["bus", "unit", "carrier"](*args, **kwargs)

    @deprecated_grouper
    def get_name_bus_and_carrier(self, *args: Any, **kwargs: Any) -> list:
        """Use `pypsa.statistics.groupers["name", "bus", "carrier"]` instead."""
        return groupers["name", "bus", "carrier"](*args, **kwargs)

    @deprecated_grouper
    def get_country_and_carrier(self, *args: Any, **kwargs: Any) -> list:
        """Use `pypsa.statistics.groupers["country", "carrier"]` instead."""
        return groupers["country", "carrier"](*args, **kwargs)

    @deprecated_grouper
    def get_bus_and_carrier_and_bus_carrier(self, *args: Any, **kwargs: Any) -> list:
        """Use `pypsa.statistics.groupers["bus", "carrier", "bus_carrier"]` instead."""
        return groupers["bus", "carrier", "bus_carrier"](*args, **kwargs)

    @deprecated_grouper
    def get_carrier_and_bus_carrier(self, *args: Any, **kwargs: Any) -> list:
        """Use `pypsa.statistics.groupers["carrier", "bus_carrier"]` instead."""
        return groupers["carrier", "bus_carrier"](*args, **kwargs)


deprecated_groupers = DeprecatedGroupers()
