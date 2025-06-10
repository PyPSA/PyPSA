"""Statistics package for PyPSA networks."""

from pypsa.common import pass_empty_series_if_keyerror
from pypsa.statistics.deprecated import (
    aggregate_timeseries,
    filter_active_assets,
    filter_bus_carrier,
    get_grouping,
)
from pypsa.statistics.expressions import (
    StatisticsAccessor,
    get_operation,
    get_transmission_branches,
    get_transmission_carriers,
    get_weightings,
    port_efficiency,
)
from pypsa.statistics.grouping import deprecated_groupers, groupers

# Backwards compatibility
get_carrier = deprecated_groupers.get_carrier
get_bus_carrier = deprecated_groupers.get_bus_carrier
get_bus = deprecated_groupers.get_bus
get_country = deprecated_groupers.get_country
get_unit = deprecated_groupers.get_unit
get_name = deprecated_groupers.get_name
get_bus_and_carrier = deprecated_groupers.get_bus_and_carrier
get_bus_unit_and_carrier = deprecated_groupers.get_bus_unit_and_carrier
get_name_bus_and_carrier = deprecated_groupers.get_name_bus_and_carrier
get_country_and_carrier = deprecated_groupers.get_country_and_carrier
get_bus_and_carrier_and_bus_carrier = (
    deprecated_groupers.get_bus_and_carrier_and_bus_carrier
)
get_carrier_and_bus_carrier = deprecated_groupers.get_carrier_and_bus_carrier

__all__ = [
    "groupers",
    "StatisticsAccessor",
    "get_transmission_branches",
    "get_transmission_carriers",
    "get_weightings",
    "get_operation",
    "port_efficiency",
    # Deprecated
    "pass_empty_series_if_keyerror",
    "get_carrier",
    "get_bus_carrier",
    "get_bus",
    "get_country",
    "get_unit",
    "get_name",
    "get_bus_and_carrier",
    "get_bus_unit_and_carrier",
    "get_name_bus_and_carrier",
    "get_country_and_carrier",
    "get_bus_and_carrier_and_bus_carrier",
    "get_carrier_and_bus_carrier",
    "aggregate_timeseries",
    "filter_active_assets",
    "filter_bus_carrier",
    "get_grouping",
]
