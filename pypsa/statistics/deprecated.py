"""Deprecated functions for pypsa.statistics."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import pandas as pd

    from pypsa import Network

from deprecation import deprecated

from pypsa.statistics.abstract import AbstractStatisticsAccessor

logger = logging.getLogger(__name__)


@deprecated(
    deprecated_in="0.32",
    removed_in="1.0",
    details="Use n.statistics._get_grouping instead.",
)
def get_grouping(  # noqa
    n: Network,
    c: str,
    groupby: Callable | Sequence[str] | str | bool,
    port: str | None = None,
    nice_names: bool = False,
) -> dict:
    return n.statistics._get_grouping(n, c, groupby, port, nice_names)


@deprecated(
    deprecated_in="0.32",
    removed_in="1.0",
    details="Use n.statistics._aggregate_timeseries instead.",
)
def aggregate_timeseries(  # noqa
    df: pd.DataFrame, weights: pd.Series, agg: str = "sum"
) -> pd.Series:
    return AbstractStatisticsAccessor._aggregate_timeseries(df, weights, agg)


@deprecated(
    deprecated_in="0.32",
    removed_in="1.0",
    details="Use n.statistics._filter_active_assets instead.",
)
def filter_active_assets(  # noqa
    n: Network, c: str, df: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
    return n.statistics._filter_active_assets(n, c, df)


@deprecated(
    deprecated_in="0.32",
    removed_in="1.0",
    details="Use n.statistics._filter_bus_carrier instead.",
)
def filter_bus_carrier(  # noqa
    n: Network,
    c: str,
    port: str,
    bus_carrier: str | Sequence[str] | None,
    df: pd.DataFrame,
) -> pd.DataFrame:
    return n.statistics._filter_bus_carrier(n, c, port, bus_carrier, df)
