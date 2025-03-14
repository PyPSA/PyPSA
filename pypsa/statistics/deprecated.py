"""Deprecated functions for pypsa.statistics."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pypsa import Network

import pandas as pd
from deprecation import deprecated

from pypsa.statistics.abstract import AbstractStatisticsAccessor

logger = logging.getLogger(__name__)


@deprecated("Use n.statistics._get_grouping instead.")
def get_grouping(  # noqa
    n: Network,
    c: str,
    groupby: Callable | Sequence[str] | str | bool,
    port: str | None = None,
    nice_names: bool = False,
) -> dict:
    return n.statistics._get_grouping(n, c, groupby, port, nice_names)


@deprecated("Use n.statistics._aggregate_timeseries instead.")
def aggregate_timeseries(  # noqa
    df: pd.DataFrame, weights: pd.Series, agg: str = "sum"
) -> pd.Series:
    return AbstractStatisticsAccessor._aggregate_timeseries(df, weights, agg)


@deprecated("Use n.statistics._filter_active_assets instead.")
def filter_active_assets(  # noqa
    n: Network, c: str, df: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
    return n.statistics._filter_active_assets(n, c, df)


@deprecated("Use n.statistics._filter_bus_carrier instead.")
def filter_bus_carrier(  # noqa
    n: Network,
    c: str,
    port: str,
    bus_carrier: str | Sequence[str] | None,
    df: pd.DataFrame,
) -> pd.DataFrame:
    return n.statistics._filter_bus_carrier(n, c, port, bus_carrier, df)
