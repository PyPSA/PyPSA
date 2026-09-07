# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Constants."""

import importlib.resources
import re
from functools import lru_cache

import pandas as pd

PYPSA_DATA_DIR = importlib.resources.files("pypsa") / "data"

DEFAULT_EPSG = 4326
DEFAULT_TIMESTAMP = "now"
EARTH_RADIUS = 6378137.0  # equitorial radius in meters
HOURS_PER_YEAR = 8760.0

RE_PORTS = re.compile(r"^bus(\d*)$")
# Pattern for filtering bus columns without capture groups
RE_PORTS_FILTER = re.compile(r"^bus\d*$")
# Pattern to get port numbers greater or equal to 2
RE_PORTS_GE_2 = re.compile(r"^bus((?:[2-9]|[1-9]\d+))$")


@lru_cache
def piecewise_attrs(component: str) -> pd.DataFrame:
    """Get the piecewise attribute rows for a component type (empty if none)."""
    df = pd.read_csv(PYPSA_DATA_DIR / "piecewise.csv")
    return df[df.component == component]


@lru_cache
def piecewise_schema(component: str, y: str) -> pd.Series:
    """Get the piecewise schema row for a component type and y attribute (empty if undefined)."""
    df = piecewise_attrs(component)
    return df[df.y == y].squeeze()
