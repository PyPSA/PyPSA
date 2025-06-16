"""Constants."""

import re

DEFAULT_EPSG = 4326
DEFAULT_TIMESTAMP = "now"

RE_PORTS = re.compile(r"^bus(\d*)$")
# Pattern for filtering bus columns without capture groups
RE_PORTS_FILTER = re.compile(r"^bus\d*$")
# Pattern to get port numbers greater or equal to 2
RE_PORTS_GE_2 = re.compile(r"^bus((?:[2-9]|[1-9]\d+))$")
