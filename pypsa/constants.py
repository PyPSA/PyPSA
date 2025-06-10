"""Constants."""

import re

DEFAULT_EPSG = 4326
DEFAULT_TIMESTAMP = "now"

RE_PORTS = r"^bus(\d*)$"
PATTERN_PORTS = re.compile(RE_PORTS)
# Pattern to get port numbers greater or equal to 2
RE_PORTS_GE_2 = r"^bus([2-9]\d*)$"
PATTERN_PORTS_GE_2 = re.compile(RE_PORTS_GE_2)
