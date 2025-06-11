"""Python for Power Systems Analysis (PyPSA).

Energy system modelling library.
"""

__author__ = (
    "PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html"
)
__copyright__ = (
    "Copyright 2015-2025 PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html, "
    "MIT License"
)

import warnings
from typing import Any

from pypsa import (
    clustering,
    common,
    components,
    contingency,
    descriptors,
    examples,
    geo,
    io,
    optimization,
    pf,
    plot,
    statistics,
)
from pypsa._options import (
    option_context,
    options,
)
from pypsa.collection import NetworkCollection
from pypsa.components.components import Components
from pypsa.networks import Network, SubNetwork
from pypsa.version import (
    __version__,
    __version_semver__,
    __version_semver_tuple__,
    __version_short__,
    __version_short_tuple__,
)

version = __version__  # Alias for legacy access


def __getattr__(name: str) -> Any:
    if name in ["release_version"]:
        warnings.warn(
            "The attribute 'release_version' is deprecated and will be removed in a future version. "
            "Use '__version_semver__' instead. Deprecated in version 0.35 and will be removed in version 1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
    return __version_semver__


# Module access to options
describe_options = options.describe_options
get_option = options.get_option
set_option = options.set_option


__all__ = [
    "__version__",
    "__version_semver__",
    "__version_short__",
    "__version_semver_tuple__",
    "__version_short_tuple__",
    "version",
    "options",
    "set_option",
    "get_option",
    "describe_options",
    "option_context",
    "clustering",
    "common",
    "components",
    "contingency",
    "descriptors",
    "examples",
    "geo",
    "io",
    "optimization",
    "pf",
    "plot",
    "statistics",
    "Network",
    "NetworkCollection",
    "SubNetwork",
    "Components",
]
