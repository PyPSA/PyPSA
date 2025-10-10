# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Python for Power Systems Analysis (PyPSA).

Energy system modelling library.
"""

__author__ = (
    "PyPSA Developers, see https://docs.pypsa.org/latest/contributing/contributors.html"
)
__copyright__ = (
    "Copyright 2015-2025 PyPSA Developers, see https://docs.pypsa.org/latest/contributing/contributors.html, "
    "MIT License"
)


from typing import NoReturn

from pypsa import (
    clustering,
    common,
    components,
    descriptors,
    examples,
    geo,
    optimization,
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
    __version_base__,
    __version_major_minor__,
)

version = __version__  # Alias for legacy access

# Module access to options
get_option = options.get_option
set_option = options.set_option
reset_option = options.reset_option


__all__ = [
    "__version__",
    "__version_base__",
    "__version_major_minor__",
    "version",
    "options",
    "set_option",
    "get_option",
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


def __getattr__(name: str) -> NoReturn:
    """Handle deprecated version attributes."""
    # Deprecated tuple versions (removed)
    if name == "__version_short_tuple__":
        msg = (
            "pypsa.__version_short_tuple__ has been removed. "
            "Use pypsa.__version_major_minor__ with packaging.version.parse() for version comparisons."
        )
        raise DeprecationWarning(msg)

    if name == "__version_semver_tuple__":
        msg = (
            "pypsa.__version_semver_tuple__ has been removed. "
            "Use pypsa.__version_base__ with packaging.version.parse() for version comparisons."
        )
        raise DeprecationWarning(msg)

    # Deprecated version names (renamed)
    if name == "__version_semver__":
        msg = "pypsa.__version_semver__ is deprecated. Use pypsa.__version_base__ instead."
        raise DeprecationWarning(msg)

    if name == "__version_short__":
        msg = "pypsa.__version_short__ is deprecated. Use pypsa.__version_major_minor__ instead."
        raise DeprecationWarning(msg)

    # Raise AttributeError for all other attributes
    # __getattr__ is only called if the attribute is not found through normal lookup
    msg = f"module '{__name__}' has no attribute '{name}'"
    raise AttributeError(msg)
