"""General utility functions for PyPSA."""

import warnings

msg = (
    "The module pypsa.components.utils has been moved to 'pypsa.components.common'. "
    "Deprecated in version 0.33 and will be removed in version 1.0."
)
warnings.warn(msg, DeprecationWarning, stacklevel=2)

from pypsa.components.common import *  # noqa: E402, F403
