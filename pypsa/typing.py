"""Typing utilities."""

import warnings

msg = (
    "The module pypsa.typing has been moved to 'pypsa.type_utils'. "
    "Deprecated in version 0.35 and will be removed in version 1.0."
)
warnings.warn(msg, DeprecationWarning, stacklevel=2)

from pypsa.type_utils import *  # noqa: E402, F403
