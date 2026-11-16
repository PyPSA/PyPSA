# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Resolver and warning class for `pypsa.options.future.<name>` flags.

The warning class lives here, not in `_options.py`, so its identity survives the
`importlib.reload(_options)` that env-var tests run.
"""

import contextlib
import sys
import warnings
from pathlib import Path

import pypsa._options

_PACKAGE_DIR = str(Path(__file__).parent)
_warned: set[str] = set()


class PyPSA2Warning(FutureWarning):
    """Warning for behaviour that changes in PyPSA 2.0.

    Silence it by setting `pypsa.options.future.<name>` to True or False.
    """


def _user_stacklevel() -> int:
    """Return the stacklevel, seen from `future_flag`, of the first non-pypsa frame."""
    frame = sys._getframe(2)
    level = 2
    while frame is not None and (
        frame.f_code.co_filename.startswith(_PACKAGE_DIR)
        or frame.f_code.co_filename == contextlib.__file__
    ):
        frame = frame.f_back  # type: ignore[assignment]
        level += 1
    return level


def future_flag(name: str) -> bool:
    """Return whether the 2.0 behaviour of `pypsa.options.future.<name>` is on.

    An unset flag (None) keeps the current behaviour and warns once per process,
    at the first call that depends on it.
    """
    options = pypsa._options.options
    value = options.get_option(f"future.{name}")
    if value is None and name not in _warned:
        _warned.add(name)
        node = options._get_node(f"future.{name}")
        warning = node.warning if isinstance(node, pypsa._options.FutureOption) else ""
        warnings.warn(
            f"{warning} Set `pypsa.options.future.{name}` to True to opt in now or "
            "to False to keep the current behaviour.",
            PyPSA2Warning,
            stacklevel=_user_stacklevel(),
        )
    return bool(value)
