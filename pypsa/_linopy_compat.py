# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Version probes for linopy's v1 arithmetic semantics.

linopy's v1 semantics forbid a `pd.MultiIndex` as a dimension coordinate. Under
v1, PyPSA therefore builds multi-period models over a flat `snapshot` dim (see
`pypsa.optimization.window`). This module detects what the installed linopy
supports and resolves the snapshot representation against it; delete it once v1
is linopy's default and PyPSA requires that release.
"""

from __future__ import annotations

import inspect
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING

import pandas as pd
from linopy import Model
from linopy import options as linopy_options

if TYPE_CHECKING:
    from collections.abc import Iterator

try:
    from linopy.config import LinopySemanticsWarning
except ImportError:  # linopy without the v1 semantics option
    LinopySemanticsWarning = None

# Ragged piecewise curves are stored densely. Only linopy with v1 semantics lets
# the absent slots be declared. Older releases infer them from NaN padding.
SUPPORTS_BREAKPOINT_MASK = (
    "mask" in inspect.signature(Model.add_piecewise_formulation).parameters
)


def linopy_uses_v1() -> bool:
    """Whether linopy's v1 arithmetic semantics are active."""
    try:
        return linopy_options["semantics"] == "v1"
    except KeyError:
        return False


def use_flat_snapshot_index(sns: pd.Index, option: str) -> bool:
    """Whether a model built over `sns` labels `snapshot` with flat tuples.

    Resolves the `pypsa.options.optimization.model_snapshot_index` option
    against linopy's active semantics.
    """
    if option not in ("auto", "flat", "multiindex"):
        msg = (
            f"Invalid snapshot representation '{option}'. Choose 'auto', "
            "'flat' or 'multiindex'."
        )
        raise ValueError(msg)
    if not isinstance(sns, pd.MultiIndex):
        return False
    if option == "auto":
        return linopy_uses_v1()
    if option == "multiindex" and linopy_uses_v1():
        msg = (
            "linopy's v1 semantics forbids a MultiIndex snapshot dimension. "
            "Build with pypsa.options.optimization.model_snapshot_index "
            "= 'flat' (or 'auto') instead."
        )
        raise ValueError(msg)
    return option == "flat"


@contextmanager
def suppress_semantics_warnings() -> Iterator[None]:
    """Silence linopy's legacy-to-v1 divergence notices for the wrapped block.

    PyPSA builds identical models under both semantics, but on legacy linopy the
    build trips a deprecation notice per operation. A no-op under v1 and on
    linopy without the semantics option.
    """
    if LinopySemanticsWarning is None or linopy_uses_v1():
        yield
        return
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=LinopySemanticsWarning)
        yield
