# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Version probes for linopy's v1 arithmetic semantics.

linopy's v1 semantics forbid a `pd.MultiIndex` as a dimension coordinate. Under
v1, PyPSA therefore builds multi-period models over a flat `snapshot` dim (see
`pypsa.optimization.window`). This module detects what the installed linopy
supports and resolves the snapshot representation against it; delete it once v1
is linopy's default and PyPSA requires that release.

linopy 1.0 removed the legacy semantics, the option to choose them and the
frozen `CSRConstraint`. The probes below keep PyPSA working on both 0.10 and
1.0.
"""

from __future__ import annotations

import inspect
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING

import pandas as pd
from linopy import Model, Variable
from linopy import options as linopy_options
from linopy import semantics as linopy_semantics
from linopy.config import LinopySemanticsWarning

if TYPE_CHECKING:
    from collections.abc import Iterator

    import xarray as xr
    from linopy import Constraint

try:
    from linopy.constraints import CSRConstraint
except ImportError:  # linopy>=1.0: a sparse constraint is a mutable `Constraint`
    CSRConstraint = None

# linopy>=1.0 has v1 as its only semantics. Setting the deprecated
# `semantics` option there emits a FutureWarning, so skip it.
V1_ONLY = not hasattr(linopy_semantics, "is_v1")

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


@contextmanager
def v1_semantics() -> Iterator[None]:
    """Activate linopy's v1 semantics for the wrapped block.

    A no-op on linopy>=1.0, where v1 is the only semantics.
    """
    if V1_ONLY:
        yield
        return
    with linopy_options:
        linopy_options.set_value(semantics="v1")
        yield


def is_sparse_constraint(con: Constraint) -> bool:
    """Whether `con` stores its left-hand side in sparse (CSR) form."""
    if CSRConstraint is not None:
        return isinstance(con, CSRConstraint)
    return con.lhs.is_sparse


def set_constraint_scaling(con: Constraint, scaling: xr.DataArray) -> None:
    """Set the row scaling of `con`.

    linopy<1.0 freezes sparse constraints, so they are replaced by a scaled copy.
    """
    if CSRConstraint is None or not isinstance(con, CSRConstraint):
        con.scaling = scaling
        return
    mutable = con.mutable()
    mutable.scaling = scaling
    con.model.constraints.add(mutable.freeze())


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
    build trips a deprecation notice per operation. A no-op under v1.
    """
    if linopy_uses_v1():
        yield
        return
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=LinopySemanticsWarning)
        yield


def flatten_snapshot(var: Variable) -> Variable:
    """Relabel a `snapshot` MultiIndex with its flat `(period, timestep)` tuples.

    Legacy linopy strips the level coords from piecewise breakpoints, so they
    cannot carry a `snapshot` MultiIndex to match the variables. A no-op on a
    flat `snapshot`.
    """
    index = var.indexes.get("snapshot")
    if not isinstance(index, pd.MultiIndex):
        return var
    data = var.data.reset_index("snapshot", drop=True)
    return Variable(
        data.assign_coords(snapshot=index.to_flat_index()), var.model, var.name
    )


def expand_breakpoints(bp: xr.DataArray, var: Variable) -> xr.DataArray:
    """Expand piecewise breakpoints over the dims of `var` they lack.

    linopy before 1.0 cannot expand breakpoints over a dim labelled by tuples,
    such as the flat `snapshot` of multi-period models (PyPSA/linopy#1140).
    """
    missing = [d for d in var.dims if d not in bp.dims]
    return bp.expand_dims({d: var.indexes[d] for d in missing})
