# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Piecewise-constraint helpers for PyPSA optimization models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

SIGNS_T = Literal["=", "<=", ">="]


# TODO: update to pydantic once pydantic is a pypsa dependency.
@dataclass(eq=True, frozen=True)
class PiecewiseOptions:
    """Options for piecewise constraint formulation."""

    component: str
    """PyPSA component name."""
    attribute: str
    """Component attribute for which piecewise data is defined."""
    sign: SIGNS_T
    """Sign for the piecewise constraint, interpreted as y <sign> f(x)."""
    name: tuple[str, ...] = field(default_factory=tuple)
    """Optional filter for a component name to apply the piecewise constraint to, e.g. a specific generator."""
    method: str = "auto"
    """The method to use for the piecewise constraint formulation, passed to linopy's add_piecewise_formulation method."""
    cumulative_attr: bool = False
    """Whether the y-axis breakpoints represent marginal values which should be integrated to define y breakpoints.

    If True, the integral of the piecewise curve will be used to define y breakpoints.
    If False, the nominal values at each x breakpoint will be used to define y breakpoints."""

    def __post_init__(self) -> None:
        """Ensure that name is stored as a tuple for consistent processing later."""
        list_names = (
            ()
            if self.name is None
            else (self.name,)
            if isinstance(self.name, str)
            else tuple(self.name)
        )
        object.__setattr__(self, "name", list_names)


def get_piecewise_names(c: Any, pw_attr: str, active_names: pd.Index) -> pd.Index:
    """Get component names with a given piecewise attribute.

    Parameters
    ----------
    c : Any
        PyPSA component instance with a ``piecewise`` attribute.
    pw_attr : str
        The piecewise attribute to look for, e.g. ``"marginal_cost"`` or ``"capital_cost"``.
    active_names : pd.Index
        The active component names to consider, e.g. ``c.active_assets`` or ``c.extendables``.

    Returns
    -------
    pd.Index:
        If there are piecewise curves defined for the given attribute, returns a pd.Index of component
        names that have piecewise curves and are in ``active_names``.

    """
    piecewise_df = c.piecewise.get(pw_attr)
    if piecewise_df is None or piecewise_df.empty:
        return pd.Index([], name="name")

    return (
        piecewise_df.dropna(how="all", axis=1)
        .columns.unique("name")
        .intersection(active_names)
    )
