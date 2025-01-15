"""
Abstract components module.

Contains classes and logic relevant to all component types in PyPSA.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pypsa import Components


def get_active_assets(
    c: Components,
    investment_period: int | str | Sequence | None = None,
) -> pd.Series:
    """
    Get active components mask of componen type in investment period(s).

    A component is considered active when:
    - it's active attribute is True
    - it's build year + lifetime is smaller than the investment period (if given)

    Parameters
    ----------
    c : pypsa.Components
        Components instance.
    investment_period : int, str, Sequence
        Investment period(s) to check for active within build year and lifetime. If
        none only the active attribute is considered and build year and lifetime are
        ignored. If multiple periods are given the mask is True if component is
        active in any of the given periods.

    Returns
    -------
    pd.Series
        Boolean mask for active components

    """
    if investment_period is None:
        return c.static.active
    if not {"build_year", "lifetime"}.issubset(c.static):
        return c.static.active

    # Logical OR of active assets in all investment periods and
    # logical AND with active attribute
    active = {}
    for period in np.atleast_1d(investment_period):
        if period not in c.n_save.investment_periods:
            raise ValueError("Investment period not in `n.investment_periods`")
        active[period] = c.static.eval("build_year <= @period < build_year + lifetime")
    return pd.DataFrame(active).any(axis=1) & c.static.active
