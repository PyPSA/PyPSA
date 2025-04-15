"""
Components descriptor module.

Contains single helper class (ComponentsDescriptors) which is used to inherit
to Components class. Should not be used directly.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_component_type(*args: Any, **kwargs: Any) -> Any:  # noqa: D103
    msg = (
        "pypsa.components.descriptors.get_component_type is deprecated. "
        "Use c.get_component_type instead."
        "Deprecated in version 0.35 and will be removed in version 1.0."
    )
    raise DeprecationWarning(msg)


class ComponentsDescriptors:
    """
    Helper class for components descriptors methods.

    Class only inherits to Components and should not be used directly.
    """

    def get_active_assets(
        self,
        investment_period: int | str | Sequence | None = None,
    ) -> pd.Series:
        """
        Get active components mask of componen type in investment period(s).

        A component is considered active when:

        - it's active attribute is True
        - it's build year + lifetime is smaller than the investment period (if given)

        Parameters
        ----------
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
            return self.static.active
        if not {"build_year", "lifetime"}.issubset(self.static):
            return self.static.active

        # Logical OR of active assets in all investment periods and
        # logical AND with active attribute
        active = {}
        for period in np.atleast_1d(investment_period):
            if period not in self.n_save.investment_periods:
                raise ValueError("Investment period not in `n.investment_periods`")
            active[period] = self.static.eval(
                "build_year <= @period < build_year + lifetime"
            )
        return pd.DataFrame(active).any(axis=1) & self.static.active
