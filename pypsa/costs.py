# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Cost calculation utilities for PyPSA."""

from __future__ import annotations

import numpy as np
import pandas as pd


def annuity(
    discount_rate: float | pd.Series,
    lifetime: float | pd.Series,
) -> float | pd.Series:
    r"""Calculate the annuity factor for given discount rate and lifetime.

    Converts overnight investment cost to an annualized cost using the formula:

    $$\frac{r}{1 - (1 + r)^{-n}}$$

    Special cases:

    - Zero discount rate: returns `1/lifetime` (simple depreciation)
    - Infinite lifetime with r > 0: returns the discount rate
    - Infinite lifetime with r <= 0: returns 0
    - Negative discount rates are allowed (penalizes the present)

    Parameters
    ----------
    discount_rate : float | pd.Series
        Discount rate as decimal (e.g., 0.07 for 7%).
    lifetime : float | pd.Series
        Asset lifetime in years. Must be positive.

    Returns
    -------
    float | pd.Series
        Annual annuity factor to multiply overnight cost by.

    Raises
    ------
    ValueError
        If lifetime is non-positive.

    Examples
    --------
    >>> import pypsa
    >>> pypsa.costs.annuity(0.07, 25)  # doctest: +ELLIPSIS
    0.0858...
    >>> pypsa.costs.annuity(0.0, 20)  # 0% discount rate = simple depreciation
    0.05
    >>> pypsa.costs.annuity(-0.02, 20)  # doctest: +ELLIPSIS
    0.040...

    """
    # Input validation for scalars
    if isinstance(lifetime, int | float) and lifetime <= 0:
        msg = f"lifetime must be positive, got {lifetime}"
        raise ValueError(msg)

    # Handle scalar zero discount rate -> simple depreciation (1/lifetime)
    if isinstance(discount_rate, int | float) and discount_rate == 0:
        if isinstance(lifetime, int | float) and np.isinf(lifetime):
            return 0.0  # No cost if infinite lifetime and no interest
        return 1.0 / lifetime

    # Handle infinite lifetime: limit as n -> inf is r if r > 0, else 0
    if isinstance(lifetime, int | float) and np.isinf(lifetime):
        if isinstance(discount_rate, int | float):
            return max(discount_rate, 0.0)
        return discount_rate.clip(lower=0)

    # Standard annuity calculation
    result = discount_rate / (1.0 - 1.0 / (1.0 + discount_rate) ** lifetime)

    # Handle special cases for pd.Series
    if isinstance(discount_rate, pd.Series):
        # Zero discount rate -> 1/lifetime
        zero_rate_mask = discount_rate == 0
        if zero_rate_mask.any():
            result = result.where(~zero_rate_mask, 1.0 / lifetime)
        # Infinite lifetime -> max(r, 0)
        if isinstance(lifetime, pd.Series):
            inf_mask = np.isinf(lifetime)
            result = result.where(~inf_mask, discount_rate.clip(lower=0))
        elif np.isinf(lifetime):
            result = discount_rate.clip(lower=0)
    elif isinstance(lifetime, pd.Series):
        # discount_rate is scalar, lifetime is Series
        inf_val = max(discount_rate, 0.0)
        result = result.where(~np.isinf(lifetime), inf_val)

    return result


def _has_overnight_cost(overnight_cost: float | pd.Series) -> bool:
    """Check if any overnight cost values are provided (not NaN)."""
    if isinstance(overnight_cost, pd.Series):
        return overnight_cost.notna().any()
    return not np.isnan(overnight_cost)


def periodized_cost(
    capital_cost: float | pd.Series,
    overnight_cost: float | pd.Series,
    discount_rate: float | pd.Series,
    lifetime: float | pd.Series,
    fom_cost: float | pd.Series | None = None,
    nyears: float | pd.Series = 1.0,
) -> float | pd.Series:
    """Calculate fixed costs for the modeled horizon from capital or overnight cost.

    This function calculates the total fixed cost for the modeled horizon by:

    1. If `overnight_cost` is provided (not NaN): annuitize it using
       `discount_rate` and `lifetime`, then scale by `nyears` and add
       `fom_cost`.
    2. If `overnight_cost` is NaN: use `capital_cost` directly
       (already scaled to the model horizon) and add `fom_cost`.

    Parameters
    ----------
    capital_cost : float | pd.Series
        Investment cost per unit of capacity for the modeled horizon.
    overnight_cost : float | pd.Series
        Overnight (upfront) investment cost. If NaN, capital_cost is used.
    discount_rate : float | pd.Series
        Discount rate as decimal. Used only when overnight_cost is provided.
    lifetime : float | pd.Series
        Asset lifetime in years.
    fom_cost : float | pd.Series, optional
        Fixed operation and maintenance cost per unit of capacity for the modeled
        horizon. Default None.
    nyears : float | pd.Series, optional
        Modeled time horizon in years. Used only to scale annuitized overnight costs.
        If provided as a Series indexed by investment period, all values must be
        identical when overnight_cost is used. Default 1.0.

    Returns
    -------
    float | pd.Series
        Fixed cost per unit of capacity for the modeled horizon.

    Raises
    ------
    ValueError
        If overnight_cost is provided and nyears is a Series with varying values.

    """
    use_overnight = _has_overnight_cost(overnight_cost)

    if isinstance(nyears, pd.Series):
        unique_vals = nyears.unique()
        if len(unique_vals) == 1:
            nyears = float(unique_vals[0])
        elif use_overnight:
            msg = (
                "overnight_cost cannot be used when investment periods have "
                "different durations (nyears). Provide capital_cost instead, "
                "or use investment periods with equal duration."
            )
            raise ValueError(msg)

    if use_overnight:
        if isinstance(overnight_cost, pd.Series):
            has_overnight = overnight_cost.notna()
            ann_factor = annuity(discount_rate, lifetime)
            annuitized = overnight_cost * ann_factor * nyears
            base = annuitized.where(has_overnight, capital_cost)
        else:
            base = overnight_cost * annuity(discount_rate, lifetime) * nyears
    else:
        base = capital_cost

    if fom_cost is None:
        return base

    # Handle NaN in fom_cost (treat as 0)
    if isinstance(fom_cost, pd.Series):
        fom_cost = fom_cost.fillna(0.0)
    elif np.isnan(fom_cost):
        fom_cost = 0.0

    return base + fom_cost
