# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Cost calculation utilities for PyPSA."""

from __future__ import annotations

from typing import TypeVar

import numpy as np
import pandas as pd
import xarray as xr

# Type variable for numeric types that support array operations
T = TypeVar("T", float, pd.Series, xr.DataArray)


def annuity(
    discount_rate: T,
    lifetime: T,
) -> T:
    r"""Calculate the annuity factor for given discount rate and lifetime.

    Converts overnight investment cost to annualized cost using the formula:

    .. math::

        \frac{r}{1 - (1 + r)^{-n}}

    For zero discount rate, returns ``1/lifetime`` (simple depreciation).
    For infinite lifetime with positive discount rate, returns the discount rate.

    Parameters
    ----------
    discount_rate : float | pd.Series | xr.DataArray
        Discount rate as decimal (e.g., 0.07 for 7%). Must be non-negative.
    lifetime : float | pd.Series | xr.DataArray
        Asset lifetime in years. Must be positive.

    Returns
    -------
    float | pd.Series | xr.DataArray
        Annuity factor to multiply overnight cost by.

    Raises
    ------
    ValueError
        If discount_rate is negative or lifetime is non-positive.

    Examples
    --------
    >>> import pypsa
    >>> pypsa.costs.annuity(0.07, 25)  # doctest: +ELLIPSIS
    0.0858...
    >>> pypsa.costs.annuity(0.0, 20)  # 0% discount rate = simple depreciation
    0.05

    """
    # Input validation for scalars
    if isinstance(discount_rate, int | float) and discount_rate < 0:
        msg = f"discount_rate must be non-negative, got {discount_rate}"
        raise ValueError(msg)
    if isinstance(lifetime, int | float) and lifetime <= 0:
        msg = f"lifetime must be positive, got {lifetime}"
        raise ValueError(msg)

    # Handle scalar zero discount rate -> simple depreciation (1/lifetime)
    if isinstance(discount_rate, int | float) and discount_rate == 0:
        if isinstance(lifetime, int | float) and np.isinf(lifetime):
            return 0.0  # No cost if infinite lifetime and no interest
        return 1.0 / lifetime

    # Handle infinite lifetime: limit of annuity as n -> inf is r
    if isinstance(lifetime, int | float) and np.isinf(lifetime):
        return discount_rate

    # Standard annuity calculation
    result = discount_rate / (1.0 - 1.0 / (1.0 + discount_rate) ** lifetime)

    # Handle special cases for array types
    if isinstance(discount_rate, pd.Series):
        # Zero discount rate -> 1/lifetime
        zero_rate_mask = discount_rate == 0
        if zero_rate_mask.any():
            result = result.where(~zero_rate_mask, 1.0 / lifetime)
        # Infinite lifetime -> discount_rate
        if isinstance(lifetime, pd.Series):
            inf_mask = np.isinf(lifetime)
            result = result.where(~inf_mask, discount_rate)
        elif np.isinf(lifetime):
            result = discount_rate
    elif isinstance(discount_rate, xr.DataArray):
        # Zero discount rate -> 1/lifetime
        result = xr.where(discount_rate == 0, 1.0 / lifetime, result)
        # Infinite lifetime -> discount_rate
        result = xr.where(np.isinf(lifetime), discount_rate, result)
    elif isinstance(lifetime, pd.Series):
        # discount_rate is scalar, lifetime is Series
        result = result.where(~np.isinf(lifetime), discount_rate)
    elif isinstance(lifetime, xr.DataArray):
        result = xr.where(np.isinf(lifetime), discount_rate, result)

    return result


def annuity_factor(
    discount_rate: T,
    lifetime: T,
) -> T:
    """Calculate annuity factor, returning 1.0 where discount_rate is NaN.

    This is a convenience wrapper around :func:`annuity` that handles the
    common case where ``discount_rate`` is NaN (meaning ``capital_cost`` is
    already annuitized and should be used directly).

    Parameters
    ----------
    discount_rate : float | pd.Series | xr.DataArray
        Discount rate as decimal. If NaN, returns 1.0 (no annuitization needed).
    lifetime : float | pd.Series | xr.DataArray
        Asset lifetime in years.

    Returns
    -------
    float | pd.Series | xr.DataArray
        Annuity factor: ``annuity(discount_rate, lifetime)`` if ``discount_rate``
        is not NaN, otherwise 1.0.

    Examples
    --------
    >>> import pypsa
    >>> import numpy as np
    >>> pypsa.costs.annuity_factor(0.07, 25)  # doctest: +ELLIPSIS
    0.0858...
    >>> pypsa.costs.annuity_factor(np.nan, 25)  # NaN = already annuitized
    1.0
    >>> pypsa.costs.annuity_factor(0.0, 20)  # 0% rate = simple depreciation
    0.05

    """
    if isinstance(discount_rate, xr.DataArray):
        # Use safe value where NaN to avoid calculation errors
        safe_rate = xr.where(np.isnan(discount_rate), 0.07, discount_rate)
        safe_lifetime = xr.where(np.isnan(lifetime), 25.0, lifetime)
        return xr.where(
            np.isnan(discount_rate),
            1.0,
            annuity(safe_rate, safe_lifetime),
        )

    if isinstance(discount_rate, pd.Series):
        is_nan = discount_rate.isna()
        safe_rate = discount_rate.where(~is_nan, 0.07)
        if isinstance(lifetime, pd.Series):
            safe_lifetime = lifetime.where(~lifetime.isna(), 25.0)
        else:
            safe_lifetime = lifetime
        result = annuity(safe_rate, safe_lifetime)
        return result.where(~is_nan, 1.0)

    # Scalar case
    if np.isnan(discount_rate):
        return 1.0
    return annuity(discount_rate, lifetime)


def effective_annual_cost(
    capital_cost: T,
    overnight_cost: T,
    discount_rate: T,
    lifetime: T,
    fom_cost: T | None = None,
) -> T:
    """Calculate effective annual cost from capital or overnight cost.

    This function calculates the total annual fixed cost by:

    1. If ``overnight_cost`` is provided (not NaN): annuitize it using
       ``discount_rate`` and ``lifetime``, then add ``fom_cost``
    2. If ``overnight_cost`` is NaN: use ``capital_cost`` directly
       (already annuitized) and add ``fom_cost``

    Parameters
    ----------
    capital_cost : float | pd.Series | xr.DataArray
        Annuitized investment cost (currency/MW/year or currency/MWh/year).
    overnight_cost : float | pd.Series | xr.DataArray
        Overnight (upfront) investment cost. If NaN, capital_cost is used.
    discount_rate : float | pd.Series | xr.DataArray
        Discount rate as decimal. Used only when overnight_cost is provided.
    lifetime : float | pd.Series | xr.DataArray
        Asset lifetime in years.
    fom_cost : float | pd.Series | xr.DataArray, optional
        Fixed operation and maintenance cost (currency/MW/year). Default 0.

    Returns
    -------
    float | pd.Series | xr.DataArray
        Effective annual cost per unit of capacity.

    Examples
    --------
    >>> import pypsa
    >>> import numpy as np
    >>> # Using overnight cost with discount rate
    >>> pypsa.costs.effective_annual_cost(0, 1000, 0.07, 25, 20)  # doctest: +ELLIPSIS
    105.8...
    >>> # Using capital cost directly (overnight_cost is NaN)
    >>> pypsa.costs.effective_annual_cost(100, np.nan, np.nan, 25, 20)
    120
    >>> # Zero discount rate (simple depreciation)
    >>> pypsa.costs.effective_annual_cost(0, 1000, 0.0, 20, 0)
    50.0

    """
    if isinstance(overnight_cost, xr.DataArray):
        has_overnight = ~np.isnan(overnight_cost)
        # Where overnight_cost is provided, annuitize it; otherwise use capital_cost
        annuitized = xr.where(
            has_overnight,
            overnight_cost * annuity_factor(discount_rate, lifetime),
            capital_cost,
        )
    elif isinstance(overnight_cost, pd.Series):
        has_overnight = overnight_cost.notna()
        ann_factor = annuity_factor(discount_rate, lifetime)
        annuitized = (overnight_cost * ann_factor).where(has_overnight, capital_cost)
    # Scalar case
    elif np.isnan(overnight_cost):
        annuitized = capital_cost
    else:
        annuitized = overnight_cost * annuity_factor(discount_rate, lifetime)

    if fom_cost is None:
        return annuitized

    # Handle NaN in fom_cost (treat as 0)
    if isinstance(fom_cost, xr.DataArray):
        fom_cost = xr.where(np.isnan(fom_cost), 0.0, fom_cost)
    elif isinstance(fom_cost, pd.Series):
        fom_cost = fom_cost.fillna(0.0)
    elif np.isnan(fom_cost):
        fom_cost = 0.0

    return annuitized + fom_cost
