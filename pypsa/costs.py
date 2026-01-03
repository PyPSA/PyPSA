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

    For infinite lifetime, returns the discount rate (the mathematical limit).

    Parameters
    ----------
    discount_rate : float | pd.Series | xr.DataArray
        Discount rate as decimal (e.g., 0.07 for 7%). Must be positive.
    lifetime : float | pd.Series | xr.DataArray
        Asset lifetime in years. Must be positive.

    Returns
    -------
    float | pd.Series | xr.DataArray
        Annuity factor to multiply overnight cost by.

    Raises
    ------
    ValueError
        If discount_rate is non-positive or lifetime is non-positive.

    Note
    ----
    For cases where ``discount_rate=0`` means "already annuitized", use
    :func:`annuity_factor` instead, which returns 1.0 for zero discount rates.

    Examples
    --------
    >>> import pypsa
    >>> pypsa.costs.annuity(0.07, 25)  # 7% discount, 25 year lifetime
    0.0858...

    """
    # Input validation for scalars
    if isinstance(discount_rate, int | float):
        if discount_rate < 0:
            msg = f"discount_rate must be non-negative, got {discount_rate}"
            raise ValueError(msg)
        if discount_rate == 0:
            msg = "discount_rate=0 is undefined for annuity. Use annuity_factor() instead."
            raise ValueError(msg)
    if isinstance(lifetime, int | float) and lifetime <= 0:
        msg = f"lifetime must be positive, got {lifetime}"
        raise ValueError(msg)

    # Handle infinite lifetime: limit of annuity as n -> inf is r
    if isinstance(lifetime, int | float) and np.isinf(lifetime):
        return discount_rate

    # Standard annuity calculation
    result = discount_rate / (1.0 - 1.0 / (1.0 + discount_rate) ** lifetime)

    # Handle infinite lifetime for array types
    if isinstance(lifetime, pd.Series):
        result = result.where(~np.isinf(lifetime), discount_rate)
    elif isinstance(lifetime, xr.DataArray):
        result = xr.where(np.isinf(lifetime), discount_rate, result)

    return result


def annuity_factor(
    discount_rate: T,
    lifetime: T,
) -> T:
    """Calculate annuity factor, returning 1.0 where discount_rate is 0.

    This is a convenience wrapper around :func:`annuity` that handles the
    common case where ``discount_rate=0`` means ``capital_cost`` is already
    annuitized.

    Parameters
    ----------
    discount_rate : float | pd.Series | xr.DataArray
        Discount rate as decimal. If 0, returns 1.0 (no annuitization).
    lifetime : float | pd.Series | xr.DataArray
        Asset lifetime in years.

    Returns
    -------
    float | pd.Series | xr.DataArray
        Annuity factor: ``annuity(discount_rate, lifetime)`` if ``discount_rate > 0``,
        otherwise 1.0.

    Examples
    --------
    >>> import pypsa
    >>> pypsa.costs.annuity_factor(0.07, 25)
    0.0858...
    >>> pypsa.costs.annuity_factor(0, 25)  # Zero discount = already annuitized
    1.0

    """
    if isinstance(discount_rate, xr.DataArray):
        return xr.where(
            discount_rate > 0,
            annuity(xr.where(discount_rate > 0, discount_rate, 1.0), lifetime),
            1.0,
        )

    if isinstance(discount_rate, pd.Series):
        safe_rate = discount_rate.where(discount_rate > 0, 1.0)
        result = annuity(safe_rate, lifetime)
        return result.where(discount_rate > 0, 1.0)

    if discount_rate > 0:
        return annuity(discount_rate, lifetime)
    return 1.0


def effective_annual_cost(
    capital_cost: T,
    discount_rate: T,
    lifetime: T,
    fom_cost: T = 0.0,
) -> T:
    """Calculate effective annual cost from overnight cost, discount rate, and fom_cost.

    This function calculates the total annual fixed cost by:

    1. If ``discount_rate > 0``: annuitizing the overnight ``capital_cost`` using
       the annuity factor, then adding fixed O&M (``fom_cost``)
    2. If ``discount_rate == 0``: treating ``capital_cost`` as already annuitized
       and adding ``fom_cost``

    Parameters
    ----------
    capital_cost : float | pd.Series | xr.DataArray
        Overnight investment cost (currency/MW or currency/MWh).
    discount_rate : float | pd.Series | xr.DataArray
        Discount rate as decimal. If 0, capital_cost is treated as pre-annuitized.
    lifetime : float | pd.Series | xr.DataArray
        Asset lifetime in years.
    fom_cost : float | pd.Series | xr.DataArray, optional
        Fixed operation and maintenance cost (currency/MW/year). Default 0.

    Returns
    -------
    float | pd.Series | xr.DataArray
        Annual cost = capital_cost * annuity_factor(discount_rate, lifetime) + fom_cost.

    Examples
    --------
    >>> import pypsa
    >>> # Overnight cost 1000 EUR/kW, 7% discount, 25 years, 20 EUR/kW/year FOM
    >>> pypsa.costs.effective_annual_cost(1000, 0.07, 25, 20)
    105.8...

    """
    return capital_cost * annuity_factor(discount_rate, lifetime) + fom_cost
