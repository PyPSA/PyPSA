# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Typing utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_list_like

from pypsa.network.abstract import _NetworkABC

if TYPE_CHECKING:
    from pypsa.components.components import Components

NetworkType = TypeVar("NetworkType", bound=_NetworkABC)
ComponentsLike: TypeAlias = Union[str, "Components"]


def is_1d_list_like(x: Any) -> bool:
    """Check if x is a 1D list-like object.

    Parameters
    ----------
    x : Any
        Object to check.

    Returns
    -------
    bool
        True if x is a 1D list-like object.

    Examples
    --------
    >>> pypsa.types.is_1d_list_like([1, 2, 3])
    True
    >>> pypsa.types.is_1d_list_like(np.array([1, 2, 3]))
    True
    >>> pypsa.types.is_1d_list_like(np.array([[1, 2], [3, 4]]))
    False
    >>> pypsa.types.is_1d_list_like(pd.DataFrame({'a': [1, 2]}))
    False
    >>> pypsa.types.is_1d_list_like(pd.Series([1, 2, 3]))
    True

    """
    if isinstance(x, np.ndarray):
        return x.ndim == 1

    if isinstance(x, pd.DataFrame):
        return False  # DataFrame has always 2 dimensions

    return is_list_like(x)
