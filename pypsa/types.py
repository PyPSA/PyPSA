from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_list_like


def is_1d_list_like(x: Any) -> bool:
    """
    Check if x is a 1D list-like object.

    Parameters
    ----------
    x : Any
        Object to check.

    Returns
    -------
    bool
        True if x is a 1D list-like object.
    """

    if isinstance(x, np.ndarray):
        return x.ndim == 1

    if isinstance(x, pd.DataFrame):
        return False  # DataFrame has always 2 dimensions

    else:
        return is_list_like(x)
