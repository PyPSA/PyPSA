"""
Abstract components module.
Contains classes and properties relevant to all component types in PyPSA. Also imports
logic from other modules:
- components.types
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable

import pandas as pd

from pypsa.components.store import ComponentsStore


class _NetworkABC(ABC):
    _snapshot_weightings: pd.DataFrame
    _investment_period_weightings: pd.DataFrame
    all_components: set[str]
    dynamic: Callable
    components: ComponentsStore
