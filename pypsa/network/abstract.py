"""#TODO."""

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
