"""Functionality for contingency analysis, such as branch outages."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from deprecation import deprecated

from pypsa.common import deprecated_common_kwargs

if TYPE_CHECKING:
    from pypsa import Network, SubNetwork

logger = logging.getLogger(__name__)


@deprecated(
    deprecated_in="0.35",
    removed_in="1.0",
    details="Use `n.calculate_BODF` instead.",
)
def calculate_BODF(sub_network: SubNetwork, *args: Any, **kwargs: Any) -> Any:
    """Use `n.calculate_BODF` instead."""
    return sub_network.calculate_BODF(*args, **kwargs)


@deprecated(
    deprecated_in="0.35",
    removed_in="1.0",
    details="Use `n.lpf_contingency` instead.",
)
@deprecated_common_kwargs
def network_lpf_contingency(n: Network, *args: Any, **kwargs: Any) -> Any:
    """Use `n.lpf_contingency` instead."""
    return n.lpf_contingency(*args, **kwargs)
