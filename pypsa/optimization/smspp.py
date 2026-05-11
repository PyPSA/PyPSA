# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Optional SMS++ optimization pipeline integration.

This module provides an accessor `Network.smspp` which runs an external
optimization pipeline via `pypsa2smspp.Transformation`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from pypsa import Network

logger = logging.getLogger(__name__)


def _require_smspp_deps() -> None:
    """Ensure optional SMS++ dependencies are available at runtime."""
    try:
        import pypsa2smspp  # noqa: F401, PLC0415
        import pysmspp  # noqa: PLC0415

        if not pysmspp.is_smspp_installed():
            logger.warning(
                "pySMSpp detects that SMS++ is not installed. SMS++ pipeline will not work."
            )

    except ImportError as err:
        raise ImportError(
            "SMS++ backend requires optional dependencies (pypsa2smspp and pysmspp)."
            + " Install with: pip install 'pypsa[smspp]'"
        ) from err


class SMSppAccessor:
    """SMS++ accessor for running an external optimization pipeline."""

    def __init__(self, n: Network) -> None:
        """Initialize the accessor with a bound network."""
        self._n = n

    def __call__(
        self,
        config: str | Path | dict[str, Any] | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Network:
        """Run the SMS++ pipeline via pypsa2smspp and return the resulting Network.

        Parameters
        ----------
        config : str | Path | dict[str, Any] | None, default None
            Transformation configuration. If a path is provided, it is interpreted
            as a YAML config file. If a dict is provided, it is forwarded as
            configuration overrides. If None, defaults are used by pypsa2smspp.
        verbose : bool, default False
            Verbosity forwarded to the Transformation runner.
        network : Network | None
            If None, uses the bound network. Otherwise uses the provided one.
        **kwargs : Any
            Reserved for future forwarding.

        Returns
        -------
        Network
            The network returned by the transformation pipeline.

        """
        _require_smspp_deps()

        import pypsa2smspp  # noqa: PLC0415

        logger.info("Running SMS++ pipeline with config: %s", config)
        tr = pypsa2smspp.Transformation(config)
        return tr.run(self._n, verbose=verbose)
