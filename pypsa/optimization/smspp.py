# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Optional SMS++ optimization pipeline integration.

This module provides an accessor `Network.smspp` which runs an external
optimization pipeline via `pypsa2smspp.Transformation`.
"""

from __future__ import annotations

import logging
from importlib import import_module
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from pypsa import Network

logger = logging.getLogger(__name__)


def _require_smspp_deps() -> None:
    """Ensure optional SMS++ dependencies are available at runtime."""
    missing: list[str] = []

    if find_spec("pypsa2smspp") is None:
        missing.append("pypsa2smspp")

    if find_spec("pysmspp") is None and find_spec("pySMSpp") is None:
        missing.append("pySMSpp (import: pysmspp)")

    if missing:
        raise ImportError(
            "SMS++ backend requires optional dependencies. Missing: "
            + ", ".join(missing)
            + ". Install with: pip install 'pypsa[smspp]'"
        )


class SMSppAccessor:
    """SMS++ accessor for running an external optimization pipeline."""

    def __init__(self, n: Network) -> None:
        """Initialize the accessor with a bound network."""
        self._n = n

    def __call__(
        self,
        config: str | Path | dict[str, Any] | None = None,
        verbose: bool = False,
        network: Network | None = None,
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

        module = import_module("pypsa2smspp")
        Transformation = module.Transformation

        n_in = self._n if network is None else network

        # Do not coerce to str: Transformation now supports None and possibly dict overrides
        logger.info("Running SMS++ pipeline with config: %s", config)
        tr = Transformation(config)
        return tr.run(n_in, verbose=verbose)
