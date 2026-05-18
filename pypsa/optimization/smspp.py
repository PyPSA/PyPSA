# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Optional SMS++ optimization pipeline integration.

This module provides the `n.optimize.smspp` accessor, which runs an external
optimization pipeline via `pypsa2smspp.Transformation`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pypsa2smspp import Transformation
    from pysmspp import SMSNetwork, SMSPPSolverTool

    from pypsa import Network

logger = logging.getLogger(__name__)


def _require_smspp_deps() -> None:
    """Ensure optional SMS++ dependencies are available at runtime."""
    try:
        import pypsa2smspp  # noqa: F401, PLC0415
        import pysmspp  # noqa: PLC0415
    except ImportError as err:
        raise ImportError(
            "SMS++ backend requires optional dependencies (pypsa2smspp and pysmspp)."
            + " Install with: pip install 'pypsa[smspp]'"
        ) from err

    if not pysmspp.is_smspp_installed():
        logger.warning(
            "pySMSpp detects that SMS++ is not installed. SMS++ pipeline will not work."
        )


class SMSppAccessor:
    """SMS++ accessor for running an external optimization pipeline."""

    def __init__(self, n: Network) -> None:
        """Initialize the accessor with a bound network."""
        self._n = n
        self.transformation: Transformation | None = None
        self.sms_network: SMSNetwork | None = None
        self.result: SMSPPSolverTool | None = None

    def __call__(
        self,
        solver_options: Mapping[str, Any] | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> tuple[str, str]:
        """Run the full SMS++ pipeline via pypsa2smspp.

        Parameters
        ----------
        solver_options : mapping, optional
            Keyword arguments forwarded to ``pypsa2smspp.Transformation``.
        verbose : bool, default False
            Verbosity forwarded to pypsa2smspp stages.
        **kwargs : Any
            Additional keyword arguments forwarded to
            ``pypsa2smspp.Transformation``.

        Returns
        -------
        status : str
            The status of the optimization.
        condition : str
            The termination condition of the optimization.

        """
        self.create_model(solver_options=solver_options, verbose=verbose, **kwargs)
        return self.solve_model(verbose=verbose)

    def create_model(
        self,
        solver_options: Mapping[str, Any] | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> SMSNetwork:
        """Create the SMS++ model from the bound PyPSA network.

        Parameters
        ----------
        solver_options : mapping, optional
            Keyword arguments forwarded to ``pypsa2smspp.Transformation``.
        verbose : bool, default False
            Verbosity forwarded to the pypsa2smspp conversion step.
        **kwargs : Any
            Additional keyword arguments forwarded to
            ``pypsa2smspp.Transformation``.

        """
        _require_smspp_deps()

        import pypsa2smspp  # noqa: PLC0415

        transformation_kwargs = self._resolve_transformation_kwargs(
            solver_options,
            kwargs,
        )
        self.transformation = pypsa2smspp.Transformation(**transformation_kwargs)
        self.sms_network = None
        self.result = None

        self.sms_network = self.transformation.create_model(self._n, verbose=verbose)
        return self.sms_network

    def solve_model(self, verbose: bool = False) -> tuple[str, str]:
        """Solve the SMS++ model and retrieve its solution."""
        if self.transformation is None:
            msg = "No SMS++ transformation is available. Call `n.optimize.smspp.create_model()` first."
            raise ValueError(msg)

        self.result = self.transformation.optimize(verbose=verbose)

        self.transformation.retrieve_solution(
            self._n,
            verbose=verbose,
        )

        return self._status_condition(self.result)

    @staticmethod
    def _resolve_transformation_kwargs(
        solver_options: Mapping[str, Any] | None,
        kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Resolve PyPSA solver options to pypsa2smspp Transformation kwargs."""
        if solver_options is None:
            options: dict[str, Any] = {}
        elif isinstance(solver_options, Mapping):
            options = dict(solver_options)
        else:
            msg = (
                "SMS++ solver_options must be None or a mapping of keyword "
                f"arguments for pypsa2smspp.Transformation; got "
                f"{type(solver_options).__name__}."
            )
            raise TypeError(msg)

        options.update(kwargs)
        return options

    @staticmethod
    def _status_condition(result: Any) -> tuple[str, str]:
        """Map a pypsa2smspp result object to PyPSA's status tuple."""
        if isinstance(result, tuple) and len(result) == 2:
            return str(result[0]), str(result[1])

        status = getattr(result, "status", None)
        if status is None:
            return "ok", "optimal"

        condition = str(status)
        if condition.lower() in {"ok", "optimal"} or "success" in condition.lower():
            return "ok", condition
        return "failed", condition
