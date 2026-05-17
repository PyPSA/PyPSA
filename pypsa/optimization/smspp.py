# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Optional SMS++ optimization pipeline integration.

This module provides an accessor `Network.smspp` which runs an external
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
        self.result: (
            SMSPPSolverTool | None
        ) = (  # TODO: to revise type hinting with a Result object when available
            None
        )

    def __call__(
        self,
        solver_options: dict[str, Any] | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> tuple[str, str]:
        """Run the SMS++ pipeline via pypsa2smspp.

        Parameters
        ----------
        solver_options : dict[str, Any] | None, default None
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
        self.optimize(verbose=verbose)
        return self.retrieve_solution(verbose=verbose)

    def create_model(
        self,
        solver_options: dict[str, Any] | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> SMSNetwork:
        """Create the SMS++ model from the bound PyPSA network."""
        _require_smspp_deps()

        import pypsa2smspp  # noqa: PLC0415

        transformation_kwargs = self._resolve_transformation_kwargs(
            solver_options,
            kwargs,
        )
        self.transformation = pypsa2smspp.Transformation(**transformation_kwargs)
        self.result = None

        self.sms_network = self.transformation.create_model(self._n, verbose=verbose)
        return self.sms_network

    def optimize(self, verbose: bool = False) -> SMSPPSolverTool:
        """Optimize the SMS++ model created by :meth:`create_model`."""
        if self.transformation is None:
            msg = "Call `create_model` before `optimize`."
            raise ValueError(msg)

        self.result = self.transformation.optimize(verbose=verbose)
        return self.result

    def retrieve_solution(self, verbose: bool = False) -> tuple[str, str]:
        """Retrieve the SMS++ solution and assign it to the bound PyPSA network."""
        if self.transformation is None:
            msg = "Call `create_model` before `retrieve_solution`."
            raise ValueError(msg)
        if self.result is None:
            msg = "Call `optimize` before `retrieve_solution`."
            raise ValueError(msg)

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
        if "success" in condition.lower():
            return "ok", condition
        return "failed", condition
