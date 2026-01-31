# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Global Sensitivity Analysis (GSA) for PyPSA networks using SALib."""

from __future__ import annotations

import logging
import shutil
import signal
import tempfile
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pypsa import Network
    from pypsa.collection import NetworkCollection

logger = logging.getLogger(__name__)


def generate_gsa_samples(
    parameters: dict,
    n_samples: int,
    method: str = "sobol",
    distributions: str | dict = "uniform",
    calc_second_order: bool = False,
    seed: int | None = None,
) -> tuple[dict, np.ndarray]:
    """Generate GSA samples from PyPSA-style parameter specification.

    This is a convenience function similar to MGA's generate_directions_* functions.
    It converts PyPSA parameter specifications to SALib format and generates samples.

    Parameters
    ----------
    parameters : dict
        Nested dict: {component: {attribute: {name: (min, max)}}}
        Example:
        {
            "Generator": {
                "capital_cost": {
                    "solar": (0.6, 1.0),
                    "wind": (0.8, 1.0),
                }
            },
            "StorageUnit": {
                "capital_cost": {
                    "battery": (0.6, 1.0),
                }
            }
        }
    n_samples : int
        Number of samples to generate (N parameter in SALib)
    method : str
        SALib sampling method: "sobol" (default), "morris", "fast", etc.
    distributions : str | dict
        Distribution type(s). Either:
        - Single string: "uniform" (applied to all parameters)
        - Dict mapping parameter names to distribution types
    calc_second_order : bool
        Whether to calculate second-order indices (only for Sobol)
    seed : int | None
        Random seed for reproducibility

    Returns
    -------
    uncertainty : dict
        SALib problem dict with "num_vars", "names", "bounds", "dists"
    samples : np.ndarray
        Parameter samples with shape (n_samples, n_parameters)

    Examples
    --------
    >>> uncertainty, samples = generate_gsa_samples(
    ...     parameters={
    ...         "Generator": {
    ...             "capital_cost": {
    ...                 "solar": (0.6, 1.0),
    ...                 "wind": (0.8, 1.0),
    ...             }
    ...         }
    ...     },
    ...     n_samples=64,
    ...     method="sobol",
    ...     seed=42,
    ... )
    >>> nc = n.optimize.optimize_gsa(samples, uncertainty)
    """
    # Convert PyPSA parameters to SALib problem dict
    uncertainty = _convert_pypsa_to_salib(parameters, distributions)

    # Generate samples using SALib
    if method == "sobol":
        samples = sobol_sample.sample(
            uncertainty,
            N=n_samples,
            calc_second_order=calc_second_order,
            seed=seed,
        )
    else:
        msg = f"Method '{method}' not yet supported. Only 'sobol' is available."
        raise NotImplementedError(msg)

    return uncertainty, samples


def _convert_pypsa_to_salib(
    parameters: dict,
    distributions: str | dict = "uniform",
) -> dict:
    """Convert PyPSA parameter specification to SALib problem dict.

    Parameters
    ----------
    parameters : dict
        {component: {attribute: {name: (min, max)}}}
    distributions : str | dict
        Distribution type(s)

    Returns
    -------
    dict
        SALib problem dict with "num_vars", "names", "bounds", "dists"
    """
    names = []
    bounds = []
    dists = []

    for component, attrs in parameters.items():
        for attr, specs in attrs.items():
            for name, (min_val, max_val) in specs.items():
                param_name = f"{component}-{attr}-{name}"
                names.append(param_name)
                bounds.append([min_val, max_val])

                # Determine distribution
                if isinstance(distributions, str):
                    dist = distributions
                else:
                    dist = distributions.get(param_name, "uniform")

                # Map to SALib distribution names
                if dist in ["uniform", "unif"]:
                    dists.append("unif")
                elif dist in ["normal", "norm"]:
                    dists.append("norm")
                else:
                    dists.append(dist)

    return {
        "num_vars": len(names),
        "names": names,
        "bounds": bounds,
        "dists": dists,
    }


def _parse_parameter_name(param_name: str) -> tuple[str, str, str]:
    """Parse parameter name to (component, attribute, component_name).

    Parameters
    ----------
    param_name : str
        Format: "Component-attribute-name" or "Component-attribute"

    Returns
    -------
    component : str
        Component type (e.g., "Generator")
    attribute : str
        Attribute name (e.g., "capital_cost")
    component_name : str
        Component name (e.g., "solar") or "" for all components
    """
    parts = param_name.split("-")
    component = parts[0]
    attribute = parts[1]
    component_name = "-".join(parts[2:]) if len(parts) > 2 else ""
    return component, attribute, component_name


def _apply_parameter_sample(
    n: Network,
    sample: np.ndarray,
    uncertainty: dict,
) -> None:
    """Apply parameter sample values to network components.

    Parameters
    ----------
    n : Network
        Network to modify
    sample : np.ndarray
        Array of parameter values (same order as uncertainty["names"])
    uncertainty : dict
        SALib uncertainty dict with "names" key
    """
    for param_name, value in zip(uncertainty["names"], sample):
        component, attr, comp_name = _parse_parameter_name(param_name)

        if comp_name:
            # Modify specific component
            n.c[component].static.loc[comp_name, attr] *= value
        else:
            # Modify all components of this type
            n.c[component].static[attr] *= value


def _worker_init() -> None:
    """Initialize worker processes with proper signal handling."""
    # Ignore SIGINT in worker processes (let parent handle it)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise from workers
        format="[Worker %(process)d] %(levelname)s: %(message)s",
    )


class OptimizationAbstractGSAMixin:
    """Mixin class for GSA (Global Sensitivity Analysis) optimization.

    Class inherits to [pypsa.optimization.OptimizationAccessor][]. All attributes and
    methods can be used within any Network instance via `n.optimize`.
    """

    _n: Network

    def optimize_gsa(
        self,
        samples: np.ndarray,
        uncertainty: dict,
        snapshots: Sequence | None = None,
        multi_investment_periods: bool = False,
        max_parallel: int = 4,
        model_kwargs: dict | None = None,
        **kwargs: Any,
    ) -> NetworkCollection:
        """Run GSA by optimizing network for each parameter sample.

        Parameters
        ----------
        samples : np.ndarray
            Parameter samples from SALib (e.g., sobol.sample())
            Shape: (n_samples, n_parameters)
        uncertainty : dict
            SALib uncertainty dict with at least "names" key
        snapshots : Sequence | None, optional
            Snapshots to optimize. If None, uses all snapshots. Defaults to None.
        multi_investment_periods : bool, default False
            Whether to optimize with multiple investment periods.
        max_parallel : int, default 4
            Number of parallel processes for solving different samples.
        model_kwargs : dict | None, optional
            Keyword arguments used by `linopy.Model`, such as `solver_dir` or `chunk`.
            Defaults to None (uses module default options).
        **kwargs
            Additional arguments passed to `n.optimize()`, such as `solver_name`.

        Returns
        -------
        NetworkCollection
            Collection of optimized networks indexed by sample IDs.
        """
        from pypsa.collection import NetworkCollection

        if snapshots is None:
            snapshots = self._n.snapshots

        if model_kwargs is None:
            from pypsa._options import options

            model_kwargs = options.params.optimize.model_kwargs.copy()

        # Export network to temp NetCDF file (like MGA)
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            fn = f.name

        try:
            self._n.export_to_netcdf(fn)

            # Create temporary directory for sample networks
            temp_dir = tempfile.mkdtemp()

            try:
                # Parallel execution
                with get_context("spawn").Pool(
                    processes=max_parallel,
                    initializer=_worker_init,
                    maxtasksperchild=1,
                ) as pool:
                    results = pool.starmap(
                        OptimizationAbstractGSAMixin._evaluate_single_sample,
                        [
                            (
                                fn,
                                sample,
                                uncertainty,
                                snapshots,
                                multi_investment_periods,
                                model_kwargs,
                                kwargs,
                                temp_dir,
                                i,
                            )
                            for i, sample in enumerate(samples)
                        ],
                    )

                # Filter out failed optimizations and load networks
                networks_list = []
                sample_ids = []

                for i, result_path in enumerate(results):
                    if result_path is not None:
                        try:
                            n_result = self._n.__class__(result_path)
                            networks_list.append(n_result)
                            sample_ids.append(i)
                        except Exception as e:
                            logger.warning(f"Failed to load network from {result_path}: {e}")

                if not networks_list:
                    msg = "All GSA optimizations failed"
                    raise RuntimeError(msg)

                if len(networks_list) < len(samples):
                    logger.warning(
                        "%d out of %d optimizations failed",
                        len(samples) - len(networks_list),
                        len(samples),
                    )

                # Create NetworkCollection
                index = pd.Index(sample_ids, name="sample")

                return NetworkCollection(networks_list, index=index)
            finally:
                # Clean up temporary directory
                if Path(temp_dir).exists():
                    shutil.rmtree(temp_dir)

        finally:
            if Path(fn).exists():
                Path(fn).unlink()

    @staticmethod
    def _evaluate_single_sample(
        fn: str,
        sample: np.ndarray,
        uncertainty: dict,
        snapshots: Sequence,
        multi_investment_periods: bool,
        model_kwargs: dict,
        kwargs: dict,
        temp_dir: str,
        sample_id: int,
    ) -> str | None:
        """Evaluate network at a single parameter sample (worker function).

        Parameters
        ----------
        fn : str
            Path to temporary network file (NetCDF)
        sample : np.ndarray
            Parameter values for this sample
        uncertainty : dict
            SALib uncertainty dict with "names" key
        snapshots : Sequence
            Snapshots to optimize
        multi_investment_periods : bool
            Whether to optimize multiple investment periods
        model_kwargs : dict
            Keyword arguments for linopy.Model
        kwargs : dict
            Keyword arguments for n.optimize()
        temp_dir : str
            Temporary directory to save optimized networks
        sample_id : int
            Sample index for naming the output file

        Returns
        -------
        str | None
            Path to saved network file, or None if optimization failed
        """
        from pypsa.networks import Network

        try:
            # Load network
            n = Network(fn)

            # Apply parameter sample
            _apply_parameter_sample(n, sample, uncertainty)

            # Run optimization
            status, condition = n.optimize(
                snapshots=snapshots,
                multi_investment_periods=multi_investment_periods,
                model_kwargs=model_kwargs,
                **kwargs,
            )

            if status != "ok":
                logger.warning(f"Optimization failed: {status}, {condition}")
                return None

            # Save network to temporary file
            output_path = Path(temp_dir) / f"sample_{sample_id}.nc"
            n.export_to_netcdf(str(output_path))
            return str(output_path)

        except KeyboardInterrupt:
            logger.info("Worker process interrupted")
            return None
        except Exception as e:
            logger.warning(f"Sample evaluation failed: {e}")
            return None
