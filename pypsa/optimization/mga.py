# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Run modelling-to-generate-alternatives (MGA) optimizations."""

from __future__ import annotations

import logging
import signal
import tempfile
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING, Any

import linopy
import numpy as np
import pandas as pd
from linopy import LinearExpression, QuadraticExpression, merge
from scipy.stats.qmc import Halton

from pypsa._options import options
from pypsa.descriptors import nominal_attrs

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pypsa import Network
logger = logging.getLogger(__name__)


def generate_directions_random(
    keys: Sequence[str],
    n_directions: int,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate random directions for MGA in a given low-dimensional space.

    The directions are normalized to unit vectors.

    Parameters
    ----------
    keys : Sequence[str]
        A sequence of strings representing the keys (dimensions) for which to
        generate random directions.
    n_directions : int
        The number of random directions to generate.
    seed : int | None, optional
        A seed for the random number generator to ensure reproducibility.
        If None, a random seed will be used. Defaults to None.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the generated random directions, where each row
        represents a direction and each column corresponds to a key from `keys`.

    """
    # Use numpy with random seed 0 to generate a random array with
    # `len(keys)` columns and `n_directions` rows.
    directions = (
        np.random.RandomState(seed).uniform(size=(n_directions, len(keys))) - 0.5
    )
    # Normalize lengths of row vectors to 1
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    # Return as dataframe with keys as columns
    return pd.DataFrame(directions, columns=keys)


def generate_directions_evenly_spaced(
    keys: Sequence[str],
    n_directions: int,
) -> pd.DataFrame:
    """Generate evenly spaced directions in a 2D space.

    This function generates directions that are uniformly distributed on a unit circle.
    It only supports exactly two keys (dimensions).

    Parameters
    ----------
    keys : Sequence[str]
        A sequence of exactly two strings representing the keys (dimensions) for which to
        generate evenly spaced directions.
    n_directions : int
        The number of evenly spaced directions to generate.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the generated evenly spaced directions, where each row
        represents a direction and each column corresponds to a key from `keys`.

    See Also
    --------
    [pypsa.optimization.mga.generate_directions_random][]

    """
    # Check that there are exactly two keys
    if len(keys) != 2:
        msg = "This function only supports two keys for 2D space."
        raise ValueError(msg)
    # Generate evenly spaced directions in 2D space
    angles = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    return pd.DataFrame(directions, columns=keys)


def generate_directions_halton(
    keys: Sequence[str],
    n_directions: int,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate directions using a Halton sequence for MGA in a given low-dimensional space.

    The directions are normalized to unit vectors, providing a quasi-random distribution
    that tends to fill the space more uniformly than a purely random sequence.

    Parameters
    ----------
    keys : Sequence[str]
        A sequence of strings representing the keys (dimensions) for which to
        generate directions.
    n_directions : int
        The number of directions to generate.
    seed : int | None, optional
        A seed for the random number generator to ensure reproducibility when
        initializing the Halton sampler. If None, a random seed will be used.
        Defaults to None.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the generated directions, where each row
        represents a direction and each column corresponds to a key from `keys`.

    See Also
    --------
    [pypsa.optimization.mga.generate_directions_random][]

    """
    n = len(keys)
    halton_sampler = Halton(n, rng=np.random.default_rng(seed))
    directions: list[np.ndarray] = []
    while len(directions) < n_directions:
        # Sample a point, then transform from unit cube to cube around origin.
        d = 2 * halton_sampler.random(1) - 1
        # Only take points within the unit hypersphere in order to
        # get a uniform distribution.
        if np.linalg.norm(d) <= 1:
            # Scale to lie on the unit hypersphere.
            directions.append((d / np.linalg.norm(d)).flatten())
    # Return as dataframe with keys as columns
    return pd.DataFrame(directions, columns=keys)


def _convert_to_dict(obj: Any) -> Any:
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="list")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {k: _convert_to_dict(v) for k, v in obj.items()}
    return obj


def _worker_init() -> None:
    """Initialize worker processes with proper signal handling."""
    # Ignore SIGINT in worker processes (let parent handle it)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise from workers
        format="[Worker %(process)d] %(levelname)s: %(message)s",
    )


class OptimizationAbstractMGAMixin:
    """Mixin class for MGA (Models to Generate Alternatives) optimization.

    Class inherits to [pypsa.optimization.OptimizationAccessor][]. All attributes and
    methods can be used within any Network instance via `n.optimize`.

    """

    _n: Network

    def build_linexpr_from_weights(
        self,
        weights: dict,
        model: linopy.Model | None = None,
    ) -> LinearExpression:
        """Build a linopy LinearExpression from the given weights.

        Parameters
        ----------
        weights : dict
            A dictionary specifying the weights for different components and
            their attributes. The structure should be
            `{'component_name': {'attribute_name': coefficients}}`.
            Coefficients can be a float, `pd.Series`, or `pd.DataFrame`.
        model : linopy.Model | None, optional
            The linopy model to use. If None, `self._n.model` is used.
            Defaults to None.

        Returns
        -------
        linopy.LinearExpression
            A linear expression built according to the specified weights.

        """
        m = model or self._n.model
        expr = []
        for c, attrs in weights.items():
            for attr, coeffs in attrs.items():
                if isinstance(coeffs, dict):
                    coeffs = pd.Series(coeffs)
                if attr == nominal_attrs[c] and isinstance(coeffs, pd.Series):
                    coeffs = coeffs.reindex(self._n.c[c].extendables, fill_value=0)
                    coeffs.index.name = ""
                elif isinstance(coeffs, pd.Series):
                    coeffs = coeffs.reindex(columns=self._n.c[c].static.index)
                elif isinstance(coeffs, pd.DataFrame):
                    coeffs = coeffs.reindex(
                        columns=self._n.c[c].static.index, index=self._n.snapshots
                    )
                expr.append(m[f"{c}-{attr}"] * coeffs)
        return merge(expr)

    def _add_near_opt_constraint(
        self,
        multi_investment_periods: bool = False,
        slack: float = 0.05,
    ) -> None:
        """Add a near-optimal cost constraint to the linopy model.

        This constraint ensures that the total cost of the solution remains within
        a certain slack of the previously calculated optimal cost.

        Parameters
        ----------
        multi_investment_periods : bool, default False
            Whether the optimization considers multiple investment periods.
            If True, investment period weightings are applied to cost calculations.
        slack : float, default 0.05
            The percentage by which the total cost is allowed to exceed the
            optimal cost. For example, a slack of 0.05 means the total cost
            must be <= 1.05 * optimal_cost.

        """
        # Check that the network has a model and that it has been solved
        n = self._n
        if not n.is_solved:
            msg = "Network needs to be solved with `n.optimize()` before adding near-optimal constraint."
            raise ValueError(msg)
        # Find optimal costs and fixed costs
        if not multi_investment_periods:
            optimal_cost = n.statistics.capex().sum() + n.statistics.opex().sum()
            fixed_cost = n.statistics.installed_capex().sum()
        else:
            w = n.investment_period_weightings.objective
            optimal_cost = (
                n.statistics.capex().sum() * w + n.statistics.opex().sum() * w
            ).sum()
            fixed_cost = (n.statistics.installed_capex().sum() * w).sum()

        # Add constraint
        objective = n.model.objective
        if not isinstance(objective, (LinearExpression | QuadraticExpression)):
            objective = objective.expression

        n.model.add_constraints(
            objective + fixed_cost <= (1 + slack) * optimal_cost, name="budget"
        )

    def optimize_mga(
        self,
        snapshots: Sequence | None = None,
        multi_investment_periods: bool = False,
        weights: dict | None = None,
        sense: str | int = "min",
        slack: float = 0.05,
        model_kwargs: dict | None = None,
        **kwargs: Any,
    ) -> tuple[str, str]:
        """Run modelling-to-generate-alternatives (MGA) on network to find near-optimal solutions.

        Parameters
        ----------
        snapshots : list-like
            Set of snapshots to consider in the optimization. The default is None.
        multi_investment_periods : bool, default False
            Whether to optimise as a single investment period or to optimize in
            multiple investment periods. Then, snapshots should be a
            `pd.MultiIndex`.
        weights : dict-like
            Weights for alternate objective function. The default is None, which
            minimizes generation capacity. The weights dictionary should be keyed
            with the component and variable (see `pypsa/data/variables.csv`), followed
            by a float, dict, pd.Series or pd.DataFrame for the coefficients of the
            objective function.
        sense : str|int
            Optimization sense of alternate objective function. Defaults to 'min'.
            Can also be 'max'.
        slack : float
            Cost slack for budget constraint. Defaults to 0.05.
        model_kwargs : dict, optional
            Keyword arguments used by `linopy.Model`, such as `solver_dir` or `chunk`.
            Defaults to module wide option (default: {}). See
            `https://go.pypsa.org/options-params` for more information.
        **kwargs:
            Keyword argument used by `linopy.Model.solve`, such as `solver_name`,

        Returns
        -------
        status : str
            The status of the optimization, either "ok" or one of the codes listed
            in [linopy.constants.SolverStatus](https://linopy.readthedocs.io/en/latest/generated/linopy.constants.SolverStatus.html).
        condition : str
            The termination condition of the optimization, either
            "optimal" or one of the codes listed in
            [linopy.constants.TerminationCondition](https://linopy.readthedocs.io/en/latest/generated/linopy.constants.TerminationCondition.html)

        """
        # Handle default parameters from options
        if model_kwargs is None:
            model_kwargs = options.params.optimize.model_kwargs.copy()

        n = self._n

        if snapshots is None:
            snapshots = n.snapshots

        if weights is None:
            weights = {
                "Generator": {"p_nom": pd.Series(1, index=n.c.generators.static.index)}
            }

        # check that network has been solved
        if not self._n.is_solved:
            msg = "Network needs to be solved with `n.optimize()` before running MGA."
            raise ValueError(msg)

        # create basic model
        m = n.optimize.create_model(
            snapshots=snapshots,
            multi_investment_periods=multi_investment_periods,
            **model_kwargs,
        )

        # add budget constraint
        self._n.optimize._add_near_opt_constraint(multi_investment_periods, slack)

        # parse optimization sense
        if (
            isinstance(sense, str)
            and sense.startswith("min")
            or isinstance(sense, int)
            and sense > 0
        ):
            sense = 1
        elif (
            isinstance(sense, str)
            and sense.startswith("max")
            or isinstance(sense, int)
            and sense < 0
        ):
            sense = -1
        else:
            msg = f"Could not parse optimization sense {sense}"
            raise ValueError(msg)

        # build alternate objective
        m.objective = self.build_linexpr_from_weights(weights, model=m) * sense

        status, condition = self._n.optimize.solve_model(**kwargs)

        # write MGA coefficients into metadata
        n.meta["slack"] = slack
        n.meta["sense"] = sense
        n.meta["weights"] = _convert_to_dict(weights)

        return status, condition

    def project_solved(
        self,
        dimensions: dict,
    ) -> pd.Series:
        """Project solved model onto low-dimensional space.

        Parameters
        ----------
        dimensions : dict
            A dictionary representing the dimensions of the
            low-dimensional space. The keys are user-defined names for
            the dimensions (matching those in the `direction`
            argument), and the values are dictionaries with the same
            structure as the `weights` argument in `optimize_mga`.

        Returns
        -------
        pd.Series
            A pd.Series representing the coordinates of a solved
            network in the dimensions given by the user. The index
            consists of the keys in the `dimensions` argument; values
            are floats.

        """
        # Check that the network has a solved linopy model
        if not self._n.is_solved:
            msg = "Network needs to be solved with `n.optimize()` before projecting result."
            raise ValueError(msg)
        # Build linear expressions and evaluate them
        return pd.Series(
            {
                key: float(
                    self.build_linexpr_from_weights(
                        dim, model=self._n.model
                    ).solution.sum()
                )
                for key, dim in dimensions.items()
            }
        )

    def optimize_mga_in_direction(
        self,
        direction: dict | pd.Series,
        dimensions: dict,
        snapshots: Sequence | None = None,
        multi_investment_periods: bool = False,
        slack: float = 0.05,
        model_kwargs: dict | None = None,
        **kwargs: Any,
    ) -> tuple[str, str, pd.Series | None]:
        """Run MGA in a given direction in a low-dimension projection.

        Parameters
        ----------
        direction : dict | pd.Series
            A dictionary or pd.Series representing the direction in the low-dimensional space.
            The keys or index are user-defined names for the dimensions, and the values are vector coordinates.
        dimensions : dict
            A dictionary representing the dimensions of the
            low-dimensional space. The keys are user-defined names for
            the dimensions (matching those in the `direction`
            argument), and the values are dictionaries with the same
            structure as the `weights` argument in `optimize_mga`.
        snapshots : Sequence | None, optional
            Set of snapshots to consider in the optimization. If None, uses all
            snapshots from the network. Defaults to None.
        multi_investment_periods : bool, default False
            Whether to optimise as a single investment period or to optimize in
            multiple investment periods. Then, snapshots should be a
            `pd.MultiIndex`.
        slack : float
            Cost slack for budget constraint. Defaults to 0.05.
        model_kwargs : dict, optional
            Keyword arguments used by `linopy.Model`, such as `solver_dir` or `chunk`.
            Defaults to module wide option (default: {}). See
            `https://go.pypsa.org/options-params` for more information.
        **kwargs:
            Keyword argument used by `linopy.Model.solve`, such as `solver_name`,

        Returns
        -------
        status : str
            The status of the optimization, either "ok" or one of the codes listed
            in [linopy.constants.SolverStatus](https://linopy.readthedocs.io/en/latest/generated/linopy.constants.SolverStatus.html).
        condition : str
            The termination condition of the optimization, either
            "optimal" or one of the codes listed in
            [linopy.constants.TerminationCondition](https://linopy.readthedocs.io/en/latest/generated/linopy.constants.TerminationCondition.html)
        coordinates : pd.Series | None
            If the optimization status is "ok", then the final return
            value is a pd.Series representing the coordinates of the
            solved network in dimensions given by the user. The index
            consists of the keys in the `dimensions` argument; values
            are floats. If the optimization status is not "ok", then
            this value is None.

        """
        # Handle default parameters from options
        if model_kwargs is None:
            model_kwargs = options.params.optimize.model_kwargs.copy()

        # Check consistency of `direction` and `dimensions` arguments: keys have to be the same
        if set(direction.keys()) != set(dimensions.keys()):
            msg = (
                "Keys of `direction` and `dimensions` arguments must match. "
                f"Got {set(direction.keys())} and {set(dimensions.keys())}."
            )
            raise ValueError(msg)

        if snapshots is None:
            snapshots = self._n.snapshots

        # check that network has been solved
        if not self._n.is_solved:
            msg = "Network needs to be solved with `n.optimize()` before running MGA."
            raise ValueError(msg)

        # create basic model
        m = self._n.optimize.create_model(
            snapshots=snapshots,
            multi_investment_periods=multi_investment_periods,
            **model_kwargs,
        )

        # build budget constraint
        self._n.optimize._add_near_opt_constraint(multi_investment_periods, slack)

        # Build objective as linear combination of direction and
        # dimensions. Flip the sign in order to maximize in the given
        # direction.
        m.objective = -sum(
            direction[key] * self.build_linexpr_from_weights(dimensions[key], model=m)
            for key in direction.keys()
        )

        status, condition = self._n.optimize.solve_model(**kwargs)
        coordinates = self.project_solved(dimensions) if status == "ok" else None

        # write MGA coefficients into metadata
        self._n.meta["slack"] = slack
        self._n.meta["dimensions"] = _convert_to_dict(dimensions)
        self._n.meta["direction"] = direction

        return status, condition, coordinates

    @staticmethod
    def _solve_single_direction(
        fn: str,
        direction: dict,
        dimensions: dict,
        snapshots: Sequence,
        multi_investment_periods: bool,
        slack: float,
        model_kwargs: dict,
        kwargs: dict,
    ) -> tuple[dict, pd.Series | None]:
        """Solve a single direction for parallel execution (helper method).

        This wrapper is necessary since the network is read from a file in
        this case; also simplifies the return argument management.

        """
        from pypsa.networks import Network  # noqa: PLC0415

        try:
            n = Network(fn)
            _, _, coordinates = n.optimize.optimize_mga_in_direction(
                direction=direction,
                dimensions=dimensions,
                snapshots=snapshots,
                multi_investment_periods=multi_investment_periods,
                slack=slack,
                model_kwargs=model_kwargs,
                **kwargs,
            )
        except KeyboardInterrupt:
            # Handle interruption gracefully
            logger.info("Worker process interrupted")
            return (direction, None)
        except Exception as e:
            # Log error but don't crash the worker
            logger.warning(
                "Error solving in direction",
                extra={"direction": direction, "error": str(e)},
            )
            return (direction, None)
        else:
            return (direction, coordinates)

    def optimize_mga_in_multiple_directions(
        self,
        directions: list[dict] | pd.DataFrame,
        dimensions: dict,
        snapshots: Sequence | None = None,
        multi_investment_periods: bool = False,
        slack: float = 0.05,
        model_kwargs: dict | None = None,
        max_parallel: int = 4,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run MGA optimization in multiple directions in parallel.

        This method performs modelling-to-generate-alternatives (MGA) optimization
        across multiple directions simultaneously using parallel processing. Each
        direction represents a different objective in the low-dimensional projection
        space defined by the dimensions parameter.

        Note that, in order to achieve parallelism, this method exports the network
        to a temporary NetCDF file which is then re-imported in each parallel process.
        This leads to a slight overhead in IO and disk space. The temporary file is
        always cleaned up after the optimization is complete, regardless of whether
        any errors occurred during the optimization.

        Parameters
        ----------
        directions : list[dict] | pd.DataFrame
            Multiple directions in the low-dimensional space. If a list, each element
            should be a dictionary with keys matching those in `dimensions` and values
            representing vector coordinates. If a DataFrame, rows represent directions
            and columns represent dimension names.
        dimensions : dict
            A dictionary representing the dimensions of the low-dimensional space.
            The keys are user-defined names for the dimensions (matching those in the
            `directions` argument), and the values are dictionaries with the same
            structure as the `weights` argument in `optimize_mga`.
        snapshots : Sequence | None, optional
            Set of snapshots to consider in the optimization. If None, uses all
            snapshots from the network. Defaults to None.
        multi_investment_periods : bool, default False
            Whether to optimize as a single investment period or to optimize in
            multiple investment periods. Then, snapshots should be a `pd.MultiIndex`.
        slack : float
            Cost slack for budget constraint. Defaults to 0.05.
        model_kwargs : dict, optional
            Keyword arguments used by `linopy.Model`, such as `solver_dir` or `chunk`.
            Defaults to module wide option (default: {}). See
            `https://go.pypsa.org/options-params` for more information.
        max_parallel : int
            Maximum number of parallel processes to use for solving multiple directions.
            Defaults to 4.
        **kwargs:
            Keyword argument used by `linopy.Model.solve`, such as `solver_name`,

        Returns
        -------
        directions_df : pd.DataFrame
            DataFrame containing the successfully solved directions, where each row
            represents a direction and columns correspond to dimension names.
        coordinates_df : pd.DataFrame
            DataFrame containing the coordinates of each successfully solved network
            in the user-defined dimensions. Rows correspond to solved directions
            and columns to dimension names.

        Examples
        --------
        >>> dimensions = {
        ...     "wind": {"Generator": {"p_nom": {"wind": 1}}},
        ...     "solar": {"Generator": {"p_nom": {"solar": 1}}}
        ... }
        >>> directions = pypsa.optimization.mga.generate_directions_random(["wind", "solar"], 10)
        >>> dirs_df, coords_df = n.optimize.optimize_mga_in_multiple_directions(
        ...     directions, dimensions, max_parallel=2
        ... )
        >>> dirs_df # doctest: +SKIP
                wind     solar
        0  0.958766  0.284198
        1 -0.937432 -0.348170
        2 -0.805652  0.592389
        ...
        >>> coords_df # doctest: +ELLIPSIS
        wind  solar
        0   0.0    0.0
        1   0.0    0.0
        2   0.0    0.0
        ...

        """
        # Handle default parameters from options
        if model_kwargs is None:
            model_kwargs = options.params.optimize.model_kwargs.copy()

        # Iterate over rows of `directions` if a DataFrame
        if isinstance(directions, pd.DataFrame):
            directions = list(directions.T.to_dict().values())

        # Create temporary file to export the network. Note: cannot pass
        # the network as an argument directly since it is not picklable.
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            fn = f.name

        # Wrap in try-finally to ensure the temporary file is deleted
        # even if an error occurs
        try:
            self._n.export_to_netcdf(fn)

            # Use a process pool to solve in parallel
            with (
                get_context("spawn").Pool(
                    processes=max_parallel,
                    initializer=_worker_init,
                    maxtasksperchild=1,  # Kill workers after each task to prevent memory leaks
                ) as pool
            ):
                try:
                    results = pool.starmap(
                        OptimizationAbstractMGAMixin._solve_single_direction,
                        [
                            (
                                fn,
                                direction,
                                dimensions,
                                snapshots,
                                multi_investment_periods,
                                slack,
                                model_kwargs,
                                kwargs,
                            )
                            for direction in directions
                        ],
                    )
                except Exception:
                    # Terminate all workers if something goes wrong
                    pool.terminate()
                    pool.join()
                    raise

            # Separate successful and failed results
            successful = [
                (direction, coords)
                for direction, coords in results
                if coords is not None
            ]
            failed_count = len(results) - len(successful)

            if failed_count > 0:
                logger.warning(
                    "%s out of %s optimizations failed", failed_count, len(results)
                )

            if not successful:
                return pd.DataFrame(), pd.DataFrame()

            successful_directions, successful_coordinates = zip(
                *successful, strict=True
            )
            return (
                pd.DataFrame(successful_directions),
                pd.DataFrame(successful_coordinates),
            )
        finally:
            # Clean up temporary file
            if Path(fn).exists():
                Path(fn).unlink()
