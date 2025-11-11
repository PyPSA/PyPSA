# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Run modelling-to-generate-alternatives (MGA) optimizations."""

from __future__ import annotations

import hashlib
import json
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


def hash_mga(
    n: Network,
    dimensions: dict,
    slack: float = 0.05,
    snapshots: pd.DatetimeIndex | None = None,
    multi_investment_periods: bool = False,
) -> str:
    """Hash the MGA optimization problem for a given network.

    This functions includes network (component) information to ensure cache
    invalidation when the network metadata, topology, components, or snapshots change.

    This hash determines the cache filename - all directions (see `hash_direction`) for the same network configuration will be stored in the same CSV file.

    The hash includes:
    - Network metadata (name, meta dict)
    - Snapshots selection (which time periods to optimize)
    - Multi-investment periods flag
    - Component structure:
        * Component counts (e.g., 585 buses, 1423 generators)
        * Component indices hash (detects added/removed/renamed components)
        * Key feature hashes (detects topology/carrier changes)
    - MGA parameters (dimensions, slack)

    Parameters
    ----------
    n : pypsa.Network
        The network to optimize.
    dimensions : dict
        Dictionary defining the dimensions of the optimization problem.
    slack : float, optional
        Slack value for the optimization, by default 0.05.
    snapshots : pd.DatetimeIndex, optional
        Snapshots to include in optimization. If None, uses all network snapshots.
    multi_investment_periods : bool, optional
        Whether to optimize as multiple investment periods, by default False.

    Returns
    -------
    str
        8-character hash string identifying the network configuration.
        This determines which cache CSV file to use.

    Examples
    --------
    >>> dimensions = {'wind': {'Generator': {'p_nom': 1.0}},
    ...               'solar': {'Generator': {'p_nom': 1.0}}}
    >>> network_hash = hash_mga(n, dimensions, slack=0.05)
    >>> # All directions will be stored in: mga_cache_{network_hash}.csv

    """
    # Use actual snapshots or all network snapshots
    snapshots = snapshots if snapshots is not None else n.snapshots

    # Consider all components
    component_structure = {}
    for c in n.iterate_components():
        if len(c.df) == 0:
            continue

        # Hash the component indices to detect additions/removals
        # Sort indices for order-independent hashing
        index_str = "|".join(sorted(c.df.index.astype(str)))
        index_hash = hashlib.sha256(index_str.encode()).hexdigest()[:16]

        # Hash key features to detect topology/carrier changes
        feature_hashes = {}
        for col in ["carrier", "bus", "bus0", "bus1"]:
            if col in c.df.columns:
                # Sort unique values for consistent hashing
                unique_vals = sorted(c.df[col].dropna().unique().astype(str).tolist())
                col_str = "|".join(unique_vals)
                feature_hashes[col] = hashlib.sha256(col_str.encode()).hexdigest()[:16]

        component_structure[c.name] = {
            "count": len(c.df),
            "index_hash": index_hash,
            "feature_hashes": feature_hashes,
        }

    # Build complete hash input
    hash_input = {
        "network_name": n._name,
        "network_meta": n._meta,
        "snapshots": snapshots.tolist(),
        "multi_investment_periods": multi_investment_periods,
        "component_structure": component_structure,
        "dimensions": dimensions,
        "slack": slack,
    }

    hash_str = json.dumps(hash_input, sort_keys=True, default=str).encode("utf-8")
    hash_value = hashlib.sha256(hash_str).hexdigest()[:8]
    logger.info("Extended hash value for MGA optimization: %s", hash_value)
    logger.debug("Hash input size: %.2f KB", len(hash_str) / 1024)
    return hash_value


def hash_direction(
    direction: dict | pd.Series,
) -> str:
    """Hash a single MGA direction for per-direction caching.

    Creates a unique identifier for a specific direction vector. This hash is used
    as the row index (dir_hash) within the CSV cache file for that network configuration (see `hash_mga`) which determines the filename.

    Parameters
    ----------
    direction : dict | pd.Series
        A single direction vector in the low-dimensional space.
        E.g., {"wind": 0.707, "solar": 0.707}

    Returns
    -------
    str
        16-character hash string identifying this specific direction vector.

    Examples
    --------
    >>> direction = {"wind": 0.707, "solar": 0.707}
    >>> dir_hash = hash_direction(direction)
    >>> # This hash is used as the row index in the CSV file

    """
    # Convert direction to dict if it is a Series
    if isinstance(direction, pd.Series):
        direction_dict = direction.to_dict()
    else:
        direction_dict = direction

    # Hash just the direction vector
    hash_str = json.dumps(direction_dict, sort_keys=True).encode("utf-8")
    hash_value = hashlib.sha256(hash_str).hexdigest()[:16]
    return hash_value


def get_network_cache_file(
    cache_dir: str,
    network_hash: str,
) -> Path:
    """Get the cache file path for a given network configuration.

    Parameters
    ----------
    cache_dir : str
        Directory containing cached results.
    network_hash : str
        Hash identifying the network configuration.

    Returns
    -------
    Path
        Path to the cache file for this network configuration.

    """
    return Path(cache_dir) / f"mga_cache_{network_hash}.csv"


def load_cache_database(
    cache_file: Path,
) -> pd.DataFrame:
    """Load the cache database from a CSV file.

    Parameters
    ----------
    cache_file : Path
        Path to the cache database file.

    Returns
    -------
    pd.DataFrame
        Cache database with all computed directions, or empty DataFrame if file doesn't exist.

    """
    if not cache_file.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(cache_file, index_col="dir_hash")
    except Exception as e:
        logger.warning("Failed to load cache database from %s: %s", cache_file, e)
        return pd.DataFrame()


def save_cache_database(
    cache_file: Path,
    cache_db: pd.DataFrame,
) -> None:
    """Save the cache database to a CSV file.

    Parameters
    ----------
    cache_file : Path
        Path to the cache database file.
    cache_db : pd.DataFrame
        Cache database with all computed directions.

    """
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        cache_db.to_csv(cache_file, index_label="dir_hash")
    except Exception as e:
        logger.warning("Failed to save cache database to %s: %s", cache_file, e)


def get_cached_direction(
    cache_dir: str,
    network_hash: str,
    direction_hash: str,
) -> tuple[dict, pd.Series] | None:
    """Load cached result for a specific direction from the network cache database.

    Parameters
    ----------
    cache_dir : str
        Directory containing cached results.
    network_hash : str
        Hash identifying the network configuration.
    direction_hash : str
        Hash identifying the specific direction.

    Returns
    -------
    tuple[dict, pd.Series] | None
        Tuple of (direction, coordinates) if cached, None otherwise.

    """
    cache_file = get_network_cache_file(cache_dir, network_hash)
    cache_db = load_cache_database(cache_file)

    if cache_db.empty or direction_hash not in cache_db.index:
        return None

    try:
        row = cache_db.loc[direction_hash]

        # Extract direction columns (those starting with "dir_")
        direction_cols = [col for col in cache_db.columns if col.startswith("dir_")]
        direction = {col.replace("dir_", ""): row[col] for col in direction_cols}

        # Extract coordinate columns (those starting with "coord_")
        coord_cols = [col for col in cache_db.columns if col.startswith("coord_")]
        if coord_cols:
            coordinates = pd.Series(
                {col.replace("coord_", ""): row[col] for col in coord_cols}
            )
        else:
            coordinates = None

        logger.debug(
            "Cache HIT: direction %s (network %s)",
            direction_hash[:8],
            network_hash[:8],
        )
        return direction, coordinates  # noqa: TRY300
    except Exception as e:
        logger.warning("Failed to parse cached direction %s: %s", direction_hash, e)
        return None


def save_cached_direction(
    cache_dir: str,
    network_hash: str,
    direction_hash: str,
    direction: dict,
    coordinates: pd.Series,
) -> None:
    """Save result for a specific direction to the network cache database.

    Parameters
    ----------
    cache_dir : str
        Directory to store cached results.
    network_hash : str
        Hash identifying the network configuration.
    direction_hash : str
        Hash identifying the specific direction.
    direction : dict
        The direction vector.
    coordinates : pd.Series
        The computed coordinates in dimension space.

    """
    cache_file = get_network_cache_file(cache_dir, network_hash)
    cache_db = load_cache_database(cache_file)

    # Prepare new row
    new_row = {}

    # Add direction columns with "dir_" prefix
    for key, value in direction.items():
        new_row[f"dir_{key}"] = value

    # Add coordinate columns with "coord_" prefix
    if coordinates is not None:
        for key, value in coordinates.items():
            new_row[f"coord_{key}"] = value

    # Convert to DataFrame row
    new_row_df = pd.DataFrame([new_row], index=[direction_hash])
    new_row_df.index.name = "dir_hash"

    # Add or update in cache database
    if direction_hash in cache_db.index:
        # Update existing row
        cache_db.loc[direction_hash] = new_row_df.loc[direction_hash]
    else:
        # Append new row
        cache_db = pd.concat([cache_db, new_row_df])

    # Save back to file
    save_cache_database(cache_file, cache_db)
    logger.debug(
        "Cache SAVE: direction %s to network cache %s (%d total directions)",
        direction_hash[:8],
        network_hash[:8],
        len(cache_db),
    )


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
                    coeffs = coeffs.reindex(self._n.c[c].extendables).fillna(0)
                    coeffs.index.name = ""
                elif isinstance(coeffs, pd.Series):
                    coeffs = coeffs.reindex(columns=self._n.c[c].static.index).fillna(0)
                elif isinstance(coeffs, pd.DataFrame):
                    coeffs = coeffs.reindex(
                        columns=self._n.c[c].static.index, index=self._n.snapshots
                    ).fillna(0)
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
            `https://`go.pypsa.org/options-params` for more information.
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
            `https://`go.pypsa.org/options-params` for more information.
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
        cache_key: str | None = None,  # Network hash for cache file (optional)
        cache_dir: str | None = None,  # Cache directory (optional)
        snapshots: Sequence | None = None,
        multi_investment_periods: bool = False,
        slack: float = 0.05,
        model_kwargs: dict | None = None,
        kwargs: dict | None = None,
    ) -> tuple[dict, pd.Series | None]:
        """Solve a single direction for parallel execution (helper method).

        Checks cache before solving and saves results after solving.
        Uses a two-level caching approach:
        1. Network hash (network config) determines the cache file
        2. Direction hash (specific direction) determines the entry within that file

        """
        from pypsa.networks import Network  # noqa: PLC0415

        # Load network to compute hashes
        n = Network(fn)

        # Check cache BEFORE solving
        if cache_dir is not None and cache_key is not None:
            # cache_key is the network hash (from hash_mga, without directions)
            network_hash = cache_key

            # Compute direction-specific hash
            direction_hash = hash_direction(direction)

            # Look up in cache database
            cached_result = get_cached_direction(
                cache_dir, network_hash, direction_hash
            )
            if cached_result is not None:
                cached_direction, cached_coordinates = cached_result
                logger.info(
                    "Cache HIT: direction %s from network cache %s",
                    direction_hash[:8],
                    network_hash[:8],
                )
                return (direction, cached_coordinates)
            else:
                logger.debug(
                    "Cache MISS: direction %s not in network cache %s",
                    direction_hash[:8],
                    network_hash[:8],
                )

        # Not in cache, solve it
        try:
            # Handle None values
            if kwargs is None:
                kwargs = {}
            if model_kwargs is None:
                model_kwargs = {}

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
                "Error solving direction %s: %s",
                direction,
                str(e),
            )
            return (direction, None)
        else:
            # Save to cache
            if cache_dir is not None and cache_key is not None:
                network_hash = cache_key
                direction_hash = hash_direction(direction)
                save_cached_direction(
                    cache_dir, network_hash, direction_hash, direction, coordinates
                )
            return (direction, coordinates)

    def optimize_mga_in_multiple_directions(
        self,
        directions: list[dict] | pd.DataFrame,
        dimensions: dict,
        cache_dir: str | None = None,
        snapshots: Sequence | None = None,
        multi_investment_periods: bool = False,
        slack: float = 0.05,
        model_kwargs: dict | None = None,
        max_parallel: int = 4,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run MGA optimization in multiple directions in parallel, reusing existing results if available.

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
        cache_dir : str | None, optional
            Directory to store cached results. If None, caching is disabled.
        snapshots : Sequence | None, optional
            Set of snapshots to consider in the optimization. If None, uses all
            snapshots from the network. Defaults to None.
        multi_investment_periods : bool, default False
            Whether to optimize as a single investment period or to optimize in
            multiple investment periods. Then, snapshots should be a ``pd.MultiIndex``.
        slack : float
            Cost slack for budget constraint. Defaults to 0.05.
        model_kwargs: dict
            Keyword arguments used by `linopy.Model`, such as `solver_dir` or
            `chunk`.
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

        Notes
        -----
        Caching uses a two-level approach:
        1. Network hash (from `hash_mga`) identifies the cache file
        2. Direction hash (from `hash_direction`) identifies entries within that file

        Each network configuration gets ONE cache file:
        - `<cache_dir>/mga_cache_<network_hash>.csv`

        Within that file, all computed directions are stored as a CSV table:
        dir_hash,dir_wind,dir_solar,coord_wind,coord_solar
        abc123,0.707,0.707,100.5,75.3
        def456,-0.707,0.707,95.2,80.1
        ...

        Benefits:
        - Clean cache directory (one file per network config)
        - Per-direction lookup before solving
        - Reuse individual directions across MGA runs
        - Robust recovery from partial failures
        - Easy to inspect (all directions in one place)

        """
        if cache_dir is not None:
            # Ensure cache dir exists
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

            # Compute network hash for cache file
            network_hash = hash_mga(
                self._n,
                dimensions,
                slack,
                snapshots=snapshots,
                multi_investment_periods=multi_investment_periods,
            )

            logger.info("Using network cache: %s", network_hash[:8])
        else:
            logger.info("Caching disabled")
            network_hash = None
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
                                    network_hash,
                                    cache_dir,
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
                # Separate successful results
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
