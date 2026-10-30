# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Numerical scaling of optimisation models via linopy.

Factors are derived from `Bus.scaling` (see `Components.scaling`),
`GlobalConstraint.scaling` and the `objective_scaling` argument and passed to
linopy where variables and constraints are created. linopy transforms
solutions, duals and the objective back to original units. The `scaling`
argument of `n.optimize` chooses these factors automatically.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, ParamSpec, TypeAlias, TypeVar

import numpy as np
import pandas as pd

from pypsa.components.common import invariant

if TYPE_CHECKING:
    from xarray import DataArray

    from pypsa import Network
    from pypsa.components.components import Components

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

KVL = "Kirchhoff-Voltage-Law"

KVL_SCALING = 1e5


@dataclass(frozen=True)
class ScalingFactors:
    """Numerical scaling factors of a network.

    Pass an instance as `scaling` to `n.optimize` or `n.optimize.create_model`,
    or write it to the network via `apply` and pass `objective` as
    `objective_scaling`. See
    [Numerical Scaling](../../user-guide/optimization/numerical-scaling.md).

    Attributes
    ----------
    bus : float | pd.Series
        `Bus.scaling`, a scalar for all buses or a Series indexed by bus name.
    global_constraint : float | pd.Series
        `GlobalConstraint.scaling`, a scalar for all global constraints or a
        Series indexed by global constraint name.
    objective : float
        `objective_scaling`.

    """

    bus: float | pd.Series = 1.0
    global_constraint: float | pd.Series = 1.0
    objective: float = 1.0

    def apply(self, n: Network) -> None:
        """Write `bus` and `global_constraint` to the `scaling` attributes of `n`.

        Parameters
        ----------
        n : pypsa.Network
            Network to write the factors to.

        Raises
        ------
        ValueError
            If a Series lacks factors for some components.

        """
        for c, value in [
            ("Bus", self.bus),
            ("GlobalConstraint", self.global_constraint),
        ]:
            static = n.c[c].static
            if not isinstance(value, pd.Series):
                static["scaling"] = float(value)
                continue
            names = static.index.get_level_values("name")
            missing = names[~names.isin(value.index)].unique()
            if not missing.empty:
                msg = f"{c} scaling factors missing for {list(missing)}."
                raise ValueError(msg)
            static["scaling"] = value.reindex(names).to_numpy(dtype=float)


STANDARD = ScalingFactors(bus=1e-4, global_constraint=1e-6, objective=2e-3)
PRESETS = {"standard": STANDARD}
Strategy: TypeAlias = Callable[["Network"], ScalingFactors]
Scaling: TypeAlias = Literal[False, "standard"] | ScalingFactors | Strategy


def factor(scaling: Components | float, names: pd.Index) -> DataArray | float:
    """Return scaling factors for a model element defined over `names`.

    Parameters
    ----------
    scaling : pypsa.Components | float
        Source of the factors: a component (uses `Components.scaling`) or a
        scalar.
    names : pd.Index
        Names the element is defined over.

    Returns
    -------
    xr.DataArray | float
        Factors over `name`, or `1.0` if all factors are `1` so that unscaled
        models stay unchanged.

    """
    if isinstance(scaling, int | float):
        return float(scaling)
    da = scaling.scaling.sel(name=names)
    if (da == 1).all():
        return 1.0
    return da


def global_constraint_factor(n: Network, name: str) -> float:
    """Return the row factor of the global constraint `name`.

    Parameters
    ----------
    n : pypsa.Network
        Network with `GlobalConstraint.scaling` defined.
    name : str
        Name of the global constraint.

    Returns
    -------
    float
        `GlobalConstraint.scaling` of `name`.

    Raises
    ------
    ValueError
        If the factor is not finite and positive, or differs across scenarios.

    """
    value = float(invariant(n.c.global_constraints.static.scaling)[name])
    if not np.isfinite(value) or value <= 0:
        msg = f"GlobalConstraint.scaling must be finite and positive, got {value} for '{name}'."
        raise ValueError(msg)
    return value


def check_sub_networks(n: Network) -> None:
    """Raise if buses of one sub-network carry different `Bus.scaling` factors.

    Parameters
    ----------
    n : pypsa.Network
        Network with determined topology.

    Raises
    ------
    ValueError
        If buses of one sub-network carry different factors.

    """
    scaling = n.c.buses.scaling.to_series()
    sub_network = invariant(n.c.buses.static.sub_network)
    mixed = scaling.groupby(sub_network).nunique().loc[lambda s: s > 1]
    if not mixed.empty:
        msg = (
            "Buses of one sub-network must share the same `Bus.scaling`, since "
            f"Kirchhoff voltage law rows mix their flows. Mixed: {list(mixed.index)}"
        )
        raise ValueError(msg)


@contextmanager
def applied(n: Network, factors: ScalingFactors) -> Iterator[None]:
    """Apply `factors` to `n` and restore the previous `scaling` attributes on exit.

    Parameters
    ----------
    n : pypsa.Network
        Network to scale temporarily.
    factors : ScalingFactors
        Factors to apply.

    Yields
    ------
    None

    """
    saved = {c: n.c[c].static.scaling.copy() for c in ("Bus", "GlobalConstraint")}
    try:
        factors.apply(n)
        yield
    finally:
        for c, values in saved.items():
            n.c[c].static["scaling"] = values


def reject_scaling(method: str, kwargs: dict, model_kwargs: dict) -> None:
    """Raise if `scaling` is passed to a method that does not support it.

    Parameters
    ----------
    method : str
        Name of the calling method, used in the error message.
    kwargs : dict
        Keyword arguments of the calling method.
    model_kwargs : dict
        Keyword arguments forwarded to `n.optimize.create_model`.

    Raises
    ------
    TypeError
        If `scaling` is in `kwargs` or `model_kwargs`.

    """
    if "scaling" in kwargs or "scaling" in model_kwargs:
        msg = f"`{method}` does not accept `scaling`."
        raise TypeError(msg)


def check_unset(n: Network, objective_scaling: float) -> None:
    """Raise if any factor that `scaling` chooses is already set.

    Parameters
    ----------
    n : pypsa.Network
        Network to check.
    objective_scaling : float
        Objective factor passed alongside `scaling`.

    Raises
    ------
    ValueError
        If `Bus.scaling`, `GlobalConstraint.scaling` or `objective_scaling`
        differ from 1.

    """
    set_factors = [
        f"{c}.scaling"
        for c in ("Bus", "GlobalConstraint")
        if (n.c[c].static.scaling != 1).any()
    ]
    if objective_scaling != 1:
        set_factors.append("objective_scaling")
    if set_factors:
        msg = (
            f"`scaling` chooses all factors itself, but {', '.join(set_factors)} "
            "differ from 1. Reset them to 1 or pass `scaling=False`."
        )
        raise ValueError(msg)


def resolve_factors(
    n: Network, scaling: Scaling, objective_scaling: float
) -> ScalingFactors:
    """Resolve the `scaling` argument into factors.

    Parameters
    ----------
    n : pypsa.Network
        Network to scale, with all factors at 1.
    scaling : Scaling
        A key of `PRESETS`, a `ScalingFactors` instance or a callable
        `f(n) -> ScalingFactors`.
    objective_scaling : float
        Objective factor passed alongside `scaling`.

    Returns
    -------
    ScalingFactors

    Raises
    ------
    TypeError
        If `scaling` is neither a preset, `ScalingFactors` nor callable, or a
        callable does not return `ScalingFactors`.
    ValueError
        If factors are already set or `scaling` is an unknown string.

    """
    if isinstance(scaling, str) and scaling not in PRESETS:
        msg = f"Unknown scaling '{scaling}', use one of {list(PRESETS)}."
        raise ValueError(msg)
    if not (isinstance(scaling, str | ScalingFactors) or callable(scaling)):
        msg = f"scaling must be False, a preset name, ScalingFactors or callable, got {scaling!r}."
        raise TypeError(msg)
    check_unset(n, objective_scaling)
    if isinstance(scaling, str):
        return PRESETS[scaling]
    if isinstance(scaling, ScalingFactors):
        return scaling
    factors = scaling(n)
    if not isinstance(factors, ScalingFactors):
        msg = f"Scaling callable must return ScalingFactors, got {type(factors).__name__}."
        raise TypeError(msg)
    logger.info("Scaling factors: %s", factors)
    return factors


def with_scaling(func: Callable[P, R]) -> Callable[P, R]:
    """Resolve the `scaling` argument of a model builder and apply it during the build.

    Parameters
    ----------
    func : Callable
        `create_model` of the optimization accessor.

    Returns
    -------
    Callable
        Builder that applies the resolved factors to the network for the
        duration of the build and restores them afterwards.

    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> R:
        scaling = kwargs.pop("scaling", False)
        if scaling is False:
            return func(*args, **kwargs)
        n = args[0]._n
        factors = resolve_factors(n, scaling, kwargs.get("objective_scaling", 1.0))
        with applied(n, factors):
            return func(*args, **kwargs | {"objective_scaling": factors.objective})

    return wrapper
