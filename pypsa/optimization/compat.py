# type: ignore
# ruff: noqa
"""
Use compatibility methods for optimization problem definition with Linopy.

This module intends to make the transition from the native pypsa
optimization code to the linopy implementation easier.
"""

from __future__ import annotations

import linopy
from deprecation import deprecated


@deprecated(
    deprecated_in="0.29",
    removed_in="1.0",
    details="Use native linopy syntax instead.",
)
def get_var(n, c, key):
    """
    Get variables directly from network.
    """
    return n.model[f"{c}-{key}"]


@deprecated(
    deprecated_in="0.29",
    removed_in="1.0",
    details="Use native linopy syntax instead.",
)
def define_variables(
    n, lower, upper, name, attr="", axes=None, spec="", mask=None, **kwargs
):
    """
    Define a variable for a given network.

    This function mimics the behavior of pypsa.linopt.define_variables,
    and was created for compatibility reasons.
    """
    name = f"{name}-{attr}" + (f"-{spec}" if spec else "")
    return n.model.add_variables(
        lower, upper, coords=axes, name=name, mask=mask, **kwargs
    )


@deprecated(
    deprecated_in="0.29",
    removed_in="1.0",
    details="Use native linopy syntax instead.",
)
def define_constraints(
    n, lhs, sense, rhs, name, attr="", axes=None, spec="", mask=None, **kwargs
):
    """
    Define a constraint for a given network.

    This function mimics the behavior of
    pypsa.linopt.define_constraints, and was created for compatibility
    reasons.
    """
    name = f"{name}-{attr}" + (f"-{spec}" if spec else "")
    return n.model.add_constraints(lhs, sense, rhs, name=name, mask=mask, **kwargs)


@deprecated(
    deprecated_in="0.29",
    removed_in="1.0",
    details="Use native linopy syntax instead.",
)
def linexpr(*tuples, as_pandas=True, return_axes=False):
    """
    Define a linear expression.

    This function mimics the behavior of pypsa.linopt.linexpr, and was
    created for compatibility reasons.
    """
    return linopy.LinearExpression.from_tuples(*tuples)


@deprecated(
    deprecated_in="0.29",
    removed_in="1.0",
    details="Use native linopy syntax instead.",
)
def join_exprs(arr, **kwargs):
    """
    Sum over linear expression.

    This function mimics the behavior of pypsa.linopt.join_exprs, and
    was created for compatibility reasons.
    """
    return linopy.LinearExpression.sum(arr, **kwargs)
