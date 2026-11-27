# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Build PyPSA's linopy model from the spec and the network's data."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from linopy import Model

from pypsa.optimization.spec.checks import check
from pypsa.optimization.spec.layers import (
    _formulation,
    _per_call,
    _repeated,
    _stack,
    _unit_dims,
    spec_names,
    spec_scalings,
)
from pypsa.optimization.spec.naming import _relabel, _renamer, _split_by_label
from pypsa.optimization.spec.scaling import _rules, _scale
from pypsa.optimization.spec.sources import NetworkSources

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

    from pypsa import Network
    from pypsa.optimization.spec.layers import Override

logger = logging.getLogger(__name__)


def build_model(
    n: Network,
    network_index: pd.Index,
    model_index: pd.Index,
    *,
    linearized_unit_commitment: bool = False,
    model_kwargs: dict | None = None,
    piecewise_options: Sequence[Any] = (),
    overrides: Sequence[Override] = (),
    objective_scaling: float = 1.0,
    expressions: bool = False,
    **options: Any,
) -> Model:
    """Build the model of `n` over the build's snapshots from the spec, with its composites, its stored overrides and then *overrides* laid over it.

    Raises:
        mathspec.errors.LanguageError: A layer does not hold, as where an
            override in *overrides* removes an entry a stored override
            removes already.

    """
    from mathspec.errors import LanguageError  # noqa: PLC0415

    check(
        n,
        network_index,
        multi_invest=options.get("multi_invest", False),
        linearized_unit_commitment=linearized_unit_commitment,
        piecewise_options=piecewise_options,
    )
    model_kwargs = model_kwargs or {}
    n.calculate_dependent_values()
    stack, _ = _stack(n, linearized_unit_commitment, piecewise_options)
    per_call = _per_call(overrides)
    try:
        s, expanded = _formulation(linearized_unit_commitment, [*stack, *per_call])
    except LanguageError as e:
        if repeated := _repeated(n, per_call):
            text = "; ".join(repeated)
            msg = f"{text[:1].upper()}{text[1:]}. Drop the removal from `overrides`."
            raise LanguageError(msg) from e
        raise
    if stored := list(n.optimize.overrides):
        logger.info("The build applies the stored overrides %s.", stored)
    sources = NetworkSources(
        n,
        s,
        network_index,
        model_index,
        providers=n.composites.providers(),
        **options,
    )
    mode = (options.get("transmission_losses") or {}).get("mode")
    rename = _renamer(s, spec_names(linearized_unit_commitment), mode)
    m = Model.from_spec(
        expanded,
        sources,
        expressions=expressions,
        rename=rename,
        dim_order=_unit_dims(),
        **model_kwargs,
    )
    _relabel(m, snapshots=model_index, network_index=network_index)
    rules = _rules(spec_scalings(linearized_unit_commitment), rename)
    _scale(m, n, sources, rules, objective_scaling)
    _split_by_label(m)
    return m
