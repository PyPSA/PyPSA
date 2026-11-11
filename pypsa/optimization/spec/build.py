# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Build PyPSA's linopy model from `pypsa.yaml` and the network's data."""

from __future__ import annotations

import functools
import re
from importlib.resources import files
from typing import TYPE_CHECKING, Any

import pandas as pd
import xarray as xr
from linopy import Model
from linopy.constraints import Constraint
from linopy.variables import Variable

from pypsa.optimization.spec.sources import UNITS, NetworkSources

if TYPE_CHECKING:
    from mathspec.spec import Spec

    from pypsa import Network

UNIT_DIMS = {*UNITS.values(), "bus"}
SPEC = files("pypsa.optimization.spec")


@functools.cache
def spec(linearized_unit_commitment: bool = False) -> Spec:
    """The spec of `n.optimize()`, with the linearised unit commitment patch on request."""
    import mathspec

    base = mathspec.to_spec((SPEC / "pypsa.yaml").read_text())
    if not linearized_unit_commitment:
        return base
    return mathspec.override(base, [(SPEC / "linearized_uc.yaml").read_text()])


def pypsa_name(name: str, description: str | None) -> str:
    """The PyPSA name of a spec entry: the first backticked name of its description."""
    head = (description or "").split(" — ", 1)[0]
    found = re.findall(r"`([^`]+)`", head)
    return found[0] if found else name


def _pypsa_dims(dims: tuple[str, ...], has_scenarios: bool) -> dict[str, str]:
    """The renaming of spec dims to PyPSA's: the one unit or bus dim becomes `name`."""
    named = [d for d in dims if d in UNIT_DIMS]
    return {named[0]: "name"} if len(named) == 1 else {}


def _relabel(m: Model, n: Network, names: dict, snapshots: pd.Index) -> None:
    """Rename every variable, constraint and dim of `m` as PyPSA names them.

    Entries without a single row are dropped, as PyPSA does not build them,
    and the placeholder scenario of a network without scenarios is squeezed.
    """
    s_variables = names["variables"]
    s_constraints = names["constraints"]
    squeeze = not n.has_scenarios

    def frame(ds: xr.Dataset) -> xr.Dataset:
        ds = ds.rename(_pypsa_dims(tuple(ds.dims), n.has_scenarios))
        if squeeze and "scenario" in ds.dims:
            ds = ds.squeeze("scenario", drop=True)
        if "snapshot" in ds.dims:
            ds = ds.assign_coords(snapshot=snapshots)
        return ds

    variables = {}
    for name, var in m.variables.items():
        if not (var.labels != -1).any():
            continue
        new = s_variables.get(name, name)
        variables[new] = Variable(frame(var.data), m, new)
    m.variables.data.clear()
    m.variables.data.update(variables)

    constraints = {}
    for name, con in m.constraints.items():
        if not (con.labels != -1).any():
            continue
        new = s_constraints.get(name, name)
        constraints[new] = Constraint.from_dataset(frame(con.data), m, new)
    m.constraints.data.clear()
    m.constraints.data.update(constraints)
    m.variables._label_position_index = None
    m.constraints._label_position_index = None


def build_model(
    n: Network,
    network_index: pd.Index,
    model_index: pd.Index,
    *,
    linearized_unit_commitment: bool = False,
    model_kwargs: dict | None = None,
    **options: Any,
) -> Model:
    """Build the model of `n` over the build's snapshots from the spec."""
    model_kwargs = model_kwargs or {}
    s = spec(linearized_unit_commitment)
    sources = NetworkSources(n, s, network_index, model_index, **options)
    m = Model.from_spec(s, sources, **model_kwargs)
    names = {
        "variables": {k: pypsa_name(k, v.description) for k, v in s.variables.items()},
        "constraints": {
            k: pypsa_name(k, v.description) for k, v in s.constraints.items()
        },
    }
    _relabel(m, n, names, model_index)
    return m
