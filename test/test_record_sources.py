# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Experiment: the spec model built from a datarecord record, without a network.

Each case writes a network to a record in a store and builds the sources of
the spec from the record alone. The sources and the model must match those
built from the network. See `dev/datarecord-sources.md`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("datarecord")

import mathspec
from datarecord import Workspace
from datarecord.tools.pypsa import PyPSA
from linopy import Model
from test_optimization_spec_equivalence import (
    _committable_extendable,
    _multi_period,
    _scigrid_de,
)

import pypsa
from pypsa.optimization.spec.build import spec
from pypsa.optimization.spec.record_sources import (
    NetworkReadError,
    RecordSources,
    activity_patch,
)
from pypsa.optimization.spec.sources import NetworkSources

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from datarecord import RecordLike

#: The component types whose members are labels of a dim of their own in a record.
AXIS_TYPES = {"Bus", "Carrier", "SubNetwork", "Shape", "LineType", "TransformerType"}


def dated(n: pypsa.Network) -> pypsa.Network:
    """Give integer timesteps a timestamp each, as a record holds only timestamps."""
    steps = n.snapshots.get_level_values(-1)
    if isinstance(steps, pd.DatetimeIndex):
        return n
    stamps = pd.Timestamp("2030-01-01") + pd.to_timedelta(steps.astype(int), unit="h")
    index: pd.Index = pd.DatetimeIndex(stamps)
    if isinstance(n.snapshots, pd.MultiIndex):
        periods = n.snapshots.get_level_values(0)
        index = pd.MultiIndex.from_arrays([periods, stamps], names=n.snapshots.names)
    index.name = "snapshot"
    n._snapshots_data.index = index
    for c in n.components:
        for frame in c.dynamic.values():
            frame.index = index
    return n


def unique(n: pypsa.Network) -> pypsa.Network:
    """Prefix a name two component types share with its type, as a record holds each name once."""
    seen: set[str] = set()
    for c in n.components:
        if c.name in AXIS_TYPES:
            continue
        clash = [name for name in c.names if name in seen]
        if clash:
            c.rename_component_names(**{name: f"{c.name} {name}" for name in clash})
        seen |= set(c.names)
    return n


def cut(n: pypsa.Network, snapshots: int) -> pypsa.Network:
    n.set_snapshots(n.snapshots[:snapshots])
    return n


def ac_dc_stochastic() -> pypsa.Network:
    n = pypsa.examples.ac_dc_meshed()
    n.set_scenarios({"low": 0.3, "high": 0.7})
    p_set = n.c.loads.dynamic.p_set
    p_set.loc[:, "high"] = (p_set.loc[:, "high"] * 1.2).to_numpy()
    return n


def ac_dc_inactive_unit() -> pypsa.Network:
    n = pypsa.examples.ac_dc_meshed()
    n.c.generators.static.loc["Frankfurt Wind", "active"] = False
    return n


def multi_period_expansion_limits() -> pypsa.Network:
    n = _multi_period()
    limits = {
        "line_volume": ("transmission_volume_expansion_limit", 150, "<=", "AC", None),
        "line_cost": ("transmission_expansion_cost_limit", 4_000, "==", "AC", 2030),
        "capacity": ("tech_capacity_expansion_limit", 300, "==", "gencarrier", 2020),
    }
    for name, (kind, constant, sense, carrier, period) in limits.items():
        n.add(
            "GlobalConstraint",
            name,
            type=kind,
            constant=constant,
            sense=sense,
            carrier_attribute=carrier,
            investment_period=period,
        )
    return n


def committable_extendable() -> pypsa.Network:
    """A committable and extendable unit without a capacity limit, so its big-M is inferred."""
    n = dated(_committable_extendable())
    n.add("Carrier", "AC")
    n.c.generators.static.loc["flex", "p_nom_max"] = np.inf
    return n


def first_lines(n: pypsa.Network) -> pd.Index:
    return n.c.lines.static.index[:2]


#: Each case with its build options, one a function of the network where it names units.
CASES: dict[str, tuple[Callable[[], pypsa.Network], dict[str, Any]]] = {
    "ac_dc_meshed": (pypsa.examples.ac_dc_meshed, {}),
    "ac_dc_stochastic": (ac_dc_stochastic, {}),
    "ac_dc_inactive_unit": (ac_dc_inactive_unit, {}),
    "storage_hvdc": (lambda: unique(pypsa.examples.storage_hvdc()), {}),
    "scigrid_de": (lambda: unique(cut(_scigrid_de(), 4)), {}),
    "scigrid_de_outages": (
        lambda: unique(cut(_scigrid_de(), 1)),
        {"branch_outages": first_lines},
    ),
    "multi_period": (lambda: dated(_multi_period()), {}),
    "multi_period_expansion_limits": (
        lambda: dated(multi_period_expansion_limits()),
        {},
    ),
    "multi_period_storage_stochastic": (
        lambda: dated(_multi_period(storage=True, scenarios=True)),
        {},
    ),
    "stochastic_network": (lambda: cut(pypsa.examples.stochastic_network(), 24), {}),
    "model_energy": (lambda: cut(pypsa.examples.model_energy(), 24), {}),
    "unit_commitment_extendable": (committable_extendable, {}),
}


@pytest.fixture(scope="module", params=CASES)
def case(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[NetworkSources, RecordSources]:
    """The sources of a case, from the network and, in a strict build, from its record in a store."""
    build, options = CASES[request.param]
    n = build()
    n.calculate_dependent_values()
    options = {k: v(n) if callable(v) else v for k, v in options.items()}
    options["multi_invest"] = n.has_investment_periods
    record = stored(n, tmp_path_factory.mktemp(request.param))
    from_network = NetworkSources(n, spec(), n.snapshots, n.snapshots, **options)
    return from_network, RecordSources(record, spec(), strict=True, **options)


def stored(n: pypsa.Network, path: Path) -> RecordLike:
    workspace = Workspace()
    source = PyPSA.to_datarecord(n, workspace.con)
    return workspace.store(str(path)).commit(source, parent=None).record


def assert_same(name: str, expected: Any, found: Any) -> None:
    """Assert two sources hold the same values at the same labels, in any order."""
    if isinstance(expected, pd.Index):
        assert set(expected) == set(found), name
    elif isinstance(expected, pd.DataFrame):
        columns = list(expected.columns)
        frames = [
            f[columns].astype(str).sort_values(columns) for f in (expected, found)
        ]
        assert frames[0].reset_index(drop=True).equals(frames[1].reset_index(drop=True))
    else:
        left, right = xr.DataArray(expected), xr.DataArray(found)
        assert set(left.dims) == set(right.dims), name
        right = right.transpose(*left.dims)
        for dim in left.dims:
            assert set(left.indexes[dim]) == set(right.indexes[dim]), (name, dim)
        right = right.reindex_like(left)
        if "O" in (left.dtype.kind, right.dtype.kind):
            assert (left.astype(str) == right.astype(str)).all(), name
        else:
            np.testing.assert_allclose(
                left.to_numpy().astype(float),
                right.to_numpy().astype(float),
                err_msg=name,
            )


def entries(m: Model) -> dict[str, tuple[tuple[int, ...], int]]:
    """Return the shape and the count of built cells of each variable and constraint."""
    items = [*m.variables.items(), *m.constraints.items()]
    return {k: (v.shape, int((v.labels.values != -1).sum())) for k, v in items}


def solved(s: Any, sources: NetworkSources) -> Model:
    m = Model.from_spec(s, sources, expressions=False)
    status, condition = m.solve(solver_name="highs", output_flag=False)
    assert (status, condition) == ("ok", "optimal")
    return m


def test_sources_match(case: tuple[NetworkSources, RecordSources]) -> None:
    from_network, from_record = case
    for name in from_network:
        assert_same(name, from_network[name], from_record[name])
    assert "network" not in vars(from_record)


def test_model_matches(case: tuple[NetworkSources, RecordSources]) -> None:
    from_network, from_record = case
    expected, found = solved(spec(), from_network), solved(spec(), from_record)
    assert entries(found) == entries(expected)
    assert found.objective.value == pytest.approx(expected.objective.value, rel=1e-9)


def test_strict_build_refuses_a_network_read(tmp_path: Path) -> None:
    record = stored(pypsa.examples.ac_dc_meshed(), tmp_path)
    sources = RecordSources(record, spec(), strict=True)
    with pytest.raises(NetworkReadError, match="reads n.sub_networks"):
        sources.n.sub_networks


@pytest.mark.parametrize("name", ["multi_period", "ac_dc_inactive_unit"])
def test_activity_as_a_condition(name: str, tmp_path: Path) -> None:
    n = CASES[name][0]()
    n.calculate_dependent_values()
    options = {"multi_invest": n.has_investment_periods}
    expected = solved(
        spec(), NetworkSources(n, spec(), n.snapshots, n.snapshots, **options)
    )
    patched = mathspec.override(spec(), [activity_patch(spec())])
    assert "Generator_active" in patched.conditions
    assert "Generator_active" not in patched.parameters
    found = solved(patched, RecordSources(stored(n, tmp_path), patched, **options))
    assert entries(found) == entries(expected)
    assert found.objective.value == pytest.approx(expected.objective.value, rel=1e-9)
