# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import pytest

import pypsa


def _values(obj):
    return np.asarray(getattr(obj, "values", obj), dtype=float)


def _simple_network(bus_scaling: float = 1.0) -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots(range(2))
    n.add("Bus", "electricity", scaling=bus_scaling)
    n.add(
        "Generator",
        "generator",
        bus="electricity",
        p_nom_extendable=True,
        capital_cost=1,
        marginal_cost=1,
    )
    n.add("Load", "load", bus="electricity", p_set=[10, 20])
    return n


def test_bus_scaling_is_passed_to_linopy_and_solution_is_unscaled() -> None:
    ref = _simple_network()
    scaled = _simple_network(bus_scaling=0.001)

    for n in (ref, scaled):
        status, condition = n.optimize(solver_name="highs", log_to_console=False)
        assert (status, condition) == ("ok", "optimal")

    np.testing.assert_allclose(
        scaled.c.generators.static.p_nom_opt,
        ref.c.generators.static.p_nom_opt,
    )
    np.testing.assert_allclose(
        scaled.c.generators.dynamic.p,
        ref.c.generators.dynamic.p,
    )
    np.testing.assert_allclose(
        _values(scaled.model.variables["Generator-p"].scaling),
        0.001,
    )
    np.testing.assert_allclose(
        _values(scaled.model.variables["Generator-p_nom"].scaling),
        0.001,
    )
    np.testing.assert_allclose(
        _values(scaled.model.constraints["Bus-nodal_balance"].scaling),
        1000,
    )


def test_multi_port_components_use_bus0_scaling() -> None:
    n = pypsa.Network()
    n.set_snapshots([0])
    n.add("Bus", "source", scaling=0.01)
    n.add("Bus", "sink", scaling=0.001)
    n.add("Generator", "generator", bus="source", p_nom=10, marginal_cost=1)
    n.add("Link", "link", bus0="source", bus1="sink", p_nom=10, efficiency=1)
    n.add("Load", "load", bus="sink", p_set=1)

    n.optimize.create_model()

    np.testing.assert_allclose(_values(n.model.variables["Link-p"].scaling), 0.01)


def test_global_constraint_scaling_is_passed_to_linopy() -> None:
    n = pypsa.Network()
    n.set_snapshots([0])
    n.add("Carrier", "gas", co2_emissions=1)
    n.add("Bus", "bus")
    n.add("Generator", "generator", bus="bus", carrier="gas", p_nom=10, marginal_cost=1)
    n.add("Load", "load", bus="bus", p_set=1)
    n.add(
        "GlobalConstraint",
        "co2_limit",
        type="primary_energy",
        carrier_attribute="co2_emissions",
        sense="<=",
        constant=10,
        scaling=7,
    )

    n.optimize.create_model()

    np.testing.assert_allclose(
        _values(n.model.constraints["GlobalConstraint-co2_limit"].scaling),
        7,
    )


def test_invalid_bus_scaling_raises() -> None:
    n = _simple_network(bus_scaling=0)

    with pytest.raises(ValueError, match="Bus scaling"):
        n.optimize.create_model()
