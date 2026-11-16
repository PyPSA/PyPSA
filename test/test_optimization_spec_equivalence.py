# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Equivalence gate of the spec build against the code-built formulation.

Each case solves a fixed network and compares its optimal objective with the
value recorded on `master`. Record the values again on a `master` checkout with
`python test/test_optimization_spec_equivalence.py <commit>`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

import pypsa

if TYPE_CHECKING:
    from collections.abc import Callable

REFERENCE = Path(__file__).parent / "data" / "equivalence" / "objectives.json"
RELATIVE_TOLERANCE = 1e-6
EXACT_MILP = {"mip_rel_gap": 0.0}


def _scigrid_de() -> pypsa.Network:
    n = pypsa.examples.scigrid_de()
    n.c.lines.static.loc[["316", "527", "602"], "s_nom"] = 1715
    return n


def _multi_period(storage: bool = False, scenarios: bool = False) -> pypsa.Network:
    n = pypsa.Network(snapshots=range(10))
    n.investment_periods = [2020, 2030, 2040, 2050]
    n.add("Carrier", ["gencarrier", "AC"])
    n.add("Bus", [1, 2], carrier="AC")
    for i, period in enumerate(n.investment_periods):
        factor = (10 + i) / 10
        n.add(
            "Generator",
            [f"gen1-{period}", f"gen2-{period}"],
            bus=[1, 2],
            lifetime=30,
            build_year=period,
            capital_cost=[100 / factor, 100 * factor],
            marginal_cost=[i + 2, i + 1],
            p_nom_extendable=True,
            carrier="gencarrier",
        )
        n.add(
            "Line",
            f"line-{period}",
            bus0=1,
            bus1=2,
            length=1,
            build_year=period,
            lifetime=40,
            capital_cost=30 + i,
            x=0.0001,
            r=0.001,
            s_nom_extendable=True,
            carrier="AC",
        )
    load = np.tile(np.arange(100.0, 110.0), len(n.investment_periods))
    n.add("Load", ["load1", "load2"], bus=[1, 2])
    n.c.loads.dynamic.p_set = pd.DataFrame(
        {"load1": load, "load2": load}, index=n.snapshots
    )
    if storage:
        n.remove("Generator", n.c.generators.static.query('bus == "1"').index)
        n.c.generators.static.capital_cost *= 5
        for i, period in enumerate(n.investment_periods):
            n.add(
                "StorageUnit",
                f"sto1-{period}",
                bus=1,
                lifetime=30,
                build_year=period,
                capital_cost=100 / (10 + i),
                marginal_cost=i,
                p_nom_extendable=True,
            )
    if scenarios:
        n.set_scenarios({"high": 0.6, "low": 0.4})
        p_set = n.c.loads.dynamic.p_set
        p_set.loc[:, "high"] = (p_set.loc[:, "high"] * 1.3).to_numpy()
        p_set.loc[:, "low"] = (p_set.loc[:, "low"] * 0.7).to_numpy()
    return n


def _single_bus(snapshots: int, load: list[float]) -> pypsa.Network:
    n = pypsa.Network(snapshots=range(snapshots))
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=load)
    return n


def ac_dc_meshed() -> pypsa.Network:
    n = pypsa.examples.ac_dc_meshed()
    n.optimize()
    return n


def ac_dc_meshed_operational_limit() -> pypsa.Network:
    n = pypsa.examples.ac_dc_meshed()
    n.remove("GlobalConstraint", n.c.global_constraints.static.index)
    n.add(
        "GlobalConstraint",
        "gas_limit",
        type="operational_limit",
        carrier_attribute="gas",
        constant=30_000,
    )
    n.optimize()
    return n


def ac_dc_meshed_stochastic() -> pypsa.Network:
    n = pypsa.examples.ac_dc_meshed()
    n.set_scenarios({"low": 0.3, "high": 0.7})
    p_set = n.c.loads.dynamic.p_set
    p_set.loc[:, "high"] = (p_set.loc[:, "high"] * 1.2).to_numpy()
    n.optimize()
    return n


def storage_hvdc() -> pypsa.Network:
    n = pypsa.examples.storage_hvdc()
    n.optimize()
    return n


def storage_hvdc_operational_limit() -> pypsa.Network:
    n = pypsa.examples.storage_hvdc()
    n.remove("GlobalConstraint", n.c.global_constraints.static.index)
    n.add(
        "GlobalConstraint",
        "battery_limit",
        type="operational_limit",
        carrier_attribute="battery",
        constant=5_000,
    )
    n.c.storage_units.static["state_of_charge_initial"] = 1_000
    n.c.storage_units.static.p_nom_extendable = True
    n.c.storage_units.static.cyclic_state_of_charge = False
    n.optimize()
    return n


def scigrid_de() -> pypsa.Network:
    n = _scigrid_de()
    n.optimize(n.snapshots[:4])
    return n


def scigrid_de_security_constrained() -> pypsa.Network:
    n = _scigrid_de()
    outages = n.c.lines.static.index[:2]
    n.optimize.optimize_security_constrained(n.snapshots[0], branch_outages=outages)
    return n


def scigrid_de_transmission_losses() -> pypsa.Network:
    n = _scigrid_de()
    n.c.lines.static.s_max_pu = 0.7
    losses = {"mode": "tangents", "segments": 2}
    n.optimize(n.snapshots[0], transmission_losses=losses)
    return n


def model_energy() -> pypsa.Network:
    n = pypsa.examples.model_energy()
    n.optimize(n.snapshots[:240])
    return n


def stochastic_network() -> pypsa.Network:
    n = pypsa.examples.stochastic_network()
    n.optimize(n.snapshots[:240])
    return n


def multi_period() -> pypsa.Network:
    n = _multi_period()
    n.optimize(multi_investment_periods=True)
    return n


def multi_period_expansion_limits() -> pypsa.Network:
    n = _multi_period()
    n.add(
        "GlobalConstraint",
        "line_volume",
        type="transmission_volume_expansion_limit",
        constant=150,
        sense="<=",
        carrier_attribute="AC",
    )
    n.add(
        "GlobalConstraint",
        "line_cost",
        type="transmission_expansion_cost_limit",
        constant=4_000,
        sense="==",
        carrier_attribute="AC",
        investment_period=2030,
    )
    n.add(
        "GlobalConstraint",
        "generator_capacity",
        type="tech_capacity_expansion_limit",
        constant=300,
        sense="==",
        carrier_attribute="gencarrier",
        investment_period=2020,
    )
    n.optimize(multi_investment_periods=True)
    return n


def multi_period_growth_limit() -> pypsa.Network:
    n = _multi_period()
    n.add("Generator", "backup", bus=1, p_nom=500, marginal_cost=1_000)
    n.c.carriers.static.at["gencarrier", "max_growth"] = 100
    n.c.carriers.static.at["gencarrier", "max_relative_growth"] = 0.5
    n.optimize(multi_investment_periods=True)
    return n


def multi_period_storage_primary_energy() -> pypsa.Network:
    n = _multi_period(storage=True)
    n.add("Carrier", "emitting_carrier", co2_emissions=100)
    storage_units = n.c.storage_units.static
    storage_units["state_of_charge_initial"] = 200
    storage_units["cyclic_state_of_charge"] = False
    storage_units["carrier"] = "emitting_carrier"
    n.add("GlobalConstraint", "co2limit", type="primary_energy", constant=3_000)
    n.optimize(multi_investment_periods=True)
    return n


def multi_period_storage_stochastic() -> pypsa.Network:
    n = _multi_period(storage=True, scenarios=True)
    n.optimize(multi_investment_periods=True)
    return n


def unit_commitment() -> pypsa.Network:
    n = _single_bus(6, [4000, 6000, 5000, 800, 3000, 7000])
    n.add(
        "Generator",
        "coal",
        bus="bus",
        committable=True,
        p_min_pu=0.3,
        marginal_cost=20,
        p_nom=10000,
        min_up_time=3,
        min_down_time=2,
        start_up_cost=5000,
        shut_down_cost=1000,
        ramp_limit_up=0.5,
        ramp_limit_down=0.5,
        ramp_limit_start_up=0.6,
        ramp_limit_shut_down=0.6,
    )
    n.add(
        "Generator",
        "gas",
        bus="bus",
        committable=True,
        p_min_pu=0.1,
        marginal_cost=70,
        p_nom=5000,
        up_time_before=0,
        min_up_time=2,
        start_up_cost=500,
    )
    n.add("Generator", "peak", bus="bus", marginal_cost=500, p_nom=10000)
    n.optimize(solver_options=EXACT_MILP)
    return n


def unit_commitment_link() -> pypsa.Network:
    n = pypsa.Network(snapshots=range(4))
    n.add("Bus", ["gas", "electricity"])
    n.add("Generator", "gas", bus="gas", marginal_cost=10, p_nom=20000)
    n.add(
        "Link",
        "OCGT",
        bus0="gas",
        bus1="electricity",
        committable=True,
        p_min_pu=0.1,
        efficiency=0.5,
        up_time_before=0,
        min_up_time=3,
        start_up_cost=3333,
        p_nom=12000,
    )
    n.add("Generator", "wind", bus="electricity", p_nom=800)
    n.add("Load", "load", bus="electricity", p_set=[4000, 6000, 800, 5000])
    n.optimize(solver_options=EXACT_MILP)
    return n


def _committable_extendable() -> pypsa.Network:
    n = _single_bus(4, [75, 30, 75, 0])
    n.add(
        "Generator",
        "flex",
        bus="bus",
        committable=True,
        p_nom_extendable=True,
        p_min_pu=0.4,
        p_nom_max=100.0,
        marginal_cost=40.0,
        capital_cost=100.0,
        start_up_cost=50.0,
    )
    n.add(
        "Generator",
        "peak",
        bus="bus",
        p_nom_extendable=True,
        p_nom_max=100.0,
        marginal_cost=120.0,
        capital_cost=50.0,
    )
    return n


def unit_commitment_extendable() -> pypsa.Network:
    n = _committable_extendable()
    n.optimize(solver_options=EXACT_MILP)
    return n


def unit_commitment_extendable_linearized() -> pypsa.Network:
    n = _committable_extendable()
    n.optimize(linearized_unit_commitment=True)
    return n


def unit_commitment_linearized() -> pypsa.Network:
    load = np.repeat([500.0, 600.0, 800.0, 1000.0, 700.0, 600.0], [5, 5, 5, 5, 10, 10])
    n = _single_bus(40, list(load))
    for i in range(40):
        rng = np.random.default_rng(i + 1)
        unit: dict[str, Any] = {
            "p_min_pu": rng.integers(1, 5) / 10,
            "marginal_cost": rng.integers(1, 11) * 10,
            "min_up_time": rng.integers(0, 6),
            "min_down_time": rng.integers(0, 6),
            "p_nom": rng.integers(1, 10) * 5,
            "start_up_cost": rng.integers(1, 5) * 100,
        }
        shut_down_cost = rng.integers(1, 5) * 100 if i >= 20 else unit["start_up_cost"]
        n.add(
            "Generator",
            f"{i}",
            bus="bus",
            committable=True,
            up_time_before=0,
            shut_down_cost=shut_down_cost,
            **unit,
        )
    n.optimize(linearized_unit_commitment=True)
    return n


def piecewise_costs() -> pypsa.Network:
    n = _single_bus(2, [80, 230])
    n.add("Generator", "flat", bus="bus", p_nom=100, marginal_cost=50)
    n.add(
        "Generator",
        "curve",
        bus="bus",
        p_nom=100,
        marginal_cost={0.0: 0.0, 0.1: 60.0, 0.5: 35.0, 1.0: 100.0},
    )
    n.add(
        "Generator",
        "build",
        bus="bus",
        p_nom_extendable=True,
        p_nom_max=100,
        marginal_cost=10,
        capital_cost=pd.DataFrame(
            {"p_nom": [0.0, 10, 50, 100.0], "capital_cost": [0.0, 1, 1.5, 2.0]}
        ),
    )
    n.optimize(solver_options=EXACT_MILP)
    return n


def multiport_process() -> pypsa.Network:
    n = pypsa.Network(snapshots=range(3))
    n.add("Carrier", ["AC", "H2", "heat", "electrolyser"])
    n.add("Bus", ["elec", "h2", "heat"], carrier=["AC", "H2", "heat"])
    n.add("Generator", "gen", bus="elec", carrier="AC", p_nom=200, marginal_cost=10)
    n.add(
        "Process",
        "electrolyser",
        bus0="elec",
        bus1="h2",
        bus2="heat",
        carrier="electrolyser",
        rate0=-1,
        rate1=0.7,
        rate2=0.2,
        p_nom_extendable=True,
        capital_cost=5,
    )
    n.add("Load", "h2_load", bus="h2", carrier="H2", p_set=[50, 60, 40])
    n.add("Load", "heat_load", bus="heat", carrier="heat", p_set=5)
    n.add("Generator", "heat_sink", bus="heat", carrier="heat", p_nom=100, sign=-1)
    n.optimize()
    return n


CASES: dict[str, Callable[[], pypsa.Network]] = {
    case.__name__: case
    for case in [
        ac_dc_meshed,
        ac_dc_meshed_operational_limit,
        ac_dc_meshed_stochastic,
        storage_hvdc,
        storage_hvdc_operational_limit,
        scigrid_de,
        scigrid_de_security_constrained,
        scigrid_de_transmission_losses,
        model_energy,
        stochastic_network,
        multi_period,
        multi_period_expansion_limits,
        multi_period_growth_limit,
        multi_period_storage_primary_energy,
        multi_period_storage_stochastic,
        unit_commitment,
        unit_commitment_link,
        unit_commitment_extendable,
        unit_commitment_extendable_linearized,
        unit_commitment_linearized,
        piecewise_costs,
        multiport_process,
    ]
}


def total_cost(case: str) -> float:
    """Return the optimal objective of a case, with the constant term."""
    n = CASES[case]()
    assert n.objective is not None
    assert n.objective_constant is not None
    return n.objective + n.objective_constant


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
@pytest.mark.parametrize("case", CASES)
def test_objective_matches_master(case: str, sparse: bool) -> None:
    expected = json.loads(REFERENCE.read_text())["objectives"][case]
    model_kwargs = {"sparse": sparse}
    with pypsa.option_context("params.optimize.model_kwargs", model_kwargs):
        objective = total_cost(case)
    assert objective == pytest.approx(expected, rel=RELATIVE_TOLERANCE)


if __name__ == "__main__":
    pypsa.options.params.optimize.include_objective_constant = True
    objectives = {}
    for case in CASES:
        start = time.perf_counter()
        objectives[case] = total_cost(case)
        print(f"{case}: {time.perf_counter() - start:.1f} s", file=sys.stderr)
    REFERENCE.parent.mkdir(exist_ok=True)
    record = {"commit": sys.argv[1], "objectives": objectives}
    REFERENCE.write_text(json.dumps(record, indent=2) + "\n")
