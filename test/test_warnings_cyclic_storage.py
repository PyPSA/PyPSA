# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""
Test warnings for the CP > IP > C precedence of cyclic and initial storage levels.
"""

import logging

import pandas as pd
import pytest

import pypsa

STORAGE_UNIT = {
    "component": "StorageUnit",
    "attrs": {"p_nom": 1, "max_hours": 1, "state_of_charge_initial": 0.5},
    "c": "cyclic_state_of_charge",
    "cp": "cyclic_state_of_charge_per_period",
    "ip": "state_of_charge_initial_per_period",
    "ignored": "Cyclic state of charge constraint overrules initial storage level setting",
    "ip_wins": "Per-period initial state of charge (state_of_charge_initial_per_period=True) "
    "overrides global cyclic",
    "cp_wins": "Per-period cyclic (cyclic_state_of_charge_per_period=True) overrides global cyclic",
}

STORE = {
    "component": "Store",
    "attrs": {"e_nom": 1, "e_initial": 0.5},
    "c": "e_cyclic",
    "cp": "e_cyclic_per_period",
    "ip": "e_initial_per_period",
    "ignored": "Cyclic energy level constraint overrules initial value setting",
    "ip_wins": "Per-period initial energy level (e_initial_per_period=True) "
    "overrides global cyclic",
    "cp_wins": "Per-period cyclic (e_cyclic_per_period=True) overrides global cyclic",
}

SPECS = pytest.mark.parametrize(
    "spec", [STORAGE_UNIT, STORE], ids=["StorageUnit", "Store"]
)


@pytest.fixture
def network():
    def build(multi_invest: bool = True) -> pypsa.Network:
        n = pypsa.Network()
        if multi_invest:
            n.set_snapshots(
                pd.MultiIndex.from_tuples(
                    [(2030, 0), (2030, 1), (2040, 0), (2040, 1)],
                    names=["period", "timestep"],
                )
            )
            n.set_investment_periods([2030, 2040])
        else:
            n.set_snapshots(range(4))
        n.add("Bus", "bus")
        n.add("Carrier", "carrier")
        n.add("Load", "load", bus="bus", p_set=0.1)
        n.add(
            "Generator", "gen", bus="bus", carrier="carrier", p_nom=1, marginal_cost=10
        )
        return n

    return build


def optimize_with_flags(n, spec, caplog, **flags):
    n.add(
        spec["component"],
        "storage",
        bus="bus",
        carrier="carrier",
        marginal_cost=1,
        **spec["attrs"],
        **{spec[flag]: value for flag, value in flags.items()},
    )
    with caplog.at_level(logging.WARNING):
        n.optimize(multi_investment_periods=not n.investment_periods.empty)
    return [record.message for record in caplog.records]


@SPECS
@pytest.mark.parametrize("multi_invest", [False, True])
def test_warning_cyclic_overrules_initial(network, spec, caplog, multi_invest):
    """C alone discards the initial level, in single- and multi-period networks."""
    messages = optimize_with_flags(network(multi_invest), spec, caplog, c=True)

    assert any(spec["ignored"] in message for message in messages)


@SPECS
def test_warning_per_period_cyclic_overrules_initial(network, spec, caplog):
    """CP wins over IP, so the initial level is still discarded."""
    messages = optimize_with_flags(network(), spec, caplog, cp=True, ip=True)

    assert any(spec["ignored"] in message for message in messages)


@SPECS
def test_warning_per_period_cyclic_overrides_global(network, spec, caplog):
    """CP wins over C, which only warns about the cycling regime."""
    messages = optimize_with_flags(network(), spec, caplog, c=True, cp=True)

    assert any(spec["cp_wins"] in message for message in messages)


@SPECS
def test_warning_initial_per_period_overrides_global(network, spec, caplog):
    """IP wins over C, so the initial level is used rather than ignored."""
    messages = optimize_with_flags(network(), spec, caplog, c=True, ip=True)

    assert any(spec["ip_wins"] in message for message in messages)
    assert not any(spec["ignored"] in message for message in messages)
