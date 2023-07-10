#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import pytest
from conftest import optimize

APIS = ["linopy"]


@pytest.mark.parametrize("api", APIS)
def test_operational_limit_ac_dc_meshed(ac_dc_network, api):
    n = ac_dc_network

    limit = 30_000

    n.global_constraints.drop(n.global_constraints.index, inplace=True)

    n.add(
        "GlobalConstraint",
        "gas_limit",
        type="operational_limit",
        carrier_attribute="gas",
        sense="<=",
        constant=limit,
    )

    optimize(n, api)
    assert n.statistics.dispatch().loc[:, "gas"].sum().round(3) == limit


@pytest.mark.parametrize("api", APIS)
def test_operational_limit_storage_hvdc(storage_hvdc_network, api):
    n = storage_hvdc_network

    limit = 5_000

    n.global_constraints.drop(n.global_constraints.index, inplace=True)

    n.add(
        "GlobalConstraint",
        "battery_limit",
        type="operational_limit",
        carrier_attribute="battery",
        sense="<=",
        constant=limit,
    )

    n.storage_units["state_of_charge_initial"] = 1_000
    n.storage_units.p_nom_extendable = True
    n.storage_units.cyclic_state_of_charge = False

    optimize(n, api)

    soc_diff = (
        n.storage_units.state_of_charge_initial.sum()
        - n.storage_units_t.state_of_charge.sum(1).iloc[-1]
    )
    assert soc_diff.round(3) == limit
