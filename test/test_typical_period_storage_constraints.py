# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import pytest


@pytest.fixture(scope="module")
def typical_day_model(model_energy_network):
    m = model_energy_network.cluster.temporal.typical_periods(
        num_typical_periods=3, num_days_per_period=1
    )
    m.optimize()
    return m


@pytest.fixture(scope="module")
def typical_week_model(model_energy_network):
    m = model_energy_network.cluster.temporal.typical_periods(
        num_typical_periods=3, num_days_per_period=7
    )
    m.optimize()
    return m


@pytest.mark.parametrize(
    "constraint_name",
    [
        "StorageUnit-energy_balance",
        "StorageUnit-e-intra-typical-period-lower",
        "StorageUnit-e-intra-typical-period-upper",
        "StorageUnit-state_of_charge-inter-typical-period-lower",
        "StorageUnit-state_of_charge-inter-typical-period-upper",
        "StorageUnit-energy_balance_typical_period_inter",
        "Store-energy_balance",
        "Store-e-intra-typical-period-lower",
        "Store-e-intra-typical-period-upper",
        "Store-e-inter-typical-period-lower",
        "Store-e-inter-typical-period-upper",
        "Store-energy_balance_typical_period_inter",
    ],
)
@pytest.mark.parametrize("network_model", ["typical_day_model", "typical_week_model"])
def test_storage_constraints_exist(request, constraint_name, network_model):
    n = request.getfixturevalue(network_model)
    assert constraint_name in n.model.constraints
