# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import pandas as pd
import pytest

import pypsa


@pytest.fixture(scope="module")
def _model():
    return pypsa.examples.model_energy()


@pytest.fixture(scope="class")
def typical_day_model(_model):
    m = _model.cluster.temporal.typical_periods(
        num_typical_periods=3, num_days_per_period=1
    )
    return m


@pytest.fixture(scope="class")
def typical_week_model(_model):
    m = _model.cluster.temporal.typical_periods(
        num_typical_periods=3, num_days_per_period=7
    )
    return m


class TestBuild:
    @pytest.fixture(scope="class")
    def typical_day_model_built(self, typical_day_model):
        typical_day_model.optimize.create_model()
        return typical_day_model

    @pytest.fixture(scope="class")
    def typical_week_model_built(self, typical_week_model):
        typical_week_model.optimize.create_model()
        return typical_week_model

    @pytest.mark.parametrize(
        "constraint_name",
        [
            "StorageUnit-energy_balance",
            "StorageUnit-state_of_charge-intra-typical-period-lower",
            "StorageUnit-state_of_charge-intra-typical-period-upper",
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
    @pytest.mark.parametrize(
        "network_model", ["typical_day_model_built", "typical_week_model_built"]
    )
    def test_storage_constraints_exist(self, request, constraint_name, network_model):
        n = request.getfixturevalue(network_model)
        assert constraint_name in n.model.constraints


class TestComponentDefs:
    @pytest.mark.parametrize(
        ("component", "attr"), [("StorageUnit", "p_nom"), ("Store", "e")]
    )
    @pytest.mark.parametrize("extendable", [[True], [False], [True, False]])
    @pytest.mark.parametrize(
        "network_model", ["typical_day_model", "typical_week_model"]
    )
    def test_simultaneous_ext_and_non_ext_storage_constraints_exist(
        self,
        request,
        component,
        attr,
        extendable,
        network_model,
    ):
        """Check that extendable / non-extendable `p_nom`/`e`, and a mixture of the two all create expected constraints"""
        n = request.getfixturevalue(network_model)
        template = n.components[component].static.iloc[0]
        n.remove(component, n.components[component].static.index)
        for i, ext in enumerate(extendable):
            n.add(component, f"s{i}", **{f"{attr}_extendable": ext, **template})
        n.optimize.create_model()
        # TODO: check f"{component}-{attr}-inter-typical-period-{bound}"


class TestResults:
    @pytest.fixture(scope="class")
    def typical_day_model_opt(self, typical_day_model):
        typical_day_model.optimize()
        return typical_day_model

    @pytest.fixture(scope="class")
    def typical_week_model_opt(self, typical_week_model):
        typical_week_model.optimize()
        return typical_week_model

    def calc_orig_ts(self, n):
        valid_combinations = (
            pd.merge(
                n.typical_period_map.rename("cluster").reset_index(),
                n.typical_periods.rename("cluster").reset_index(),
            )
            .set_index(["day", "snapshot"])
            .cluster
        )
        return valid_combinations

    @pytest.mark.parametrize(
        "network_model", ["typical_day_model_opt", "typical_week_model_opt"]
    )
    def test_store_remains_within_limits(
        self,
        request,
        network_model,
    ):
        """Check that Store `e` never exceeds defined limits."""
        n = request.getfixturevalue(network_model)
        ts = self.calc_orig_ts(n)
        storage_level = (
            n.components["Store"].dynamic["e_inter_period"].align(ts, axis=0)[0]
            + n.components["Store"].dynamic["e"].align(ts, axis=0)[0]
        )
        max_storage_level = n.components["Store"].static["e_nom_opt"]
        assert (storage_level.max() <= max_storage_level * 1.0001).all()

    @pytest.mark.parametrize(
        "network_model", ["typical_day_model", "typical_week_model"]
    )
    def test_storageunit_remains_within_limits(
        self,
        request,
        network_model,
    ):
        """Check that Storage Unit `state_of_charge` never exceeds defined limits."""
        n = request.getfixturevalue(network_model)
        ts = self.calc_orig_ts(n)
        storage_level = (
            n.components["StorageUnit"]
            .dynamic["state_of_charge_inter_period"]
            .align(ts, axis=0)[0]
            + n.components["StorageUnit"]
            .dynamic["state_of_charge"]
            .align(ts, axis=0)[0]
        )
        max_storage_level = (
            n.components["StorageUnit"].static["p_nom_opt"]
            * n.components["StorageUnit"].static["max_hours"]
        )
        assert (storage_level.max() <= max_storage_level * 1.0001).all()
