# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import pandas as pd
import pytest

import pypsa
from pypsa.common import expand_series
from pypsa.descriptors import nominal_attrs

TOLERANCE = 1e-2


def describe_storage_unit_contraints(n):
    """
    Checks whether all storage units are balanced over time.

    This function requires the network to contain the separate variables
    p_store and p_dispatch, since they cannot be reconstructed from p.
    The latter results from times tau where p_store(tau) > 0 **and**
    p_dispatch(tau) > 0, which is allowed (even though not economic).
    Therefor p_store is necessarily equal to negative entries of p, vice
    versa for p_dispatch.
    """
    sus = n.c.storage_units.static
    sus_i = sus.index
    if sus_i.empty:
        return None
    sns = n.snapshots
    c = "StorageUnit"
    dynamic = n.c[c].dynamic

    eh = expand_series(n.snapshot_weightings.stores[sns], sus_i)
    stand_eff = (1 - n.get_switchable_as_dense(c, "standing_loss", sns)).pow(eh)
    dispatch_eff = n.get_switchable_as_dense(c, "efficiency_dispatch", sns)
    store_eff = n.get_switchable_as_dense(c, "efficiency_store", sns)
    inflow = n.get_switchable_as_dense(c, "inflow") * eh
    spill = eh[dynamic.spill.columns] * dynamic.spill

    description = {
        "Spillage Limit": pd.Series(
            {"min": (inflow[spill.columns] - spill).min().min()}
        )
    }
    if "p_store" in dynamic:
        soc = dynamic.state_of_charge

        store = store_eff * eh * dynamic.p_store  # .clip(upper=0)
        dispatch = 1 / dispatch_eff * eh * dynamic.p_dispatch  # (lower=0)
        start = soc.iloc[-1].where(
            sus.cyclic_state_of_charge, sus.state_of_charge_initial
        )
        previous_soc = stand_eff * soc.shift().fillna(start)

        reconstructed = (
            previous_soc.add(store, fill_value=0)
            .add(inflow, fill_value=0)
            .add(-dispatch, fill_value=0)
            .add(-spill, fill_value=0)
        )
        description["SOC Balance StorageUnit"] = (
            (reconstructed - soc).unstack().describe()
        )
    return pd.concat(description, axis=1, sort=False)


def describe_nodal_balance_constraint(n):
    """
    Helper function to double check whether network flow is balanced.
    """
    network_injection = (
        pd.concat(
            [
                n.c[c].dynamic[f"p{inout}"].rename(columns=n.c[c].static[f"bus{inout}"])
                for inout in (0, 1)
                for c in ("Line", "Transformer")
            ],
            axis=1,
        )
        .T.groupby(level=0)
        .sum()
        .T
    )
    return (
        (n.c.buses.dynamic.p - network_injection)
        .unstack()
        .describe()
        .to_frame("Nodal Balance Constr.")
    )


def describe_upper_dispatch_constraints(n):
    """
    Recalculates the minimum gap between operational status and nominal
    capacity.
    """
    description = {}
    key = " Upper Limit"
    for c, attr in nominal_attrs.items():
        dispatch_attr = "p0" if c in ["Line", "Transformer", "Link"] else attr[0]
        description[c + key] = pd.Series(
            {
                "min": (
                    n.c[c].static[attr + "_opt"]
                    * n.get_switchable_as_dense(c, attr[0] + "_max_pu")
                    - n.c[c].dynamic[dispatch_attr]
                )
                .min()
                .min()
            }
        )
    return pd.concat(description, axis=1)


def describe_lower_dispatch_constraints(n):
    description = {}
    key = " Lower Limit"
    for c, attr in nominal_attrs.items():
        if c in ["Line", "Transformer", "Link"]:
            dispatch_attr = "p0"
            description[c] = pd.Series(
                {
                    "min": (
                        n.c[c].static[attr + "_opt"]
                        * n.get_switchable_as_dense(c, attr[0] + "_max_pu")
                        + n.c[c].dynamic[dispatch_attr]
                    )
                    .min()
                    .min()
                }
            )
        else:
            dispatch_attr = attr[0]
            description[c + key] = pd.Series(
                {
                    "min": (
                        -n.c[c].static[attr + "_opt"]
                        * n.get_switchable_as_dense(c, attr[0] + "_min_pu")
                        + n.c[c].dynamic[dispatch_attr]
                    )
                    .min()
                    .min()
                }
            )
    return pd.concat(description, axis=1)


def describe_store_contraints(n):
    """
    Checks whether all stores are balanced over time.
    """
    stores = n.c.stores.static
    stores_i = stores.index
    if stores_i.empty:
        return None
    sns = n.snapshots
    c = "Store"
    dynamic = n.c[c].dynamic

    eh = expand_series(n.snapshot_weightings.stores[sns], stores_i)
    stand_eff = (1 - n.get_switchable_as_dense(c, "standing_loss", sns)).pow(eh)

    start = dynamic.e.iloc[-1].where(stores.e_cyclic, stores.e_initial)
    previous_e = stand_eff * dynamic.e.shift().fillna(start)

    return (
        (previous_e - dynamic.p - dynamic.e)
        .unstack()
        .describe()
        .to_frame("SOC Balance Store")
    )


def describe_cycle_constraints(n):
    weightings = n.c.lines.static.x_pu_eff.where(
        n.c.lines.static.carrier == "AC", n.c.lines.static.r_pu_eff
    )

    def cycle_flow(sub):
        C = pd.DataFrame(sub.C.todense(), index=sub.components.lines.static.index)
        if C.empty:
            return None
        C_weighted = 1e5 * C.mul(weightings[sub.components.lines.static.index], axis=0)
        return C_weighted.apply(lambda ds: ds @ n.c.lines.dynamic.p0[ds.index].T)

    return (
        pd.concat([cycle_flow(sub) for sub in n.c.sub_networks.static.obj], axis=0)
        .stack()
        .describe()
        .to_frame("Cycle Constr.")
    )


funcs = (
    [
        describe_cycle_constraints,
        # describe_store_contraints,
        # describe_storage_unit_contraints,
        describe_nodal_balance_constraint,
        describe_lower_dispatch_constraints,
        describe_upper_dispatch_constraints,
    ],
)


@pytest.fixture
def solved_n(ac_dc_network):
    n = ac_dc_network
    n.optimize()
    n.c.lines.static["carrier"] = n.c.lines.static.bus0.map(n.c.buses.static.carrier)
    return n


@pytest.mark.parametrize("func", *funcs)
def test_tolerance(solved_n, func):
    n = solved_n
    description = func(n).fillna(0)
    for col in description:
        assert abs(description[col]["min"]) < TOLERANCE
        if "max" in description:
            assert description[col]["max"] < TOLERANCE


def test_optimization_with_strongly_meshed_bus():
    """
    Test that an optimization with a strongly meshed bus works.

    In the linopy framework, the nodal balance constraint is separately
    defined for buses with a large number of components.
    """
    n = pypsa.Network()
    n.set_snapshots(range(2))

    n.add("Bus", "bus")
    n.add("Generator", "gen", bus="bus", p_nom=1, marginal_cost=10)
    n.add("Load", "load", bus="bus", p_set=1)

    n.add("Bus", "bus2")
    n.add("Generator", pd.RangeIndex(50), bus="bus2", p_nom=1, marginal_cost=10)
    n.add("Load", "load2", bus="bus2", p_set=1)

    n.add("Line", "line", bus0="bus", bus1="bus2", s_nom=1)

    n.optimize()

    assert n.c.buses.dynamic.marginal_price.shape == (2, 2)
    assert n.c.buses.dynamic.marginal_price.eq(10).all().all()


def test_define_generator_constraints_static():
    """
    Test define_generator_constraints functionionality without snapshots in the network.
    """
    n = pypsa.Network()
    n.add("Bus", "bus0")
    n.add("Load", "load0", bus="bus0", p_set=10)
    n.add("Generator", "gen0", bus="bus0", p_nom=10, marginal_cost=5)
    n.add("Generator", "gen1", bus="bus0", p_nom=10, marginal_cost=0, e_sum_max=0)
    n.add("Generator", "gen2", bus="bus0", p_nom=10, marginal_cost=10, e_sum_min=10)

    n.optimize()

    assert n.c.generators.dynamic.p["gen0"].eq(0).all()
    assert n.c.generators.dynamic.p["gen1"].eq(0).all()
    assert n.c.generators.dynamic.p["gen2"].eq(10).all()


def test_define_generator_constraints():
    """
    Test define_generator_constraints functionionality with snapshots in the network.
    """
    n = pypsa.Network()

    eh = 10
    snapshots = pd.date_range("2023-01-01", periods=3, freq=f"{eh}h")
    n.set_snapshots(snapshots, eh)

    n.add("Carrier", "carrier")
    n.add("Bus", "bus0")
    n.add("Load", "load0", carrier="carrier", bus="bus0", p_set=10)
    n.add("Generator", "gen0", carrier="carrier", bus="bus0", p_nom=10, marginal_cost=5)
    n.add(
        "Generator",
        "gen1",
        carrier="carrier",
        bus="bus0",
        p_nom=10,
        marginal_cost=0,
        e_sum_max=0,
    )

    e_sum_min = 10 * (len(n.snapshots) - 2) * eh
    n.add(
        "Generator",
        "gen2",
        carrier="carrier",
        bus="bus0",
        p_nom=10,
        marginal_cost=10,
        e_sum_min=e_sum_min,
    )

    e_sum_max = 10 * eh
    n.add(
        "Generator",
        "gen3",
        carrier="carrier",
        bus="bus0",
        p_nom=10,
        marginal_cost=0,
        e_sum_max=e_sum_max,
    )

    n.optimize()

    assert (
        n.snapshot_weightings.generators @ n.c.generators.dynamic.p["gen0"] == 10 * eh
    )
    assert n.c.generators.dynamic.p["gen1"].eq(0).all()
    assert (
        n.snapshot_weightings.generators @ n.c.generators.dynamic.p["gen2"] == e_sum_min
    )
    assert (
        n.snapshot_weightings.generators @ n.c.generators.dynamic.p["gen3"] == e_sum_max
    )


def test_define_fixed_operational_constraints_positive():
    """
    Test fixed operational constraints: fix to a positive value
    """
    n = pypsa.Network()
    n.add("Bus", "bus0")
    n.add("Load", "load0", bus="bus0", p_set=10)
    n.add("Generator", "gen0", bus="bus0", p_nom=4, marginal_cost=0)
    n.add("Generator", "gen1", bus="bus0", p_nom=5, marginal_cost=5)
    n.add("Generator", "gen2", bus="bus0", p_nom=10, marginal_cost=9)

    n.c.generators.dynamic.p_set["gen2"] = 10

    n.optimize()

    assert n.c.generators.dynamic.p["gen2"].eq(10).all()
    assert n.c.generators.dynamic.p["gen0"].eq(0).all()


def test_define_fixed_operational_constraints_zero():
    """
    Test fixed operational constraints: fix to a zero value
    """
    n = pypsa.Network()
    n.add("Bus", "bus0")
    n.add("Load", "load0", bus="bus0", p_set=10)
    n.add("Generator", "gen0", bus="bus0", p_nom=4, marginal_cost=0)
    n.add("Generator", "gen1", bus="bus0", p_nom=5, marginal_cost=5)
    n.add("Generator", "gen2", bus="bus0", p_nom=10, marginal_cost=9)

    n.c.generators.dynamic.p_set["gen0"] = 0

    n.optimize()

    assert n.c.generators.dynamic.p["gen0"].eq(0).all()
    assert n.c.generators.dynamic.p["gen1"].eq(5).all()
    assert n.c.generators.dynamic.p["gen2"].eq(5).all()


def test_define_fixed_operational_constraints_extendable():
    """
    Test fixed operational constraints: extendable component"
    """
    n = pypsa.Network()
    n.add("Bus", "bus0")
    n.add("Load", "load0", bus="bus0", p_set=10)
    n.add(
        "Generator",
        "gen0",
        bus="bus0",
        p_nom_extendable=True,
        capital_cost=10,
        marginal_cost=0,
    )
    n.add(
        "Generator",
        "gen1",
        bus="bus0",
        p_nom_extendable=True,
        capital_cost=10,
        marginal_cost=5,
    )
    n.add(
        "Generator",
        "gen2",
        bus="bus0",
        p_nom_extendable=True,
        capital_cost=10,
        marginal_cost=9,
    )

    n.c.generators.dynamic.p_set["gen1"] = 5

    n.optimize()

    assert n.c.generators.dynamic.p["gen0"].eq(5).all()
    assert n.c.generators.dynamic.p["gen1"].eq(5).all()
    assert n.c.generators.dynamic.p["gen2"].eq(0).all()
