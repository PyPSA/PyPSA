# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as equal

import pypsa


@pytest.fixture
def target_gen_p():
    target_path = (
        Path(__file__).parent
        / "data"
        / "storage-hvdc"
        / "results-lopf"
        / "generators-p.csv"
    )
    return pd.read_csv(target_path, index_col=0, parse_dates=True)


def test_optimize(storage_hvdc_network, target_gen_p):
    n = storage_hvdc_network
    n.optimize()
    equal(n.c.generators.dynamic.p.reindex_like(target_gen_p), target_gen_p, decimal=2)


def test_storage_energy_marginal_cost():
    n = pypsa.Network()
    n.snapshots = range(3)
    n.add("Bus", "bus")
    n.add(
        "Generator",
        "gen",
        marginal_cost=1,
        bus="bus",
        p_nom=3,
        p_max_pu=[1, 0, 0],
    )
    n.add("Load", "load", bus="bus", p_set=1)
    n.add(
        "Store",
        "store",
        bus="bus",
        marginal_cost_storage=0.2,
        e_initial=1,
        e_nom=10,
    )
    n.optimize()
    assert n.objective == 2.6


def test_spill_cost():
    sets_of_snapshots = 2
    p_set = [100, 100, 100, 100, 100]

    for has_spill_cost in [False, True]:
        n = pypsa.Network(snapshots=range(len(p_set) * sets_of_snapshots))

        n.add("Bus", "bus")

        # Add high capacity generator to help
        n.add(
            "Generator", "help", bus="bus", p_nom=1000, control="PQ", marginal_cost=100
        )

        # Add hydro unit
        if has_spill_cost:
            n.add(
                "StorageUnit",
                "hydro",
                bus="bus",
                p_nom=100,
                max_hours=10,
                inflow=[200, 200, 200, 200, 200, 50, 50, 50, 50, 50],
                spill_cost=1,
            )
        else:
            n.add(
                "StorageUnit",
                "hydro",
                bus="bus",
                p_nom=100,
                max_hours=10,
                inflow=[200, 200, 200, 200, 200, 50, 50, 50, 50, 50],
            )

        # Add Load
        n.add("Load", "load", bus="bus", p_set=p_set * sets_of_snapshots)

        overlap = 2
        for i in range(sets_of_snapshots):
            if i == 1:
                n.c.storage_units.static.state_of_charge_initial = (
                    n.c.storage_units.dynamic.state_of_charge.loc[n.snapshots[4]]
                )
            n.optimize(
                n.snapshots[i * len(p_set) : (i + 1) * len(p_set) + overlap],
            )

        spill = n.c.storage_units.dynamic["spill"].loc[:, "hydro"]
        total_spill = spill.sum()

        if has_spill_cost:
            assert total_spill == 0
        else:
            assert total_spill == 400


def test_storage_unit_p_set():
    """Test that p_set constrains net power (p_dispatch - p_store) for StorageUnit."""
    n = pypsa.Network()
    n.set_snapshots(range(4))

    n.add("Bus", "bus")
    n.add("Generator", "gen", bus="bus", p_nom=100, marginal_cost=10)
    n.add("Load", "load", bus="bus", p_set=[20, 30, 25, 35])

    n.add(
        "StorageUnit",
        "storage",
        bus="bus",
        p_nom=50,
        max_hours=2,
        p_set=[-10, 0, 5, 0],  # negative=charge, positive=discharge
        state_of_charge_initial=10,
    )

    n.optimize()

    # Check that p = p_dispatch - p_store equals p_set
    equal(n.c.storage_units.dynamic.p["storage"].values, [-10, 0, 5, 0], decimal=5)


def test_store_p_set():
    """Test p_set for Store components."""
    n = pypsa.Network()
    n.set_snapshots(range(4))

    n.add("Bus", "bus")
    n.add("Generator", "gen", bus="bus", p_nom=100, marginal_cost=10)
    n.add("Load", "load", bus="bus", p_set=[20, 30, 25, 35])

    n.add(
        "Store",
        "store",
        bus="bus",
        e_nom=100,
        p_set=[-10, 0, 5, 0],  # negative=charge, positive=discharge
        e_initial=10,
    )

    n.optimize()

    equal(n.c.stores.dynamic.p["store"].values, [-10, 0, 5, 0], decimal=5)


COMPONENTS = ["StorageUnit", "Store"]


def _fade_network(
    component="StorageUnit",
    *,
    k=0.0,
    extendable=False,
    throughput_initial=0.0,
    cycles_max=np.inf,
    level_initial=0.0,
):
    """One bus, cheap power for half of each day, a 40 MWh battery that arbitrages it.

    48 hourly snapshots. ``component`` picks the battery's type: a
    ``StorageUnit`` (10 MW, 4 h, efficiencies 0.9) or a ``Store`` (40 MWh, no
    power limit, no efficiencies). It fills during the twelve cheap hours and
    empties during the twelve expensive ones, so its level reaches the ceiling
    on both days; with ``k`` large the second day's ceiling is visibly lower
    than the first.
    """
    n = pypsa.Network()
    n.set_snapshots(pd.RangeIndex(48))
    cheap_hours = pd.Series(
        [1.0 if t % 24 < 12 else 0.0 for t in range(48)], n.snapshots
    )
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=10.0)
    n.add(
        "Generator",
        "cheap",
        bus="bus",
        p_nom=30.0,
        marginal_cost=1.0,
        p_max_pu=cheap_hours,
    )
    n.add("Generator", "expensive", bus="bus", p_nom=30.0, marginal_cost=100.0)
    ageing = {
        "degradation_per_cycle": k,
        "throughput_initial": throughput_initial,
        "cycles_max": cycles_max,
    }
    if component == "StorageUnit":
        n.add(
            "StorageUnit",
            "battery",
            bus="bus",
            p_nom=10.0,
            p_nom_extendable=extendable,
            p_nom_max=10.0,
            capital_cost=1.0,
            max_hours=4.0,
            efficiency_store=0.9,
            efficiency_dispatch=0.9,
            state_of_charge_initial=level_initial,
            **ageing,
        )
    else:
        n.add(
            "Store",
            "battery",
            bus="bus",
            e_nom=40.0,
            e_nom_extendable=extendable,
            e_nom_max=40.0,
            capital_cost=1.0,
            e_initial=level_initial,
            **ageing,
        )
    return n


def _level(n, component="StorageUnit"):
    """Solved storage level of the battery: state of charge or energy level."""
    attr = "state_of_charge" if component == "StorageUnit" else "e"
    return n.components[component].dynamic[attr]["battery"]


def _capacity(n, component="StorageUnit"):
    """Nominal energy capacity of the battery after the solve, in MWh."""
    static = n.components[component].static
    if component == "StorageUnit":
        return static.max_hours["battery"] * static.p_nom_opt["battery"]
    return static.e_nom_opt["battery"]


def _throughput_from_dispatch(n, component="StorageUnit"):
    """Cumulative throughput of the battery's storage level, from its dispatch."""
    c = n.components[component]
    if component == "StorageUnit":
        flow = (
            c.dynamic.p_store["battery"] * c.static.efficiency_store["battery"]
            + c.dynamic.p_dispatch["battery"] / c.static.efficiency_dispatch["battery"]
        )
    else:
        flow = c.dynamic.p["battery"].abs()
    return (flow * n.snapshot_weightings.stores).cumsum()


@pytest.mark.parametrize("component", COMPONENTS)
def test_capacity_fade_attributes_default_off(component):
    """The new attributes exist, default to off, and build nothing."""
    n = _fade_network(component)
    c = n.components[component]
    assert {"degradation_per_cycle", "throughput_initial", "cycles_max"} <= set(
        c.defaults.index
    )
    assert c.static.degradation_per_cycle["battery"] == 0
    assert c.static.throughput_initial["battery"] == 0
    assert c.static.cycles_max["battery"] == np.inf
    assert c.dynamic.throughput.empty

    n.optimize()
    new = ("throughput", "fade", "cycles_max", "p_dispatch_link")
    assert f"{component}-throughput" not in n.model.variables
    assert "Store-p_dispatch" not in n.model.variables
    assert not [name for name in n.model.constraints if any(s in name for s in new)]
    assert c.dynamic.throughput.empty
    assert np.isnan(c.static.mu_cycles_max["battery"])
    if component == "Store":
        assert c.dynamic.p_dispatch.empty


@pytest.mark.parametrize("component", COMPONENTS)
def test_throughput_state_accumulates_dispatch(component):
    """The state is the initial value plus the cumulative flow through the level."""
    n = _fade_network(component, k=0.02, throughput_initial=400.0)
    status, _ = n.optimize()
    assert status == "ok"

    q = n.components[component].dynamic.throughput["battery"]
    expected = 400.0 + _throughput_from_dispatch(n, component)
    assert np.allclose(q, expected, atol=1e-6)
    assert (q.diff().dropna() >= -1e-9).all()


@pytest.mark.parametrize("component", COMPONENTS)
@pytest.mark.parametrize("extendable", [False, True])
@pytest.mark.parametrize("k", [0.02, 3.3e-5])
def test_capacity_fade_derates_level(component, extendable, k):
    """The ceiling follows the throughput accrued up to each snapshot."""
    n = _fade_network(component, k=k, extendable=extendable)
    status, _ = n.optimize()
    assert status == "ok"

    e_nom = _capacity(n, component)
    q = n.components[component].dynamic.throughput["battery"]
    level = _level(n, component)
    ceiling = e_nom - k / 2 * q
    assert (level <= ceiling + 1e-6).all()

    if k == 0.02:
        # the row binds: the peak reaches the faded ceiling, not the nameplate,
        # and the second day peaks lower than the first
        assert level.max() < e_nom - 1e-3
        assert (level - ceiling).max() > -1e-6
        assert level.iloc[24:].max() < level.iloc[:24].max() - 1e-3


@pytest.mark.parametrize("component", COMPONENTS)
def test_capacity_fade_throughput_initial_lowers_ceiling(component):
    """Throughput before the horizon derates the ceiling from the first hour."""
    # 400 MWh is five full cycles on 40 MWh
    n = _fade_network(component, k=0.02, throughput_initial=400.0)
    n.optimize()

    q = n.components[component].dynamic.throughput["battery"]
    level = _level(n, component)
    assert level.max() <= 40.0 * (1 - 0.02 * 5) + 1e-6
    assert (level <= 40.0 - 0.01 * q + 1e-6).all()
    assert (level - (40.0 - 0.01 * q)).max() > -1e-6


@pytest.mark.parametrize("component", COMPONENTS)
def test_capacity_fade_warns_when_initial_exhausts_capacity(component, caplog):
    """A `throughput_initial` worth more than 100 % fade is reported at build time."""
    # 8000 MWh is 100 cycles at 2 % each
    n = _fade_network(component, k=0.02, throughput_initial=8000.0)
    with caplog.at_level(logging.WARNING):
        n.optimize.create_model()
    assert "throughput_initial" in caplog.text


def test_store_p_dispatch_counts_absolute_power():
    """`2 p_dispatch - p` is `|p|`, also when the level does not return to its start."""
    n = _fade_network("Store", k=0.02, level_initial=40.0)  # starts full, ends empty
    n.optimize()

    p = n.stores_t.p["battery"]
    p_dispatch = n.stores_t.p_dispatch["battery"]
    assert np.allclose(p_dispatch, p.clip(lower=0), atol=1e-6)
    assert np.allclose(2 * p_dispatch - p, p.abs(), atol=1e-6)

    q = n.stores_t.throughput["battery"]
    assert q.iloc[-1] == pytest.approx(p.abs().sum(), abs=1e-6)
    # `2 p_dispatch` alone would overcount by the 40 MWh net discharge
    assert 2 * p_dispatch.sum() == pytest.approx(p.abs().sum() + 40.0, abs=1e-6)
    assert _level(n, "Store").iloc[-1] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("component", COMPONENTS)
def test_get_cycles_counts_horizon_throughput(component):
    """Cycles are the horizon's throughput of the level over twice the capacity."""
    n = _fade_network(component, k=0.02, throughput_initial=400.0)
    c = n.components[component]
    assert np.isnan(c.get_cycles()["battery"])

    n.optimize()
    q = c.dynamic.throughput["battery"]
    expected = (q.iloc[-1] - 400.0) / (2 * 40.0)
    assert expected > 1.5
    assert c.get_cycles()["battery"] == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("component", COMPONENTS)
def test_capacity_fade_rolling_horizon_carries_throughput(component):
    """The last throughput of one window is the initial value of the next."""
    n = _fade_network(component, k=0.02, throughput_initial=400.0)
    n.optimize.optimize_with_rolling_horizon(horizon=24)

    c = n.components[component]
    q = c.dynamic.throughput["battery"]
    expected = 400.0 + _throughput_from_dispatch(n, component)
    assert np.allclose(q, expected, atol=1e-6)
    assert c.static.throughput_initial["battery"] == pytest.approx(q.iloc[23])


@pytest.mark.parametrize("component", COMPONENTS)
@pytest.mark.parametrize("extendable", [False, True])
def test_cycle_budget_binds_at_cycles_max(component, extendable):
    """Throughput over the horizon stops at `2 e_nom cycles_max`."""
    free = _fade_network(component, extendable=extendable)
    free.optimize()
    assert free.components[component].get_cycles()["battery"] > 1.5

    n = _fade_network(component, extendable=extendable, cycles_max=1.0)
    status, _ = n.optimize()
    assert status == "ok"
    kind = "ext" if extendable else "fix"
    assert f"{component}-{kind}-cycles_max" in n.model.constraints
    assert f"{component}-throughput" not in n.model.variables  # no state needed
    cycles = n.components[component].get_cycles()["battery"]
    assert cycles == pytest.approx(1.0, abs=1e-6)
    assert n.objective > free.objective


@pytest.mark.parametrize("component", COMPONENTS)
def test_cycle_budget_with_capacity_fade(component):
    """Budget and fade coexist on one asset and count the same throughput."""
    n = _fade_network(component, k=0.02, cycles_max=1.0)
    status, _ = n.optimize()
    assert status == "ok"

    level_attr = "state_of_charge" if component == "StorageUnit" else "e"
    assert f"{component}-fix-cycles_max" in n.model.constraints
    assert f"{component}-fix-{level_attr}-fade" in n.model.constraints
    q = n.components[component].dynamic.throughput["battery"]
    assert q.iloc[-1] == pytest.approx(2 * 40.0 * 1.0, abs=1e-6)  # one cycle
    assert (_level(n, component) <= 40.0 - 0.01 * q + 1e-6).all()


@pytest.mark.parametrize("component", COMPONENTS)
def test_mu_cycles_max_is_assigned(component):
    """The shadow price of a binding cycle budget lands in `mu_cycles_max`."""
    n = _fade_network(component, cycles_max=1.0)
    n.optimize()
    mu = n.components[component].static.mu_cycles_max["battery"]
    assert np.isfinite(mu)
    assert abs(mu) > 1e-6


@pytest.mark.parametrize("component", COMPONENTS)
@pytest.mark.parametrize("scenarios", [False, True])
def test_set_cycle_life_uses_stores_weighting(component, scenarios):
    """The budget scales with the `stores` weighting, not with `nyears`."""
    n = _fade_network(component)
    if scenarios:
        n.set_scenarios({"a": 0.5, "b": 0.5})
    c = n.components[component]
    c.static.lifetime = 20.0
    n.snapshot_weightings.objective = 365.0  # objective clock: 2 years; stores: 48 h
    c.set_cycle_life(6000, soh_end=0.8, names=["battery"])
    assert c.static.degradation_per_cycle.to_numpy() == pytest.approx(0.2 / 6000)
    assert c.static.cycles_max.to_numpy() == pytest.approx(6000 * (48 / 8760) / 20)
    assert n.nyears == pytest.approx(2.0)


@pytest.mark.parametrize("component", COMPONENTS)
def test_set_cycle_life_rejects_unknown_names(component):
    """Names that are not in the component raise instead of being skipped."""
    c = _fade_network(component).components[component]
    with pytest.raises(ValueError, match="unknown"):
        c.set_cycle_life(6000, names=["battery", "missing"])


@pytest.mark.parametrize("component", COMPONENTS)
def test_set_cycle_life_warns_on_infinite_lifetime(component, caplog):
    """Without a finite `lifetime` the budget cannot be derived and stays off."""
    n = _fade_network(component)
    c = n.components[component]
    with caplog.at_level(logging.WARNING):
        c.set_cycle_life(6000)
    assert c.static.degradation_per_cycle["battery"] == pytest.approx(0.2 / 6000)
    assert c.static.cycles_max["battery"] == np.inf
    assert "lifetime" in caplog.text
