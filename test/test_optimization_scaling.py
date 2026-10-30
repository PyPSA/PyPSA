# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

from functools import cache
from typing import Any

import linopy
import numpy as np
import pandas as pd
import pytest
from linopy.constraints import CSRConstraint

import pypsa
from pypsa.optimization.scaling import KVL, KVL_SCALING, STANDARD, ScalingFactors

RTOL = 1e-5


def sector_network() -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots(range(4))
    n.add("Carrier", "gas", co2_emissions=0.2)
    n.add("Carrier", "wind")
    n.add("Bus", ["elec", "gas"])
    n.add("Bus", "co2", unit="t")
    n.add("Generator", "gas", bus="gas", carrier="gas", p_nom=1e4, marginal_cost=20)
    n.add(
        "Generator",
        "wind",
        bus="elec",
        carrier="wind",
        p_nom_extendable=True,
        p_max_pu=[0.2, 0.9, 0.5, 0.1],
        capital_cost=500,
    )
    n.add(
        "Link",
        "turbine",
        bus0="gas",
        bus1="elec",
        bus2="co2",
        efficiency=0.5,
        efficiency2=0.2,
        p_nom_extendable=True,
        capital_cost=100,
    )
    n.add("Store", "atmosphere", bus="co2", e_nom_extendable=True, capital_cost=0.1)
    n.add("Load", "demand", bus="elec", p_set=[300, 400, 500, 300])
    n.add("GlobalConstraint", "co2_limit", sense="<=", constant=150)
    return n


def scale(n: pypsa.Network, factors: dict[str, float]) -> pypsa.Network:
    n.c.buses.static.scaling = n.c.buses.static.carrier.map(factors).fillna(1.0)
    n.c.buses.static.loc[n.c.buses.static.unit == "t", "scaling"] = factors.get(
        "t", 1.0
    )
    return n


NETWORKS = {
    "ac_dc_meshed": (pypsa.examples.ac_dc_meshed, {"AC": 1e-3, "DC": 1e-2}),
    "storage_hvdc": (pypsa.examples.storage_hvdc, {"AC": 1e-3, "DC": 1e-2}),
    "sector": (sector_network, {"AC": 1e-3, "t": 1e-6}),
}


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("network", NETWORKS)
def test_scaled_matches_unscaled(network, sparse):
    build, factors = NETWORKS[network]
    ref = build()
    ref.optimize()
    n = scale(build(), factors)
    n.c.global_constraints.static.scaling = 1e-3
    with linopy.options:
        if sparse:
            linopy.options.set_value(semantics="v1")
        n.optimize(model_kwargs={"sparse": sparse}, objective_scaling=1e-6)

    cons = n.model.constraints
    assert all(isinstance(cons[name], CSRConstraint) == sparse for name in cons)
    for name in cons:
        if name.startswith("GlobalConstraint-"):
            assert (cons[name].scaling == 1e-3).all()
    if KVL in cons:
        bus_factors = KVL_SCALING * n.c.buses.static.scaling.unique()
        assert np.isin(cons[KVL].scaling, bus_factors).all()
    assert n.objective == pytest.approx(ref.objective, rel=RTOL)
    for c, attr in [("generators", "p_nom_opt"), ("global_constraints", "mu")]:
        np.testing.assert_allclose(
            n.c[c].static[attr], ref.c[c].static[attr], rtol=RTOL, atol=1e-6
        )
    for c, attr in [("generators", "p"), ("buses", "marginal_price")]:
        np.testing.assert_allclose(
            n.c[c].dynamic[attr], ref.c[c].dynamic[attr], rtol=RTOL, atol=1e-4
        )


def rows_of(n: pypsa.Network, name: str) -> np.ndarray:
    labels = n.model.constraints[name].labels.values.ravel()
    return np.isin(n.model.matrices.clabels, labels)


def test_uniform_bus_scaling_keeps_nodal_balance_coefficients():
    ref = pypsa.examples.ac_dc_meshed()
    ref.optimize.create_model()
    n = pypsa.examples.ac_dc_meshed()
    n.c.buses.static.scaling = 1e-3
    n.optimize.create_model()

    A, A_ref = n.model.matrices.A, ref.model.matrices.A
    rows = rows_of(n, "Bus-nodal_balance")
    assert (rows == rows_of(ref, "Bus-nodal_balance")).all()
    np.testing.assert_allclose(A[rows].toarray(), A_ref[rows].toarray())
    assert np.isin(np.abs(A[rows].data), [1.0, *ref.c.links.static.efficiency]).all()


def test_t_bus_scaling_sets_link_coefficient():
    n = scale(sector_network(), {"t": 1e-6})
    n.optimize.create_model()
    M = n.model.matrices
    var = n.model.variables["Link-p"].labels.sel(snapshot=0, name="turbine").item()
    con = n.model.constraints["Bus-nodal_balance"].labels.sel(snapshot=0, name="co2")
    col = np.flatnonzero(M.vlabels == var).item()
    row = np.flatnonzero(M.clabels == con.item()).item()
    assert M.A[row, col] == pytest.approx(0.2 * 1e-6 / 1.0)


@pytest.mark.parametrize("bus_factors", [{}, {"AC": 1e-3, "t": 1e-6}])
@pytest.mark.parametrize("objective_scaling", [1.0, 1e-6])
def test_objective_constant_column_takes_objective_scaling(
    objective_scaling, bus_factors
):
    ref = sector_network()
    ref.c.generators.static.loc["wind", "p_nom"] = 100
    n = scale(ref.copy(), bus_factors)
    ref.optimize(include_objective_constant=True)
    n.optimize(include_objective_constant=True, objective_scaling=objective_scaling)

    M = n.model.matrices
    label = n.model.variables["objective_constant"].labels.item()
    col = np.flatnonzero(M.vlabels == label).item()
    assert M.ub[col] == pytest.approx(5e4 * objective_scaling)
    assert M.c[col] == pytest.approx(-1.0)
    assert n.objective_constant == pytest.approx(ref.objective_constant)
    assert n.objective == pytest.approx(ref.objective, rel=RTOL)


def test_kvl_independent_of_bus_scaling():
    ref = pypsa.examples.ac_dc_meshed()
    ref.optimize()
    n = pypsa.examples.ac_dc_meshed()
    n.c.buses.static.scaling = 1e-3
    n.optimize(objective_scaling=1e-3)

    A, A_ref = n.model.matrices.A, ref.model.matrices.A
    np.testing.assert_allclose(
        A[rows_of(n, KVL)].toarray(), A_ref[rows_of(ref, KVL)].toarray(), rtol=1e-12
    )
    np.testing.assert_allclose(
        n.model.constraints[KVL].dual,
        ref.model.constraints[KVL].dual,
        rtol=RTOL,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        n.c.lines.dynamic.p0, ref.c.lines.dynamic.p0, rtol=RTOL, atol=1e-4
    )


@pytest.fixture
def ac_dc_multi_invest(ac_dc_network):
    n = ac_dc_network
    n.snapshots = pd.MultiIndex.from_product([[2020, 2030], n.snapshots[:4]])
    n.investment_periods = [2020, 2030]
    n.add(
        "Line",
        "extra",
        bus0="London",
        bus1="Manchester",
        x=0.1,
        s_nom=500,
        build_year=2030,
        lifetime=50,
    )
    return n


@pytest.mark.parametrize(
    ("network", "kwargs"),
    [
        ("ac_dc_stochastic", {}),
        ("ac_dc_multi_invest", {"multi_investment_periods": True}),
    ],
)
def test_scaled_objective_matches(request, network, kwargs):
    n = request.getfixturevalue(network)
    ref = n.copy()
    ref.optimize(**kwargs)
    scale(n, {"AC": 1e-3, "DC": 1e-2})
    n.optimize(objective_scaling=1e-6, **kwargs)
    assert n.objective == pytest.approx(ref.objective, rel=RTOL)


@pytest.mark.parametrize(
    ("component", "value", "match"),
    [
        ("buses", 0.0, "Bus.scaling"),
        ("buses", np.inf, "Bus.scaling"),
        ("buses", -1.0, "Bus.scaling"),
        ("global_constraints", 0.0, "GlobalConstraint.scaling"),
        ("global_constraints", np.nan, "GlobalConstraint.scaling"),
    ],
)
def test_invalid_factor_raises(component, value, match):
    n = pypsa.examples.ac_dc_meshed()
    n.c[component].static.scaling = value
    with pytest.raises(ValueError, match=match):
        n.optimize.create_model()


def test_mixed_factors_in_sub_network_raise():
    n = pypsa.examples.ac_dc_meshed()
    n.c.buses.static.loc["Manchester", "scaling"] = 1e-3
    with pytest.raises(ValueError, match="sub-network"):
        n.optimize.create_model()


def test_mixed_factors_in_radial_sub_network_pass():
    n = pypsa.examples.ac_dc_meshed()
    n.remove("Line", ["0", "2"])
    n.c.buses.static.loc["Manchester", "scaling"] = 1e-3
    n.optimize.create_model()
    assert KVL not in n.model.constraints


@pytest.mark.parametrize("value", [0.0, -1.0, np.inf])
def test_invalid_objective_scaling_raises(value):
    n = pypsa.examples.ac_dc_meshed()
    with pytest.raises(ValueError, match="finite and positive"):
        n.optimize.create_model(objective_scaling=value)


def test_mga_keeps_objective_scaling():
    weights = {"Generator": {"p_nom": {"Manchester Wind": 1}}}
    results = []
    for factor in (1.0, 1e-3):
        n = pypsa.examples.ac_dc_meshed()
        n.c.buses.static.scaling = factor
        n.optimize(objective_scaling=factor)
        n.optimize.optimize_mga(weights=weights, objective_scaling=factor)
        assert n.model.objective.scaling == factor
        assert (n.model.constraints["budget"].scaling == factor).all()
        results.append(n.c.generators.static.p_nom_opt)
    pd.testing.assert_series_equal(*results, rtol=RTOL)


def test_security_constrained_objective_scaling():
    ref = pypsa.examples.ac_dc_meshed()
    ref.optimize.optimize_security_constrained()
    n = pypsa.examples.ac_dc_meshed()
    n.c.buses.static.scaling = 1e-3
    n.optimize.optimize_security_constrained(objective_scaling=1e-3)
    assert n.model.objective.scaling == 1e-3
    assert n.objective == pytest.approx(ref.objective, rel=RTOL)


def feature_network() -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots(range(6))
    n.add("Carrier", ["gas", "AC"], co2_emissions=[0.2, 0], max_growth=[np.inf] * 2)
    n.add("Bus", ["a", "b", "c"], carrier="AC")
    n.add("Bus", "h2", carrier="H2")
    common: dict[str, Any] = {"bus": "a", "carrier": "gas", "marginal_cost": 10}
    n.add(
        "Generator",
        "uc",
        **common,
        p_nom=100,
        committable=True,
        min_up_time=2,
        min_down_time=2,
        ramp_limit_up=0.5,
        ramp_limit_start_up=0.6,
        start_up_cost=5,
    )
    n.add(
        "Generator",
        "uc_ext",
        **common,
        p_nom_extendable=True,
        committable=True,
        p_nom_max=200,
        capital_cost=10,
    )
    n.add(
        "Generator",
        "modular",
        **common,
        p_nom_extendable=True,
        p_nom_mod=10,
        p_nom_max=200,
        committable=True,
        capital_cost=10,
    )
    n.add(
        "Generator",
        "maint",
        **common,
        p_nom=50,
        committable=True,
        maintainable=True,
        maintenance_duration=2,
        ramp_limit_up=0.5,
        e_sum_max=1e3,
    )
    n.add(
        "Generator",
        "maint_ext",
        **common,
        p_nom_extendable=True,
        p_nom_max=100,
        maintainable=True,
        maintenance_duration=2,
        capital_cost=1,
    )
    n.add(
        "Generator",
        "fixed_ext",
        bus="b",
        p_nom_extendable=True,
        p_nom_set=10,
        marginal_cost=15,
    )
    n.add("Generator", "pw", bus="b", p_nom=50, marginal_cost={0: 0, 0.5: 5, 1: 20})
    n.add(
        "Generator",
        "pw_capex",
        bus="b",
        p_nom_extendable=True,
        p_nom_max=100,
        marginal_cost=12,
        capital_cost={0: 0, 50: 40, 100: 100},
    )
    n.add(
        "Generator",
        "pw_eff",
        **common | {"marginal_cost": 8},
        p_nom=50,
        efficiency={0: 0, 0.5: 0.4, 1: 0.5},
    )
    n.add(
        "Line",
        ["ab", "bc", "ca"],
        bus0=["a", "b", "c"],
        bus1=["b", "c", "a"],
        x=0.1,
        r=0.01,
        s_nom_extendable=True,
        s_nom_max=1e3,
        v_ang_max=30,
    )
    n.add(
        "Transformer",
        "t",
        bus0="a",
        bus1="c",
        x=0.1,
        s_nom=100,
        phase_shift_min=-10,
        phase_shift_max=10,
        v_ang_max=30,
    )
    n.add(
        "Link",
        "electrolysis",
        bus0="b",
        bus1="h2",
        p_nom=50,
        efficiency={0: 0, 0.5: 0.4, 1: 0.7},
        ramp_limit_up=0.5,
    )
    n.add(
        "StorageUnit",
        "hydro",
        bus="c",
        p_nom=20,
        inflow=5,
        max_hours=4,
        p_set=[np.nan, 0, np.nan, np.nan, np.nan, np.nan],
    )
    n.add("Store", "h2_store", bus="h2", e_nom_extendable=True, e_cyclic=True)
    n.add("Load", "la", bus="c", p_set=80)
    n.add("Load", "lh2", bus="h2", p_set=5)
    limits: list[tuple[str, dict[str, Any]]] = [
        ("primary_energy", {"carrier_attribute": "co2_emissions", "constant": 1e4}),
        ("operational_limit", {"carrier_attribute": "gas", "constant": 1e4}),
        (
            "transmission_volume_expansion_limit",
            {"carrier_attribute": "AC", "constant": 1e6},
        ),
        (
            "transmission_expansion_cost_limit",
            {"carrier_attribute": "AC", "constant": 1e9},
        ),
        (
            "tech_capacity_expansion_limit",
            {"carrier_attribute": "gas", "constant": 1e4},
        ),
    ]
    for glc_type, attrs in limits:
        n.add("GlobalConstraint", glc_type, type=glc_type, sense="<=", **attrs)
    return n


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"transmission_losses": True},
        {"transmission_losses": {"mode": "tangents", "segments": 2}},
        {"linearized_unit_commitment": True},
    ],
)
def test_feature_network_scaled_matches_unscaled(kwargs):
    n = feature_network()
    if kwargs.get("linearized_unit_commitment"):
        n.remove("Generator", "modular")
    n.c.buses.static.scaling = [1e-3, 1e-3, 1e-3, 1e-2]
    n.c.global_constraints.static.scaling = 1e-3
    ref = n.copy()
    ref.c.buses.static.scaling = 1.0
    ref.c.global_constraints.static.scaling = 1.0
    ref.optimize(**kwargs)
    m = n.optimize.create_model(objective_scaling=1e-2, **kwargs)
    assert (m.variables["Generator-p"].scaling == 1e-3).all()
    assert (m.variables["Store-e"].scaling == 1e-2).all()
    assert (m.variables["Generator-marginal_cost_piecewise"].scaling == 1e-2).all()
    assert (
        m.constraints["Generator-marginal_cost_piecewise_chord"].scaling == 1e-2
    ).all()
    assert (m.variables["Link-p1_piecewise"].scaling == 1e-3).all()
    assert (m.variables["Generator-status"].scaling == 1).all()
    assert (m.variables["Link-p1_piecewise_delta"].scaling == 1).all()
    status, _ = n.optimize.solve_model()
    assert status == "ok"
    assert n.objective == pytest.approx(ref.objective, rel=RTOL)


def test_cvar_takes_objective_scaling(stochastic_benchmark_network):
    ref = stochastic_benchmark_network
    ref.set_risk_preference(alpha=0.9, omega=0.5)
    n = ref.copy()
    ref.optimize()
    n.c.buses.static.scaling = 1e-3
    n.optimize(objective_scaling=1e-3)

    m = n.model
    cvar = [
        elements[name]
        for elements in (m.variables, m.constraints)
        for name in elements
        if name.startswith("CVaR")
    ]
    assert len(cvar) == 3 + 1 + len(n.scenarios)
    assert all((element.scaling == 1e-3).all() for element in cvar)
    M = m.matrices
    col = np.flatnonzero(M.vlabels == m.variables["CVaR"].labels.item()).item()
    assert M.c[col] == pytest.approx(0.5)
    assert n.objective == pytest.approx(ref.objective, rel=RTOL)
    assert m.variables["CVaR"].solution.item() == pytest.approx(
        ref.model.variables["CVaR"].solution.item(), rel=RTOL
    )


def test_bus_nom_rows_take_bus_scaling():
    ref = sector_network()
    ref.c.buses.static["nom_min_wind"] = [1000.0, np.nan, np.nan]
    n = ref.copy()
    n.c.buses.static.scaling = [1e-3, 1e-2, 1e-6]
    with pytest.warns(DeprecationWarning, match="Nominal constraints"):
        ref.optimize()
    with pytest.warns(DeprecationWarning, match="Nominal constraints"):
        n.optimize()
    assert (n.model.constraints["Bus-nom_min_wind"].scaling == 1e-3).all()
    assert n.c.generators.static.p_nom_opt["wind"] == pytest.approx(1000)
    assert n.objective == pytest.approx(ref.objective, rel=RTOL)


def test_components_scaling():
    n = pypsa.examples.ac_dc_meshed()
    n.c.buses.static.loc["Manchester", "scaling"] = 1e-3
    generators = n.c.generators.scaling.to_series()
    assert generators["Manchester Wind"] == 1e-3
    assert generators["Norway Wind"] == 1
    lines = n.c.lines.scaling.to_series()
    bus0 = n.c.lines.static.bus0
    assert (lines[bus0 == "Manchester"] == 1e-3).all()
    assert (lines[bus0 != "Manchester"] == 1).all()
    assert n.c.buses.scaling.sel(name="Manchester") == 1e-3


@pytest.mark.parametrize("component", ["carriers", "global_constraints"])
def test_components_scaling_requires_bus(component):
    n = pypsa.examples.ac_dc_meshed()
    with pytest.raises(ValueError, match="not attached to a bus"):
        n.c[component].scaling  # noqa: B018


def test_sub_network_components_scaling():
    n = pypsa.examples.ac_dc_meshed()
    factors = {"AC": 1e-3, "DC": 1e-2}
    n.c.buses.static.scaling = n.c.buses.static.carrier.map(factors)
    n.determine_network_topology()
    sub_networks = n.c.sub_networks.static
    for sub_network, carrier in zip(sub_networks.obj, sub_networks.carrier):
        lines = sub_network.c.lines
        assert lines.scaling.indexes["name"].equals(lines.static.index)
        assert (lines.scaling == factors[carrier]).all()


S, OBJ, GLC = 1e-3, 1e-2, 1e-4
COST_COLUMNS = (
    "objective_constant",
    "CVaR",
    "CVaR-a",
    "CVaR-theta",
    "Generator-marginal_cost_piecewise",
    "Generator-capital_cost_piecewise",
)
ROW_OVERRIDES = {KVL: S * KVL_SCALING, "budget": OBJ, "Carrier-growth_limit": 1.0}


def element_factors(elements: linopy.Variables | linopy.Constraints) -> pd.DataFrame:
    frames = []
    for name in elements:
        element = elements[name]
        labels = element.labels.values.ravel()
        scaling = element.scaling.broadcast_like(element.labels).values.ravel()
        frame = pd.DataFrame({"name": name, "scaling": scaling}, index=labels)
        frames.append(frame[labels != -1])
    return pd.concat(frames)


def stochastic_cvar() -> pypsa.Network:
    n = pypsa.examples.ac_dc_meshed()
    n.set_scenarios({"low": 0.3, "high": 0.7})
    n.set_risk_preference(alpha=0.9, omega=0.5)
    return n


def multi_period() -> pypsa.Network:
    n = pypsa.examples.ac_dc_meshed()
    n.snapshots = pd.MultiIndex.from_product([[2020, 2030], n.snapshots[:4]])
    n.investment_periods = [2020, 2030]
    return n


def luc_network() -> pypsa.Network:
    n = feature_network()
    n.remove("Generator", "modular")
    return n


def build_model(n: pypsa.Network, mode: str, kwargs: dict) -> linopy.Model:
    if mode == "rolling":
        n.optimize.optimize_with_rolling_horizon(horizon=3, objective_scaling=OBJ)
    elif mode == "sclopf":
        with linopy.options:
            linopy.options.set_value(semantics="v1")
            n.optimize.optimize_security_constrained(
                objective_scaling=OBJ, model_kwargs={"sparse": True}
            )
    elif mode == "mga":
        n.optimize(objective_scaling=OBJ)
        weights = {"Generator": {"p_nom": {"Manchester Wind": 1}}}
        n.optimize.optimize_mga(weights=weights, objective_scaling=OBJ)
    else:
        n.optimize.create_model(objective_scaling=OBJ, **kwargs)
    return n.model


@pytest.mark.parametrize(
    ("build", "mode", "kwargs"),
    [
        (feature_network, "create", {}),
        (feature_network, "create", {"transmission_losses": True}),
        (
            feature_network,
            "create",
            {"transmission_losses": {"mode": "tangents", "segments": 2}},
        ),
        (luc_network, "create", {"linearized_unit_commitment": True}),
        (sector_network, "create", {"include_objective_constant": True}),
        (stochastic_cvar, "create", {}),
        (multi_period, "create", {"multi_investment_periods": True}),
        (feature_network, "rolling", {}),
        (pypsa.examples.ac_dc_meshed, "sclopf", {}),
        (pypsa.examples.ac_dc_meshed, "mga", {}),
    ],
)
def test_scaling_policy(build, mode, kwargs):
    n = build()
    n.c.buses.static.scaling = S
    n.c.global_constraints.static.scaling = GLC
    m = build_model(n, mode, kwargs)
    M = m.matrices

    cols = element_factors(m.variables).loc[M.vlabels]
    np.testing.assert_allclose(M.var_scaling, cols.scaling)
    assert cols.scaling.isin([1.0, S, OBJ]).all()
    is_cost = cols.name.isin(COST_COLUMNS)
    assert (cols.scaling[is_cost] == OBJ).all()
    assert (cols.scaling[M.vtypes != "C"] == 1).all()

    rows = element_factors(m.constraints).loc[M.clabels]
    A = abs(M.A).tocsr()
    has_cost = A @ (cols.scaling.values == OBJ) > 0
    has_mw = A @ (cols.scaling.values == S) > 0
    expected = pd.Series(np.select([has_cost, has_mw], [OBJ, S], 1.0), rows.index)
    expected[rows.name.str.startswith("GlobalConstraint-")] = GLC
    overrides = rows.name.map(ROW_OVERRIDES)
    expected = overrides.fillna(expected)
    mismatch = rows[~np.isclose(rows.scaling, expected)]
    assert mismatch.empty, mismatch.groupby("name").scaling.first()


USER_FACTORS = ScalingFactors(bus=1e-3, objective=1e-6)


def unit_attributes(n: pypsa.Network) -> bool:
    return all((n.c[c].static.scaling == 1).all() for c in ("Bus", "GlobalConstraint"))


def user_strategy(n: pypsa.Network) -> ScalingFactors:
    return USER_FACTORS


@cache
def unscaled_objective(network: str) -> float:
    n = NETWORKS[network][0]()
    n.optimize()
    assert n.objective is not None
    return n.objective


@pytest.mark.parametrize(
    ("scaling", "factors"),
    [
        ("standard", STANDARD),
        (user_strategy, USER_FACTORS),
        (USER_FACTORS, USER_FACTORS),
    ],
    ids=["standard", "callable", "instance"],
)
@pytest.mark.parametrize("network", ["sector", "ac_dc_meshed"])
def test_strategy_matches_unscaled(network, scaling, factors):
    n = NETWORKS[network][0]()
    n.optimize(scaling=scaling)
    assert n.objective == pytest.approx(unscaled_objective(network), rel=RTOL)
    assert n.model.objective.scaling == factors.objective
    assert unit_attributes(n)


def test_factors_reusable():
    a = sector_network()
    a.optimize.create_model(scaling=STANDARD)
    b = sector_network()
    STANDARD.apply(b)
    b.optimize.create_model(objective_scaling=STANDARD.objective)
    Ma, Mb = a.model.matrices, b.model.matrices
    np.testing.assert_array_equal(Ma.var_scaling, Mb.var_scaling)
    np.testing.assert_array_equal(Ma.b, Mb.b)
    assert a.model.objective.scaling == b.model.objective.scaling


@pytest.mark.parametrize("scaling", ["standard", user_strategy])
@pytest.mark.parametrize(
    ("component", "objective_scaling"),
    [("Bus", 1.0), ("GlobalConstraint", 1.0), (None, 1e-3)],
)
def test_strategy_with_user_factors_raises(scaling, component, objective_scaling):
    n = pypsa.examples.ac_dc_meshed()
    if component:
        n.c[component].static.scaling = 0.5
    with pytest.raises(ValueError, match="scaling=False"):
        n.optimize.create_model(scaling=scaling, objective_scaling=objective_scaling)


@pytest.mark.parametrize(
    ("scaling", "error"),
    [
        (True, TypeError),
        (1e-3, TypeError),
        ("tune", ValueError),
        (lambda n: {}, TypeError),
        (ScalingFactors(bus=pd.Series({"London": 0.5})), ValueError),
    ],
)
def test_invalid_scaling_raises(scaling, error):
    n = pypsa.examples.ac_dc_meshed()
    with pytest.raises(error):
        n.optimize.create_model(scaling=scaling)
    assert unit_attributes(n)


def test_invalid_scaling_type_precedes_user_factors():
    n = pypsa.examples.ac_dc_meshed()
    n.c.buses.static.scaling = 0.5
    with pytest.raises(TypeError):
        n.optimize.create_model(scaling=True)


@pytest.mark.parametrize(
    "passed",
    [{"scaling": "standard"}, {"model_kwargs": {"scaling": "standard"}}],
    ids=["kwargs", "model_kwargs"],
)
@pytest.mark.parametrize(
    ("method", "args"),
    [
        (pypsa.optimization.OptimizationAccessor.optimize_security_constrained, ()),
        (pypsa.optimization.OptimizationAccessor.optimize_mga, ()),
        (pypsa.optimization.OptimizationAccessor.optimize_mga_in_direction, ({}, {})),
        (
            pypsa.optimization.OptimizationAccessor.optimize_mga_in_multiple_directions,
            ([], {}),
        ),
    ],
    ids=["security_constrained", "mga", "mga_in_direction", "mga_multiple"],
)
def test_rejects_scaling(method, args, passed):
    n = pypsa.examples.ac_dc_meshed()
    with pytest.raises(TypeError, match="does not accept"):
        method(n.optimize, *args, **passed)


def test_rolling_horizon_standard():
    ref = feature_network()
    ref.optimize.optimize_with_rolling_horizon(horizon=3)
    n = feature_network()
    n.optimize.optimize_with_rolling_horizon(horizon=3, scaling="standard")
    assert n.objective == pytest.approx(ref.objective, rel=RTOL)
    assert unit_attributes(n)
