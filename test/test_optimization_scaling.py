# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import linopy
import numpy as np
import pandas as pd
import pytest
from linopy.constraints import CSRConstraint

import pypsa
from pypsa.optimization.scaling import KVL, KVL_SCALING

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


def nodal_rows(n: pypsa.Network) -> np.ndarray:
    labels = n.model.constraints["Bus-nodal_balance"].labels.values.ravel()
    return np.isin(n.model.matrices.clabels, labels)


def test_uniform_bus_scaling_keeps_nodal_balance_coefficients():
    ref = pypsa.examples.ac_dc_meshed()
    ref.optimize.create_model()
    n = pypsa.examples.ac_dc_meshed()
    n.c.buses.static.scaling = 1e-3
    n.optimize.create_model()

    A, A_ref = n.model.matrices.A, ref.model.matrices.A
    rows = nodal_rows(n)
    assert (rows == nodal_rows(ref)).all()
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


def kvl_rows(n: pypsa.Network) -> np.ndarray:
    labels = n.model.constraints[KVL].labels.values.ravel()
    return np.isin(n.model.matrices.clabels, labels)


def test_kvl_independent_of_bus_scaling():
    ref = pypsa.examples.ac_dc_meshed()
    ref.optimize()
    n = pypsa.examples.ac_dc_meshed()
    n.c.buses.static.scaling = 1e-3
    n.optimize(objective_scaling=1e-3)

    A, A_ref = n.model.matrices.A, ref.model.matrices.A
    np.testing.assert_allclose(
        A[kvl_rows(n)].toarray(), A_ref[kvl_rows(ref)].toarray(), rtol=1e-12
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
    common = {"bus": "a", "carrier": "gas", "marginal_cost": 10}
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
    for glc_type, attrs in [
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
    ]:
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
def test_registry_covers_all_model_elements(kwargs):
    n = feature_network()
    if kwargs.get("linearized_unit_commitment"):
        n.remove("Generator", "modular")
    n.c.buses.static.scaling = [1e-3, 1e-3, 1e-3, 1e-2]
    n.c.global_constraints.static.scaling = 1e-3
    ref = n.copy()
    ref.c.buses.static.scaling = 1.0
    ref.c.global_constraints.static.scaling = 1.0
    ref.optimize(**kwargs)
    m = n.optimize.create_model(**kwargs)
    assert (m.variables["Generator-p"].scaling == 1e-3).all()
    assert (m.variables["Store-e"].scaling == 1e-2).all()
    assert (m.variables["Generator-marginal_cost_piecewise"].scaling == 1e-3).all()
    assert (
        m.constraints["Generator-marginal_cost_piecewise_chord"].scaling == 1e-3
    ).all()
    assert (m.variables["Generator-status"].scaling == 1).all()
    assert (m.variables["Link-p1_piecewise_delta"].scaling == 1).all()
    status, _ = n.optimize.solve_model()
    assert status == "ok"
    assert n.objective == pytest.approx(ref.objective, rel=RTOL)


def test_registry_covers_cvar(stochastic_benchmark_network):
    n = stochastic_benchmark_network
    n.set_risk_preference(alpha=0.9, omega=0.5)
    n.c.buses.static.scaling = 1e-3
    status, _ = n.optimize(objective_scaling=1e-3)
    assert status == "ok"
