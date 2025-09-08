# test_branch_costs_objective.py
import pandas as pd
import pypsa
import pytest


def _build_network(snaps, component, cost_A, cost_B, branch_linear=None, branch_quad=None):
    """
    2-bus setup with load on BOTH buses (1 MW each), so one side must export.
    Orientation: bus0="A", bus1="B" -> p0 > 0 means A->B.

    We allow linear and/or quadratic costs on the branch.
    """
    n = pypsa.Network()
    n.set_snapshots(snaps)

    n.add("Bus", "A")
    n.add("Bus", "B")

    # 1 MW load on both buses -> the cheaper side exports ~1 MW
    n.add("Load", "L_A", bus="A", p_set=[1.0] * len(snaps))
    n.add("Load", "L_B", bus="B", p_set=[1.0] * len(snaps))

    # Generators (large enough)
    n.add("Generator", "Gen_A", bus="A", p_nom=3.0, marginal_cost=cost_A)
    n.add("Generator", "Gen_B", bus="B", p_nom=3.0, marginal_cost=cost_B)

    # Branch A<->B
    branch_kwargs = dict(bus0="A", bus1="B", r=1e-4, x=1e-4, s_nom=100.0)
    name = "A_B"
    if component == "Line":
        n.add("Line", name, **branch_kwargs)
        # ▼▼ Adapt field names if your implementation differs ▼▼
        if branch_linear is not None:
            n.lines.at[name, "marginal_cost"] = branch_linear
        if branch_quad is not None:
            n.lines.at[name, "marginal_cost_quadratic"] = branch_quad
    elif component == "Transformer":
        n.add("Transformer", name, **branch_kwargs)
        if branch_linear is not None:
            n.transformers.at[name, "marginal_cost"] = branch_linear
        if branch_quad is not None:
            n.transformers.at[name, "marginal_cost_quadratic"] = branch_quad
    else:
        raise ValueError("component must be 'Line' or 'Transformer'")

    return n


def _solve_and_reconstruct_objective(n, component, branch_has_linear, branch_has_quad):
    status, cond = n.optimize()
    assert status == "ok", f"Optimization failed: {status} ({cond})"

    # Reconstruct the objective function:
    # Sum_t [ Σ_gen p * mc + c_branch * abs(p0) + q_branch * p0^2 ]
    snaps = n.snapshots

    # Generator part
    gen_cost = 0.0
    for g in n.generators.index:
        p = n.generators_t.p[g].reindex(snaps)
        mc = n.generators.at[g, "marginal_cost"]
        mc_series = mc if isinstance(mc, pd.Series) else pd.Series([mc] * len(snaps), index=snaps)
        gen_cost += (p * mc_series).sum()

    # Branch flow (signed, at bus0)
    if component == "Line":
        flow = abs(n.lines_t.p0["A_B"].reindex(snaps))
        attrs = n.lines
    else:
        flow = abs(n.transformers_t.p0["A_B"].reindex(snaps))
        attrs = n.transformers

    branch_cost = 0.0

    if branch_has_linear:
        c = attrs.at["A_B", "marginal_cost"]
        c_series = c if isinstance(c, pd.Series) else pd.Series([c] * len(snaps), index=snaps)
        branch_cost += (c_series * flow).sum()   # linear (signed)

    if branch_has_quad:
        q = attrs.at["A_B", "marginal_cost_quadratic"]
        q_series = q if isinstance(q, pd.Series) else pd.Series([q] * len(snaps), index=snaps)
        branch_cost += (q_series * (flow**2)).sum()  # quadratic (unsigned)

    reconstructed = gen_cost + branch_cost
    return reconstructed


# --------------------------
# Scenario definitions
# --------------------------
def _scenario_costs(snaps, scenario):
    """
    Returns (cost_A, cost_B) such that the export direction is as desired.
    - 'positive': A always cheaper -> export A->B -> p0 > 0
    - 'negative': B always cheaper -> export B->A -> p0 < 0
    - 'alternating': switching with parity
    """
    if scenario == "positive":
        cost_A = pd.Series(0.0, index=snaps)
        cost_B = pd.Series(10.0, index=snaps)
    elif scenario == "negative":
        cost_A = pd.Series(10.0, index=snaps)
        cost_B = pd.Series(0.0, index=snaps)
    elif scenario == "alternating":
        cost_A = pd.Series([0.0 if t % 2 == 0 else 10.0 for t in snaps], index=snaps)
        cost_B = pd.Series([10.0 if t % 2 == 0 else 0.0 for t in snaps], index=snaps)
    else:
        raise ValueError("unknown scenario")
    return cost_A, cost_B


# ==========================================================
# Tests: LINEAR (marginal) branch costs
# ==========================================================
@pytest.mark.parametrize("component", ["Line", "Transformer"])
@pytest.mark.parametrize("scenario", ["positive", "negative", "alternating"])
def test_objective_with_linear_branch_cost(component, scenario):
    snaps = pd.RangeIndex(6)
    cost_A, cost_B = _scenario_costs(snaps, scenario)

    # Constant linear cost per MW flow (signed). Positive c penalizes A->B,
    # rewards B->A. For the test we just check consistency with the objective.
    branch_linear = 1.0
    branch_quad = None

    n = _build_network(snaps, component, cost_A, cost_B,
                       branch_linear=branch_linear, branch_quad=branch_quad)

    reconstructed = _solve_and_reconstruct_objective(
        n, component, branch_has_linear=True, branch_has_quad=False
    )

    # Compare with solver objective
    assert n.objective == pytest.approx(reconstructed, rel=1e-8, abs=1e-8)


# ==========================================================
# Tests: QUADRATIC branch costs
# ==========================================================
@pytest.mark.parametrize("component", ["Line", "Transformer"])
@pytest.mark.parametrize("scenario", ["positive", "negative", "alternating"])
def test_objective_with_quadratic_branch_cost(component, scenario):
    snaps = pd.RangeIndex(6)
    cost_A, cost_B = _scenario_costs(snaps, scenario)

    branch_linear = None
    # Quadratic coefficient (>0). Independent of flow direction.
    branch_quad = 2.0

    n = _build_network(snaps, component, cost_A, cost_B,
                       branch_linear=branch_linear, branch_quad=branch_quad)

    reconstructed = _solve_and_reconstruct_objective(
        n, component, branch_has_linear=False, branch_has_quad=True
    )

    assert n.objective == pytest.approx(reconstructed, rel=1e-8, abs=1e-8)


# ==========================================================
# Combined (linear + quadratic)
# ==========================================================
@pytest.mark.parametrize("component", ["Line", "Transformer"])
@pytest.mark.parametrize("scenario", ["positive", "negative", "alternating"])
def test_objective_with_linear_and_quadratic_branch_cost(component, scenario):
    snaps = pd.RangeIndex(6)
    cost_A, cost_B = _scenario_costs(snaps, scenario)

    branch_linear = 1.0
    branch_quad = 2.0

    n = _build_network(snaps, component, cost_A, cost_B,
                       branch_linear=branch_linear, branch_quad=branch_quad)

    reconstructed = _solve_and_reconstruct_objective(
        n, component, branch_has_linear=True, branch_has_quad=True
    )

    assert n.objective == pytest.approx(reconstructed, rel=1e-8, abs=1e-8)