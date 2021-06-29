import pypsa
import os
import pytest
from numpy.testing import assert_array_almost_equal as equal
import sys

solver_name = 'glpk'

@pytest.fixture
def n():
    csv_folder = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "ac-dc-meshed",
        "ac-dc-data"
    )
    return pypsa.Network(csv_folder)


@pytest.fixture
def n_r():
    csv_folder = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "ac-dc-meshed",
        "ac-dc-data",
        "results-lopf"
    )
    return pypsa.Network(csv_folder)


@pytest.mark.parametrize("formulation", ["angles", "cycles", "kirchhoff", "ptdf"])
@pytest.mark.parametrize("free_memory", [{}, {"pypsa"}])
def test_lopf(n, n_r, formulation, free_memory):
    """
    Test results were generated with GLPK; solution should be unique,
    so other solvers should not differ (e.g. cbc or gurobi)
    """

    n.lopf(
        snapshots=n.snapshots,
        solver_name=solver_name,
        formulation=formulation,
        free_memory=free_memory
    )

    equal(
        n.generators_t.p.loc[:,n.generators.index],
        n_r.generators_t.p.loc[:,n.generators.index],
        decimal=4
    )

    equal(
        n.lines_t.p0.loc[:,n.lines.index],
        n_r.lines_t.p0.loc[:,n.lines.index],
        decimal=4
    )

    equal(
        n.links_t.p0.loc[:,n.links.index],
        n_r.links_t.p0.loc[:,n.links.index],
        decimal=4
    )

def test_lopf_lowmem(n, n_r):

    status, _ = n.lopf(
        snapshots=n.snapshots,
        solver_name=solver_name,
        pyomo=False
    )

    assert status == 'ok'

    equal(
        n.generators_t.p.loc[:,n.generators.index],
        n_r.generators_t.p.loc[:,n.generators.index],
        decimal=2
    )

    equal(
        n.lines_t.p0.loc[:,n.lines.index],
        n_r.lines_t.p0.loc[:,n.lines.index],
        decimal=2
    )

    equal(
        n.links_t.p0.loc[:,n.links.index],
        n_r.links_t.p0.loc[:,n.links.index],
        decimal=2
    )
