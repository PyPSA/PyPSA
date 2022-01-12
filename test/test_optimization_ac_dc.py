import pypsa
import os
import pytest
from numpy.testing import assert_array_almost_equal as equal

@pytest.fixture
def n():
    csv_folder = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "ac-dc-meshed",
        "ac-dc-data"
    )
    n = pypsa.Network(csv_folder)
    # The linopy optimization considers the p_set of all components as an input
    n.links_t.p_set.drop(columns=n.links_t.p_set.columns, inplace=True) 
    return n


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


def test_optimization(n, n_r):
    """
    Test results were generated with GLPK; solution should be unique,
    so other solvers should not differ (e.g. cbc or gurobi)
    """
    status, _ = n.optimize(solver_name = 'glpk')

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
