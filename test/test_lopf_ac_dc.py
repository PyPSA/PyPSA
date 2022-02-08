import pypsa
import os
import pytest
from numpy.testing import assert_array_almost_equal as equal

solver_name = "glpk"


@pytest.fixture
def ac_dc_network_r():
    csv_folder = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "ac-dc-meshed",
        "ac-dc-data",
        "results-lopf",
    )
    return pypsa.Network(csv_folder)


@pytest.mark.parametrize("formulation", ["angles", "cycles", "kirchhoff", "ptdf"])
@pytest.mark.parametrize("free_memory", [{}, {"pypsa"}])
def test_lopf(ac_dc_network, ac_dc_network_r, formulation, free_memory):
    """
    Test results were generated with GLPK; solution should be unique,
    so other solvers should not differ (e.g. cbc or gurobi)
    """
    n = ac_dc_network
    n_r = ac_dc_network_r

    n.lopf(
        snapshots=n.snapshots,
        solver_name=solver_name,
        formulation=formulation,
        free_memory=free_memory,
    )

    equal(
        n.generators_t.p.loc[:, n.generators.index],
        n_r.generators_t.p.loc[:, n.generators.index],
        decimal=4,
    )

    equal(
        n.lines_t.p0.loc[:, n.lines.index],
        n_r.lines_t.p0.loc[:, n.lines.index],
        decimal=4,
    )

    equal(
        n.links_t.p0.loc[:, n.links.index],
        n_r.links_t.p0.loc[:, n.links.index],
        decimal=4,
    )


def test_lopf_lowmem(ac_dc_network, ac_dc_network_r):
    n = ac_dc_network
    n_r = ac_dc_network_r

    status, _ = n.lopf(snapshots=n.snapshots, solver_name=solver_name, pyomo=False)

    assert status == "ok"

    equal(
        n.generators_t.p.loc[:, n.generators.index],
        n_r.generators_t.p.loc[:, n.generators.index],
        decimal=2,
    )

    equal(
        n.lines_t.p0.loc[:, n.lines.index],
        n_r.lines_t.p0.loc[:, n.lines.index],
        decimal=2,
    )

    equal(
        n.links_t.p0.loc[:, n.links.index],
        n_r.links_t.p0.loc[:, n.links.index],
        decimal=2,
    )
