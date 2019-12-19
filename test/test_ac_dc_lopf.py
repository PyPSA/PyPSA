import pypsa
from itertools import product
import os
from numpy.testing import assert_array_almost_equal as equal
import sys

solver_name = 'glpk' if sys.platform == 'win32' else 'cbc'


def test_lopf():

    csv_folder_name = os.path.join(os.path.dirname(__file__), "..", "examples",
                                   "ac-dc-meshed", "ac-dc-data")

    n = pypsa.Network(csv_folder_name)

    results_folder_name = os.path.join(csv_folder_name,"results-lopf")

    n_r = pypsa.Network(results_folder_name)

    #test results were generated with GLPK; solution should be unique,
    #so other solvers should not differ (tested with cbc and gurobi)

    snapshots = n.snapshots

    for formulation, free_memory in product(["angles", "cycles", "kirchhoff", "ptdf"],
                                            [{}, {"pypsa"}]):
        n.lopf(snapshots=snapshots, solver_name=solver_name,
               formulation=formulation, free_memory=free_memory)

        equal(n.generators_t.p.loc[:,n.generators.index],
              n_r.generators_t.p.loc[:,n.generators.index],decimal=4)
        equal(n.lines_t.p0.loc[:,n.lines.index],
              n_r.lines_t.p0.loc[:,n.lines.index],decimal=4)
        equal(n.links_t.p0.loc[:,n.links.index],
              n_r.links_t.p0.loc[:,n.links.index],decimal=4)

    if sys.version_info.major >= 3:
        status, cond = n.lopf(snapshots=snapshots, solver_name=solver_name,
                              pyomo=False)
        assert status == 'ok'
        equal(n.generators_t.p.loc[:,n.generators.index],
              n_r.generators_t.p.loc[:,n.generators.index],decimal=2)
        equal(n.lines_t.p0.loc[:,n.lines.index],
              n_r.lines_t.p0.loc[:,n.lines.index],decimal=2)
        equal(n.links_t.p0.loc[:,n.links.index],
              n_r.links_t.p0.loc[:,n.links.index],decimal=2)


if __name__ == "__main__":
    test_lopf()
