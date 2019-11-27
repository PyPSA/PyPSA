
import pypsa
import pandas as pd
import sys
import os
from numpy.testing import assert_array_almost_equal as equal

solvers = ['glpk'] if sys.platform == 'win32' else ['cbc', 'glpk']


def test_opf():

    csv_folder_name = os.path.join(os.path.dirname(__file__), "..", "examples",
                                   "opf-storage-hvdc","opf-storage-data")

    n = pypsa.Network(csv_folder_name)

    target_path = os.path.join(csv_folder_name,"results","generators-p.csv")

    target_gen_p = pd.read_csv(target_path, index_col=0)

    #test results were generated with GLPK and other solvers may differ
    for solver_name in solvers:

        n.lopf(solver_name=solver_name, pyomo=True)

        equal(n.generators_t.p.reindex_like(target_gen_p), target_gen_p, decimal=2)

    if sys.version_info.major >= 3:

        for solver_name in solvers:

            status, cond = n.lopf(solver_name=solver_name, pyomo=False)
            assert status == 'ok'
            equal(n.generators_t.p.reindex_like(target_gen_p), target_gen_p,
                  decimal=2)


if __name__ == "__main__":
    test_opf()
