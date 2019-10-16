from __future__ import print_function, division
from __future__ import absolute_import

import pypsa

import pandas as pd

from itertools import product

import os

from numpy.testing import assert_array_almost_equal as equal



def test_opf(pyomo=True):

    csv_folder_name = os.path.join(os.path.dirname(__file__),
                                   "../examples/opf-storage-hvdc/opf-storage-data")

    n = pypsa.Network(csv_folder_name)

    target_path = os.path.join(csv_folder_name,"results","generators-p.csv")

    target_gen_p = pd.read_csv(target_path, index_col=0)

    #test results were generated with GLPK and other solvers may differ
    for solver_name, pyomo in product(["cbc", "glpk"], [True, False]):
        solver_name = "glpk"

        n.lopf(solver_name=solver_name, pyomo=pyomo)

        equal(n.generators_t.p.reindex_like(target_gen_p), target_gen_p, decimal=2)


if __name__ == "__main__":
    test_opf()
