from __future__ import print_function, division
from __future__ import absolute_import

import pypsa

import datetime
import pandas as pd

import networkx as nx

import numpy as np

from itertools import chain, product

import os



from distutils.spawn import find_executable



def test_lopf():


    csv_folder_name = "../examples/ac-dc-meshed/ac-dc-data"

    network = pypsa.Network(csv_folder_name=csv_folder_name)

    results_folder_name = os.path.join(csv_folder_name,"results-lopf")

    network_r = pypsa.Network(csv_folder_name=results_folder_name)


    #test results were generated with GLPK; solution should be unique,
    #so other solvers should not differ (tested with cbc and gurobi)
    solver_name = "cbc"

    snapshots = network.snapshots

    for formulation, free_memory in product(["angles", "cycles", "kirchhoff", "ptdf"],
                                            [{}, {"pypsa"}, {"pypsa", "pyomo-hack"}]):
        network.lopf(snapshots=snapshots,solver_name=solver_name,formulation=formulation, free_memory=free_memory)
        print(network.generators_t.p.loc[:,network.generators.index])
        print(network_r.generators_t.p.loc[:,network.generators.index])

        np.testing.assert_array_almost_equal(network.generators_t.p.loc[:,network.generators.index],network_r.generators_t.p.loc[:,network.generators.index],decimal=4)

        np.testing.assert_array_almost_equal(network.lines_t.p0.loc[:,network.lines.index],network_r.lines_t.p0.loc[:,network.lines.index],decimal=4)

        np.testing.assert_array_almost_equal(network.links_t.p0.loc[:,network.links.index],network_r.links_t.p0.loc[:,network.links.index],decimal=4)



if __name__ == "__main__":
    test_lopf()
