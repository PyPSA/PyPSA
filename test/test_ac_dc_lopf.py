from __future__ import print_function, division
from __future__ import absolute_import

import pypsa

import datetime
import pandas as pd

import networkx as nx

import numpy as np

from itertools import chain

import os



from distutils.spawn import find_executable



def test_opf():


    csv_folder_name = "../examples/ac-dc-meshed/ac-dc-data"

    network = pypsa.Network(csv_folder_name=csv_folder_name)

    results_folder_name = os.path.join(csv_folder_name,"results-lopf")

    network_r = pypsa.Network(csv_folder_name=results_folder_name)


    #test results were generated with GLPK and other solvers may differ
    solver_name = "glpk"

    snapshots = network.snapshots

    network.lopf(snapshots=snapshots,solver_name=solver_name)

    results_folder_name = "results"

    network.export_to_csv_folder(results_folder_name,time_series={"generators" : {"p" : None},
                                                                  "storage_units" : {"p" : None},
                                                                  "buses" : {"marginal_price" : None}})

    np.testing.assert_array_almost_equal(network.generators_t.p,network_r.generators_t.p)

    np.testing.assert_array_almost_equal(network.lines_t.p0,network_r.lines_t.p0)

    np.testing.assert_array_almost_equal(network.transport_links_t.p0,network_r.transport_links_t.p0)

    np.testing.assert_array_almost_equal(network.converters_t.p0,network_r.converters_t.p0)



if __name__ == "__main__":
    test_opf()
