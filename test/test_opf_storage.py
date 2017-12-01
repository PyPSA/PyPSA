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


    csv_folder_name = os.path.join(os.path.dirname(__file__), "../examples/opf-storage-hvdc/opf-storage-data")

    network = pypsa.Network(csv_folder_name)

    #test results were generated with GLPK and other solvers may differ
    solver_name = "glpk"

    snapshots = network.snapshots

    network.lopf(snapshots=snapshots,solver_name=solver_name)


    results_folder_name = "results"


    network.export_to_csv_folder(results_folder_name)

    good_results_filename =  os.path.join(csv_folder_name,"results","generators-p.csv")

    good_arr = pd.read_csv(good_results_filename,index_col=0).values

    print(good_arr)

    results_filename = os.path.join(results_folder_name,"generators-p.csv")


    arr = pd.read_csv(results_filename,index_col=0).values


    print(arr)


    np.testing.assert_array_almost_equal(arr,good_arr)



if __name__ == "__main__":
    test_opf()
