from __future__ import print_function, division
from __future__ import absolute_import

import pypsa
import os
import datetime

import pandas as pd
import networkx as nx
import numpy as np

from itertools import chain, product

from distutils.spawn import find_executable

import sys

solver_name = 'glpk' if sys.platform == 'win32' else 'cbc'

def create_lopf_version(n_teplopf):
    n_lopf = n_teplopf.copy()
    n_lopf.lines.s_nom = n_lopf.lines.s_nom_opt
    n_lopf.lines.loc[n_lopf.lines.s_nom > 0., 'operative'] = True
    n_lopf.lines.s_nom_extendable = False
    return n_lopf


def test_teplopf():

    csv_folder_name = os.path.join(os.path.dirname(__file__), "networks/tep")

    n_teplopf = pypsa.Network(csv_folder_name)

    snapshots = n_teplopf.snapshots

    for formulation, free_memory in product(["angles", "kirchhoff"],
                                            [{}, {"pypsa"}]):

        n_teplopf.teplopf(snapshots=snapshots, solver_name=solver_name,
                          formulation=formulation, free_memory=free_memory)
        n_lopf = create_lopf_version(n_teplopf)
        n_lopf.lopf(snapshots=snapshots, solver_name=solver_name,
                    formulation=formulation, free_memory=free_memory)

        np.testing.assert_array_almost_equal(
            n_teplopf.generators_t.p, n_lopf.generators_t.p, decimal=4)

        np.testing.assert_array_almost_equal(
            n_teplopf.lines_t.p0, n_lopf.lines_t.p0, decimal=4)

        np.testing.assert_array_almost_equal(
            n_teplopf.links_t.p0, n_lopf.links_t.p0, decimal=4)

        candidates = n_teplopf.lines.loc[(n_teplopf.lines.operative == False) & (
            n_teplopf.lines.s_nom_extendable == True)]
        tep_cost = (candidates.s_nom_opt * candidates.capital_cost).sum()

        np.testing.assert_almost_equal(
            n_teplopf.objective, n_lopf.objective+tep_cost, decimal=4)


if __name__ == "__main__":
    test_teplopf()
