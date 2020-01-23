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

import sys

solver_name = 'glpk' if sys.platform == 'win32' else 'cbc'


def test_operative_line_flag_lopf():

    csv_folder_name = os.path.join(os.path.dirname(__file__), "../examples/ac-dc-meshed/ac-dc-data")

    network = pypsa.Network(csv_folder_name)

    network_op = network.copy()

    # need two lines to create potential new cycle

    network_op.add("Line", "FRA-LND",
        operative=False,
        bus0="Frankfurt",
        bus1="London",
        x=0.5,
        s_nom_extendable=True,
        s_nom=10000,
        capital_cost = 0.01)

    network_op.add("Line", "FRA-NRW",
        operative=False,
        bus0="Frankfurt",
        bus1="Norwich",
        x=0.5,
        s_nom_extendable=True,
        s_nom=10000,
        capital_cost = 0.01)

    snapshots = network.snapshots

    for formulation, free_memory in product(["angles", "cycles", "kirchhoff", "ptdf"],
                                            [{}, {"pypsa"}]):
        network.lopf(snapshots=snapshots,solver_name=solver_name,formulation=formulation, free_memory=free_memory)
        network_op.lopf(snapshots=snapshots,solver_name=solver_name,formulation=formulation, free_memory=free_memory)

        np.testing.assert_array_almost_equal(network.generators_t.p.loc[:,network.generators.index],network_op.generators_t.p.loc[:,network.generators.index],decimal=4)

        np.testing.assert_array_almost_equal(network.lines_t.p0.loc[:,network.lines.index],network_op.lines_t.p0.loc[:,network.lines.index],decimal=4)

        np.testing.assert_array_almost_equal(network.links_t.p0.loc[:,network.links.index],network_op.links_t.p0.loc[:,network.links.index],decimal=4)

if __name__ == "__main__":
    test_operative_line_flag_lopf()
