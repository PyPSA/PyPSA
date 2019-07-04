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


def test_operational_line_flag_lpf():

    csv_folder_name = os.path.join(os.path.dirname(__file__), "../examples/ac-dc-meshed/ac-dc-data")

    network = pypsa.Network(csv_folder_name)

    network_op = network.copy()

    # need two lines to create potential new cycle

    network_op.add("Line", "FRA-LND",
        operational=False,
        bus0="Frankfurt",
        bus1="London",
        x=0.5,
        s_nom_extendable=True,
        s_nom=10000,
        capital_cost = 0.01)

    network_op.add("Line", "FRA-NRW",
        operational=False,
        bus0="Frankfurt",
        bus1="Norwich",
        x=0.5,
        s_nom_extendable=True,
        s_nom=10000,
        capital_cost = 0.01)

    for snapshot in network.snapshots[:2]:
        network.lpf(snapshot)
        network_op.lpf(snapshot)

    np.testing.assert_array_almost_equal(network.generators_t.p[network.generators.index].iloc[:2],network_op.generators_t.p[network.generators.index].iloc[:2])
    np.testing.assert_array_almost_equal(network.lines_t.p0[network.lines.index].iloc[:2],network_op.lines_t.p0[network.lines.index].iloc[:2])
    np.testing.assert_array_almost_equal(network.links_t.p0[network.links.index].iloc[:2],network_op.links_t.p0[network.links.index].iloc[:2])


    network.lpf(snapshots=network.snapshots)
    network_op.lpf(snapshots=network.snapshots)

    np.testing.assert_array_almost_equal(network.generators_t.p[network.generators.index],network_op.generators_t.p[network.generators.index])
    np.testing.assert_array_almost_equal(network.lines_t.p0[network.lines.index],network_op.lines_t.p0[network.lines.index])
    np.testing.assert_array_almost_equal(network.links_t.p0[network.links.index],network_op.links_t.p0[network.links.index])

if __name__ == "__main__":
    test_operational_line_flag_lpf()
