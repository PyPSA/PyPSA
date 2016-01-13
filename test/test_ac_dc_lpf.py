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

    results_folder_name = os.path.join(csv_folder_name,"results-lpf")

    network_r = pypsa.Network(csv_folder_name=results_folder_name)

    for snapshot in network.snapshots:
        network.lpf(snapshot)

    np.testing.assert_array_almost_equal(network.generators.p,network_r.generators.p)

    np.testing.assert_array_almost_equal(network.lines.p0,network_r.lines.p0)

    np.testing.assert_array_almost_equal(network.transport_links.p0,network_r.transport_links.p0)

    np.testing.assert_array_almost_equal(network.converters.p0,network_r.converters.p0)



if __name__ == "__main__":
    test_opf()
