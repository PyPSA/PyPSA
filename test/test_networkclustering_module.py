#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests continued functioning of deprecated `networkclustering` module.
"""

import numpy as np
import pandas as pd

from pypsa.networkclustering import busmap_by_kmeans, get_clustering_from_busmap


def test_default_clustering_k_means(scipy_network):
    n = scipy_network
    # delete the 'type' specifications to make this example easier
    n.lines["type"] = np.nan
    weighting = pd.Series(1, n.buses.index)
    busmap = busmap_by_kmeans(n, bus_weightings=weighting, n_clusters=50)
    C = get_clustering_from_busmap(n, busmap)
    nc = C.network
    assert len(nc.buses) == 50
