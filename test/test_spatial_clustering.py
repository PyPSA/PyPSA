#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 18:11:09 2022

@author: fabian
"""

import numpy as np
import pandas as pd
from pypsa.networkclustering import (
    get_clustering_from_busmap,
    busmap_by_kmeans,
    busmap_by_hac,
)


def test_default_clustering_k_means(scipy_network):
    n = scipy_network
    # delete the 'type' specifications to make this example easier
    n.lines["type"] = np.nan
    weighting = pd.Series(1, n.buses.index)
    busmap = busmap_by_kmeans(n, bus_weightings=weighting, n_clusters=50)
    C = get_clustering_from_busmap(n, busmap)
    nc = C.network
    assert len(nc.buses) == 50


def test_default_clustering_hac(scipy_network):
    n = scipy_network
    # delete the 'type' specifications to make this example easier
    n.lines["type"] = np.nan
    busmap = busmap_by_hac(n, n_clusters=50)
    C = get_clustering_from_busmap(n, busmap)
    nc = C.network
    assert len(nc.buses) == 50
