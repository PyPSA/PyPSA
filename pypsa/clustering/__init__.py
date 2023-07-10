#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clustering functionality for PyPSA networks.
"""

from functools import wraps

from pypsa.clustering import spatial, temporal


class ClusteringAccessor:
    """
    Clustering accessor for clustering a network spatially and temporally.
    """

    def __init__(self, network):
        self._parent = network

    @wraps(spatial.busmap_by_hac)
    def busmap_by_hac(self, *args, **kwargs):
        return spatial.busmap_by_hac(self._parent, *args, **kwargs)

    @wraps(spatial.busmap_by_kmeans)
    def busmap_by_kmeans(self, *args, **kwargs):
        return spatial.busmap_by_kmeans(self._parent, *args, **kwargs)

    @wraps(spatial.busmap_by_greedy_modularity)
    def busmap_by_greedy_modularity(self, *args, **kwargs):
        return spatial.busmap_by_greedy_modularity(self._parent, *args, **kwargs)

    @wraps(spatial.hac_clustering)
    def cluster_spatially_by_hac(self, *args, **kwargs):
        return spatial.hac_clustering(self._parent, *args, **kwargs).network

    @wraps(spatial.kmeans_clustering)
    def cluster_spatially_by_kmeans(self, *args, **kwargs):
        return spatial.kmeans_clustering(self._parent, *args, **kwargs).network

    @wraps(spatial.greedy_modularity_clustering)
    def cluster_spatially_by_greedy_modularity(self, *args, **kwargs):
        return spatial.greedy_modularity_clustering(
            self._parent, *args, **kwargs
        ).network

    def cluster_by_busmap(self, *args, **kwargs):
        """
        Cluster the network spatially by busmap.

        This function calls :func:`pypsa.clustering.spatial.get_clustering_from_busmap` internally.
        For more information, see the documentation of that function.

        Returns
        -------
        network : pypsa.Network
        """
        return spatial.get_clustering_from_busmap(self._parent, *args, **kwargs).network

    @wraps(spatial.get_clustering_from_busmap)
    def get_clustering_from_busmap(self, *args, **kwargs):
        return spatial.get_clustering_from_busmap(self._parent, *args, **kwargs)
