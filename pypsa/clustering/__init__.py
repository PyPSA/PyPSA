#!/usr/bin/env python3
"""
Clustering functionality for PyPSA networks.
"""

from functools import wraps
from typing import TYPE_CHECKING, Any

import pandas as pd

from pypsa.clustering import spatial, temporal

if TYPE_CHECKING:
    from pypsa import Network
    from pypsa.clustering.spatial import Clustering


class ClusteringAccessor:
    """
    Clustering accessor for clustering a network spatially and temporally.
    """

    def __init__(self, network: "Network") -> None:
        self._parent = network

    @wraps(spatial.busmap_by_hac)
    def busmap_by_hac(self, *args: Any, **kwargs: Any) -> pd.Series:
        return spatial.busmap_by_hac(self._parent, *args, **kwargs)

    @wraps(spatial.busmap_by_kmeans)
    def busmap_by_kmeans(self, *args: Any, **kwargs: Any) -> pd.Series:
        return spatial.busmap_by_kmeans(self._parent, *args, **kwargs)

    @wraps(spatial.busmap_by_greedy_modularity)
    def busmap_by_greedy_modularity(self, *args: Any, **kwargs: Any) -> pd.Series:
        return spatial.busmap_by_greedy_modularity(self._parent, *args, **kwargs)

    @wraps(spatial.hac_clustering)
    def cluster_spatially_by_hac(self, *args: Any, **kwargs: Any) -> "Clustering":
        return spatial.hac_clustering(self._parent, *args, **kwargs).network

    @wraps(spatial.kmeans_clustering)
    def cluster_spatially_by_kmeans(self, *args: Any, **kwargs: Any) -> "Clustering":
        return spatial.kmeans_clustering(self._parent, *args, **kwargs).network

    @wraps(spatial.greedy_modularity_clustering)
    def cluster_spatially_by_greedy_modularity(
        self, *args: Any, **kwargs: Any
    ) -> "Clustering":
        return spatial.greedy_modularity_clustering(
            self._parent, *args, **kwargs
        ).network

    def cluster_by_busmap(self, *args: Any, **kwargs: Any) -> "Clustering":
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
    def get_clustering_from_busmap(self, *args: Any, **kwargs: Any) -> "Clustering":
        return spatial.get_clustering_from_busmap(self._parent, *args, **kwargs)


__all__ = ["ClusteringAccessor", "spatial", "temporal"]
