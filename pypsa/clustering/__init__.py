"""Clustering functionality for PyPSA networks."""

from functools import wraps
from typing import TYPE_CHECKING, Any

import pandas as pd

from pypsa.clustering import spatial, temporal
from pypsa.common import deprecated_common_kwargs

if TYPE_CHECKING:
    from pypsa import Network
    from pypsa.clustering.spatial import Clustering


class ClusteringAccessor:
    """Clustering accessor for clustering a network spatially and temporally."""

    @deprecated_common_kwargs
    def __init__(self, n: "Network") -> None:
        """Initialize the ClusteringAccessor."""
        self.n = n

    @wraps(spatial.busmap_by_hac)
    def busmap_by_hac(self, *args: Any, **kwargs: Any) -> pd.Series:
        """Wrap :func:`pypsa.clustering.spatial.busmap_by_hac`."""
        return spatial.busmap_by_hac(self.n, *args, **kwargs)

    @wraps(spatial.busmap_by_kmeans)
    def busmap_by_kmeans(self, *args: Any, **kwargs: Any) -> pd.Series:
        """Wrap :func:`pypsa.clustering.spatial.busmap_by_kmeans`."""
        return spatial.busmap_by_kmeans(self.n, *args, **kwargs)

    @wraps(spatial.busmap_by_greedy_modularity)
    def busmap_by_greedy_modularity(self, *args: Any, **kwargs: Any) -> pd.Series:
        """Wrap :func:`pypsa.clustering.spatial.busmap_by_greedy_modularity`."""
        return spatial.busmap_by_greedy_modularity(self.n, *args, **kwargs)

    @wraps(spatial.hac_clustering)
    def cluster_spatially_by_hac(self, *args: Any, **kwargs: Any) -> "Clustering":
        """Wrap :func:`pypsa.clustering.spatial.hac_clustering`."""
        return spatial.hac_clustering(self.n, *args, **kwargs).n

    @wraps(spatial.kmeans_clustering)
    def cluster_spatially_by_kmeans(self, *args: Any, **kwargs: Any) -> "Clustering":
        """Wrap :func:`pypsa.clustering.spatial.kmeans_clustering`."""
        return spatial.kmeans_clustering(self.n, *args, **kwargs).n

    @wraps(spatial.greedy_modularity_clustering)
    def cluster_spatially_by_greedy_modularity(
        self, *args: Any, **kwargs: Any
    ) -> "Clustering":
        """Wrap :func:`pypsa.clustering.spatial.greedy_modularity_clustering`."""
        return spatial.greedy_modularity_clustering(self.n, *args, **kwargs).n

    def cluster_by_busmap(self, *args: Any, **kwargs: Any) -> "Clustering":
        """Cluster the network spatially by busmap.

        This function calls :func:`pypsa.clustering.spatial.get_clustering_from_busmap` internally.
        For more information, see the documentation of that function.

        Returns
        -------
        n : pypsa.Network

        """
        return spatial.get_clustering_from_busmap(self.n, *args, **kwargs).n

    @wraps(spatial.get_clustering_from_busmap)
    def get_clustering_from_busmap(self, *args: Any, **kwargs: Any) -> "Clustering":
        """Wrap :func:`pypsa.clustering.spatial.get_clustering_from_busmap`."""
        return spatial.get_clustering_from_busmap(self.n, *args, **kwargs)


__all__ = ["ClusteringAccessor", "spatial", "temporal"]
