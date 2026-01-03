# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Clustering functionality for PyPSA networks."""

from functools import wraps
from typing import TYPE_CHECKING, Any

import pandas as pd

from pypsa.clustering import spatial, temporal
from pypsa.common import _scenarios_not_implemented

if TYPE_CHECKING:
    from pypsa import Network
    from pypsa.clustering.spatial import Clustering
    from pypsa.clustering.temporal import TemporalClustering


class TemporalClusteringAccessor:
    """Temporal clustering accessor for clustering a network temporally.

    Provides methods to reduce temporal resolution of networks while preserving
    total modeled hours through snapshot weighting adjustments.

    Examples
    --------
    >>> n.cluster.temporal.resample("3h")
    >>> n.cluster.temporal.downsample(4)
    >>> n.cluster.temporal.segment(100)

    """

    def __init__(self, n: "Network") -> None:
        """Initialize the TemporalClusteringAccessor."""
        self.n = n

    @wraps(temporal.resample)
    def resample(self, *args: Any, **kwargs: Any) -> "Network":
        """Wrap [`pypsa.clustering.temporal.resample`][]."""
        return temporal.resample(self.n, *args, **kwargs).n

    @wraps(temporal.downsample)
    def downsample(self, *args: Any, **kwargs: Any) -> "Network":
        """Wrap [`pypsa.clustering.temporal.downsample`][]."""
        return temporal.downsample(self.n, *args, **kwargs).n

    @wraps(temporal.segment)
    def segment(self, *args: Any, **kwargs: Any) -> "Network":
        """Wrap [`pypsa.clustering.temporal.segment`][]."""
        return temporal.segment(self.n, *args, **kwargs).n

    @wraps(temporal.from_snapshot_map)
    def from_snapshot_map(self, *args: Any, **kwargs: Any) -> "Network":
        """Wrap [`pypsa.clustering.temporal.from_snapshot_map`][]."""
        return temporal.from_snapshot_map(self.n, *args, **kwargs).n

    def get_resample_result(self, *args: Any, **kwargs: Any) -> "TemporalClustering":
        """Get full TemporalClustering result from resample.

        Returns the full result including both the clustered network and the
        snapshot mapping. Use this when you need the snapshot_map for
        disaggregation or debugging.
        """
        return temporal.resample(self.n, *args, **kwargs)

    def get_downsample_result(self, *args: Any, **kwargs: Any) -> "TemporalClustering":
        """Get full TemporalClustering result from downsample.

        Returns the full result including both the clustered network and the
        snapshot mapping.
        """
        return temporal.downsample(self.n, *args, **kwargs)

    def get_segment_result(self, *args: Any, **kwargs: Any) -> "TemporalClustering":
        """Get full TemporalClustering result from segment.

        Returns the full result including both the clustered network and the
        snapshot mapping.
        """
        return temporal.segment(self.n, *args, **kwargs)

    def get_from_snapshot_map_result(
        self, *args: Any, **kwargs: Any
    ) -> "TemporalClustering":
        """Get full TemporalClustering result from from_snapshot_map.

        Returns the full result including both the clustered network and the
        snapshot mapping.
        """
        return temporal.from_snapshot_map(self.n, *args, **kwargs)


class ClusteringAccessor:
    """Clustering accessor for clustering a network spatially and temporally.

    <!-- md:guide clustering.ipynb -->
    """

    def __init__(self, n: "Network") -> None:
        """Initialize the ClusteringAccessor."""
        self.n = n
        self._temporal: TemporalClusteringAccessor | None = None

    @property
    def temporal(self) -> TemporalClusteringAccessor:
        """Access temporal clustering methods.

        Returns
        -------
        TemporalClusteringAccessor
            Accessor for temporal clustering operations.

        Examples
        --------
        >>> n.cluster.temporal.resample("3h")
        >>> n.cluster.temporal.downsample(4)
        >>> n.cluster.temporal.segment(100)

        """
        if self._temporal is None:
            self._temporal = TemporalClusteringAccessor(self.n)
        return self._temporal

    @_scenarios_not_implemented
    @wraps(spatial.busmap_by_hac)
    def busmap_by_hac(self, *args: Any, **kwargs: Any) -> pd.Series:
        """Wrap [`pypsa.clustering.spatial.busmap_by_hac`][]."""
        return spatial.busmap_by_hac(self.n, *args, **kwargs)

    @_scenarios_not_implemented
    @wraps(spatial.busmap_by_kmeans)
    def busmap_by_kmeans(self, *args: Any, **kwargs: Any) -> pd.Series:
        """Wrap [`pypsa.clustering.spatial.busmap_by_kmeans`][]."""
        return spatial.busmap_by_kmeans(self.n, *args, **kwargs)

    @_scenarios_not_implemented
    @wraps(spatial.busmap_by_greedy_modularity)
    def busmap_by_greedy_modularity(self, *args: Any, **kwargs: Any) -> pd.Series:
        """Wrap [`pypsa.clustering.spatial.busmap_by_greedy_modularity`][]."""
        return spatial.busmap_by_greedy_modularity(self.n, *args, **kwargs)

    @_scenarios_not_implemented
    @wraps(spatial.hac_clustering)
    def cluster_spatially_by_hac(self, *args: Any, **kwargs: Any) -> "Clustering":
        """Wrap [`pypsa.clustering.spatial.hac_clustering`][]."""
        return spatial.hac_clustering(self.n, *args, **kwargs).n

    @_scenarios_not_implemented
    @wraps(spatial.kmeans_clustering)
    def cluster_spatially_by_kmeans(self, *args: Any, **kwargs: Any) -> "Clustering":
        """Wrap [`pypsa.clustering.spatial.kmeans_clustering`][]."""
        return spatial.kmeans_clustering(self.n, *args, **kwargs).n

    @_scenarios_not_implemented
    @wraps(spatial.greedy_modularity_clustering)
    def cluster_spatially_by_greedy_modularity(
        self, *args: Any, **kwargs: Any
    ) -> "Clustering":
        """Wrap [`pypsa.clustering.spatial.greedy_modularity_clustering`][]."""
        return spatial.greedy_modularity_clustering(self.n, *args, **kwargs).n

    @_scenarios_not_implemented
    def cluster_by_busmap(self, *args: Any, **kwargs: Any) -> "Clustering":
        """Cluster the network spatially by busmap.

        This function calls [`pypsa.clustering.ClusteringAccessor.get_clustering_from_busmap`][] internally.
        For more information, see the documentation of that function.

        Returns
        -------
        n : pypsa.Network

        """
        return spatial.get_clustering_from_busmap(self.n, *args, **kwargs).n

    @_scenarios_not_implemented
    @wraps(spatial.get_clustering_from_busmap)
    def get_clustering_from_busmap(self, *args: Any, **kwargs: Any) -> "Clustering":
        """Wrap [`get_clustering_from_busmap`][pypsa.clustering.ClusteringAccessor.get_clustering_from_busmap]."""
        return spatial.get_clustering_from_busmap(self.n, *args, **kwargs)


__all__ = ["ClusteringAccessor", "spatial", "temporal"]
