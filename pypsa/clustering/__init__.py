# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Clustering functionality for PyPSA networks."""

from typing import TYPE_CHECKING, Any

import pandas as pd
from deprecation import deprecated

from pypsa.clustering import spatial, temporal
from pypsa.clustering.spatial import SpatialClusteringMixin
from pypsa.clustering.temporal import TemporalClusteringMixin

if TYPE_CHECKING:
    from pypsa import Network
    from pypsa.clustering.spatial import Clustering


class TemporalClusteringAccessor(TemporalClusteringMixin):
    """Temporal clustering accessor for clustering a network temporally.

    Provides methods to reduce temporal resolution of networks while preserving
    total modeled hours through snapshot weighting adjustments.

    Examples
    --------
    >>> n.cluster.temporal.resample("3h")  # doctest: +SKIP
    >>> n.cluster.temporal.downsample(4)  # doctest: +SKIP
    >>> n.cluster.temporal.segment(100)  # doctest: +SKIP

    """

    def __init__(self, n: "Network") -> None:
        """Initialize the TemporalClusteringAccessor."""
        self._n = n


class SpatialClusteringAccessor(SpatialClusteringMixin):
    """Spatial clustering accessor for clustering a network spatially.

    <!-- md:guide clustering.ipynb -->
    """

    def __init__(self, n: "Network") -> None:
        """Initialize the SpatialClusteringAccessor."""
        self._n = n


class ClusteringAccessor:
    """Clustering accessor for clustering a network spatially and temporally.

    <!-- md:guide clustering.ipynb -->
    """

    def __init__(self, n: "Network") -> None:
        """Initialize the ClusteringAccessor."""
        self._n = n
        self._temporal: TemporalClusteringAccessor | None = None
        self._spatial: SpatialClusteringAccessor | None = None

    @property
    def temporal(self) -> TemporalClusteringAccessor:
        """Access temporal clustering methods.

        Returns
        -------
        TemporalClusteringAccessor
            Accessor for temporal clustering operations.

        Examples
        --------
        >>> n.cluster.temporal.resample("3h")  # doctest: +SKIP
        >>> n.cluster.temporal.downsample(4)  # doctest: +SKIP
        >>> n.cluster.temporal.segment(100)  # doctest: +SKIP

        """
        if self._temporal is None:
            self._temporal = TemporalClusteringAccessor(self._n)
        return self._temporal

    @property
    def spatial(self) -> SpatialClusteringAccessor:
        """Access spatial clustering methods.

        Returns
        -------
        SpatialClusteringAccessor
            Accessor for spatial clustering operations.

        Examples
        --------
        >>> n.cluster.spatial.busmap_by_kmeans(weighting, 50)  # doctest: +SKIP
        >>> n.cluster.spatial.cluster_by_busmap(busmap)  # doctest: +SKIP
        >>> n.cluster.spatial.cluster_by_kmeans(weighting, 50)  # doctest: +SKIP

        """
        if self._spatial is None:
            self._spatial = SpatialClusteringAccessor(self._n)
        return self._spatial

    # --- Deprecated spatial methods (use n.cluster.spatial.* instead) ---

    @deprecated(
        deprecated_in="1.1.0",
        removed_in="2.0.0",
        details="Use `n.cluster.spatial.busmap_by_hac` instead.",
    )
    def busmap_by_hac(self, *args: Any, **kwargs: Any) -> pd.Series:
        """Wrap `n.cluster.spatial.busmap_by_hac`, deprecated."""  # noqa: D401
        return self.spatial.busmap_by_hac(*args, **kwargs)

    @deprecated(
        deprecated_in="1.1.0",
        removed_in="2.0.0",
        details="Use `n.cluster.spatial.busmap_by_kmeans` instead.",
    )
    def busmap_by_kmeans(self, *args: Any, **kwargs: Any) -> pd.Series:
        """Wrap `n.cluster.spatial.busmap_by_kmeans`, deprecated."""  # noqa: D401
        return self.spatial.busmap_by_kmeans(*args, **kwargs)

    @deprecated(
        deprecated_in="1.1.0",
        removed_in="2.0.0",
        details="Use `n.cluster.spatial.busmap_by_greedy_modularity` instead.",
    )
    def busmap_by_greedy_modularity(self, *args: Any, **kwargs: Any) -> pd.Series:
        """Wrap `n.cluster.spatial.busmap_by_greedy_modularity`, deprecated."""  # noqa: D401
        return self.spatial.busmap_by_greedy_modularity(*args, **kwargs)

    @deprecated(
        deprecated_in="1.1.0",
        removed_in="2.0.0",
        details="Use `n.cluster.spatial.cluster_by_hac` instead.",
    )
    def cluster_spatially_by_hac(self, *args: Any, **kwargs: Any) -> "Network":
        """Wrap `n.cluster.spatial.cluster_by_hac`, deprecated."""  # noqa: D401
        return self.spatial.cluster_by_hac(*args, **kwargs)

    @deprecated(
        deprecated_in="1.1.0",
        removed_in="2.0.0",
        details="Use `n.cluster.spatial.cluster_by_kmeans` instead.",
    )
    def cluster_spatially_by_kmeans(self, *args: Any, **kwargs: Any) -> "Network":
        """Wrap `n.cluster.spatial.cluster_by_kmeans`, deprecated."""  # noqa: D401
        return self.spatial.cluster_by_kmeans(*args, **kwargs)

    @deprecated(
        deprecated_in="1.1.0",
        removed_in="2.0.0",
        details="Use `n.cluster.spatial.cluster_by_greedy_modularity` instead.",
    )
    def cluster_spatially_by_greedy_modularity(
        self, *args: Any, **kwargs: Any
    ) -> "Network":
        """Wrap `n.cluster.spatial.cluster_by_greedy_modularity`, deprecated."""  # noqa: D401
        return self.spatial.cluster_by_greedy_modularity(*args, **kwargs)

    @deprecated(
        deprecated_in="1.1.0",
        removed_in="2.0.0",
        details="Use `n.cluster.spatial.cluster_by_busmap` instead.",
    )
    def cluster_by_busmap(self, *args: Any, **kwargs: Any) -> "Network":
        """Wrap `n.cluster.spatial.cluster_by_busmap`, deprecated."""  # noqa: D401
        return self.spatial.cluster_by_busmap(*args, **kwargs)

    @deprecated(
        deprecated_in="1.1.0",
        removed_in="2.0.0",
        details="Use `n.cluster.spatial.get_clustering_from_busmap` instead.",
    )
    def get_clustering_from_busmap(self, *args: Any, **kwargs: Any) -> "Clustering":
        """Wrap `n.cluster.spatial.get_clustering_from_busmap`, deprecated."""  # noqa: D401
        return self.spatial.get_clustering_from_busmap(*args, **kwargs)


__all__ = ["ClusteringAccessor", "SpatialClusteringAccessor", "spatial", "temporal"]
