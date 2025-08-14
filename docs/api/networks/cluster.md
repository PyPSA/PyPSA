###########
Clustering
###########

Clustering functions which can be called within a :class:`pypsa.Network` via
``n.clustering.<func>``.

Statistic methods
~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: _source/

    ~pypsa.clustering.ClusteringAccessor.busmap_by_hac
    ~pypsa.clustering.ClusteringAccessor.busmap_by_kmeans
    ~pypsa.clustering.ClusteringAccessor.busmap_by_greedy_modularity
    ~pypsa.clustering.ClusteringAccessor.cluster_spatially_by_hac
    ~pypsa.clustering.ClusteringAccessor.cluster_spatially_by_kmeans
    ~pypsa.clustering.ClusteringAccessor.cluster_spatially_by_greedy_modularity
    ~pypsa.clustering.ClusteringAccessor.cluster_by_busmap
    ~pypsa.clustering.ClusteringAccessor.get_clustering_from_busmap
