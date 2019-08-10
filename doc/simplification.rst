

##########################################
Simplifying Networks
##########################################

The simplification ``snakemake`` rule prepare approximations of the full model, in which it is computationally viable to co-optimize generation, storage and transmission capacities.

- ``simplify_network`` transforms the transmission grid to a 380 kV only equivalent network, while
- ``cluster_network`` uses a `k-means <https://en.wikipedia.org/wiki/K-means_clustering>`_ based clustering technique to partition the network into a given number of zones and then reduce the network to a representation with one bus per zone.

The simplification and clustering steps are described in detail in the paper
[The role of spatial scale in joint optimisations of generation and transmission for European highly renewable scenarios](https://arxiv.org/abs/1705.07617), 2017, [arXiv:1705.07617](https://arxiv.org/abs/1705.07617), [doi:10.1109/EEM.2017.7982024](https://doi.org/10.1109/EEM.2017.7982024).

.. bibliography:: references.bib

Simplify Network
================

.. automodule:: simplify_network

Cluster Network
===============

.. automodule:: cluster_network

Prepare Network
===============

.. automodule:: prepare_network
