..
  SPDX-FileCopyrightText: 2019-2020 The PyPSA-Eur Authors

  SPDX-License-Identifier: CC-BY-4.0

.. _cluster:

Rule ``cluster_network``
===========================

.. graphviz::
    :align: center

    digraph snakemake_dag {
        graph [bgcolor=white,
            margin=0,
            size="8,5"
        ];
        node [fontname=sans,
            fontsize=10,
            penwidth=2,
            shape=box,
            style=rounded
        ];
        edge [color=grey,
            penwidth=2
        ];
        1	 [color="0.50 0.6 0.85",
            label=prepare_network];
        2	 [color="0.36 0.6 0.85",
            fillcolor=gray,
            label=cluster_network,
            style=filled];
        2 -> 1;
        3	 [color="0.14 0.6 0.85",
            label=simplify_network];
        3 -> 2;
    }


|

.. automodule:: cluster_network
