..
  SPDX-FileCopyrightText: 2019-2020 The PyPSA-Eur Authors

  SPDX-License-Identifier: CC-BY-4.0

.. _simplify:

Rule ``simplify_network``
============================

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
        2	 [color="0.36 0.6 0.85",
            label=cluster_network];
        3	 [color="0.14 0.6 0.85",
            fillcolor=gray,
            label=simplify_network,
            style=filled];
        3 -> 2;
        4	 [color="0.61 0.6 0.85",
            label=add_electricity];
        4 -> 3;
        5	 [color="0.19 0.6 0.85",
            label=build_bus_regions];
        5 -> 3;
    }

|

.. automodule:: simplify_network
