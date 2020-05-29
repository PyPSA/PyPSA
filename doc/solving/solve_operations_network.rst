..
  SPDX-FileCopyrightText: 2019-2020 The PyPSA-Eur Authors

  SPDX-License-Identifier: CC-BY-4.0

.. _solve_operations:

Rule ``solve_operations_network``
====================================

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
        0	 [color="0.06 0.6 0.85",
            fillcolor=gray,
            label=solve_operations_network,
            style=filled];
        1	 [color="0.00 0.6 0.85",
            label=cluster_network];
        1 -> 0;
        2	 [color="0.19 0.6 0.85",
            label=solve_network];
        2 -> 0;
    }

|

.. automodule:: solve_operations_network
