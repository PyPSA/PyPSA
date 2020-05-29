..
  SPDX-FileCopyrightText: 2019-2020 The PyPSA-Eur Authors

  SPDX-License-Identifier: CC-BY-4.0

.. _prepare:

Rule ``prepare_network``
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
        0	 [color="0.53 0.6 0.85",
            label=solve_network];
        1	 [color="0.50 0.6 0.85",
            fillcolor=gray,
            label=prepare_network,
            style=filled];
        1 -> 0;
        2	 [color="0.36 0.6 0.85",
            label=cluster_network];
        2 -> 1;
    }

|

.. automodule:: prepare_network
