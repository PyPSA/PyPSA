..
  SPDX-FileCopyrightText: 2019-2020 The PyPSA-Eur Authors

  SPDX-License-Identifier: CC-BY-4.0

.. _solve:

Rule ``solve_network``
=========================

.. graphviz::
    :align: center

    digraph snakemake_dag {
        graph [bgcolor=white,
            margin=0,
            size="3,3"
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
        0	 [color="0.64 0.6 0.85",
            fillcolor=gray,
            label=solve_network,
            style=filled];
        1	 [color="0.33 0.6 0.85",
            label=prepare_network];
        1 -> 0;
    }

|

.. automodule:: solve_network
