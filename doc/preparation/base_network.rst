..
  SPDX-FileCopyrightText: 2019-2020 The PyPSA-Eur Authors

  SPDX-License-Identifier: CC-BY-4.0
  
.. _base:

Rule ``base_network``
=============================

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
        4	 [color="0.50 0.6 0.85",
            label=add_electricity];
        5	 [color="0.36 0.6 0.85",
            label=build_bus_regions];
        6	 [color="0.58 0.6 0.85",
            fillcolor=gray,
            label=base_network,
            style=filled];
        6 -> 4;
        6 -> 5;
        7	 [color="0.31 0.6 0.85",
            label=build_powerplants];
        6 -> 7;
        9	 [color="0.22 0.6 0.85",
            label=build_renewable_profiles];
        6 -> 9;
        8	 [color="0.28 0.6 0.85",
            label=build_shapes];
        8 -> 6;
        11	 [color="0.03 0.6 0.85",
            label=prepare_links_p_nom];
        11 -> 6;
    }

|

.. automodule:: base_network