..
  SPDX-FileCopyrightText: 2019-2020 The PyPSA-Eur Authors

  SPDX-License-Identifier: CC-BY-4.0

.. _busregions:

Rule ``build_bus_regions``
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
        3	 [color="0.25 0.6 0.85",
            label=simplify_network];
        4	 [color="0.50 0.6 0.85",
            label=add_electricity];
        5	 [color="0.36 0.6 0.85",
            fillcolor=gray,
            label=build_bus_regions,
            style=filled];
        5 -> 3;
        5 -> 4;
        9	 [color="0.22 0.6 0.85",
            label=build_renewable_profiles];
        5 -> 9;
        6	 [color="0.58 0.6 0.85",
            label=base_network];
        6 -> 5;
        8	 [color="0.28 0.6 0.85",
            label=build_shapes];
        8 -> 5;
    }

|

.. automodule:: build_bus_regions