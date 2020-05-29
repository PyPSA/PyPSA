..
  SPDX-FileCopyrightText: 2019-2020 The PyPSA-Eur Authors

  SPDX-License-Identifier: CC-BY-4.0

.. _shapes:

Rule ``build_shapes``
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
        4	 [color="0.61 0.6 0.85",
            label=add_electricity];
        5	 [color="0.19 0.6 0.85",
            label=build_bus_regions];
        6	 [color="0.17 0.6 0.85",
            label=base_network];
        8	 [color="0.00 0.6 0.85",
            fillcolor=gray,
            label=build_shapes,
            style=filled];
        8 -> 4;
        8 -> 5;
        8 -> 6;
        9	 [color="0.22 0.6 0.85",
            label=build_renewable_profiles];
        8 -> 9;
        10	 [color="0.11 0.6 0.85",
            label=build_hydro_profile];
        8 -> 10;
    }

|

.. automodule:: build_shapes
