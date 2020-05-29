..
  SPDX-FileCopyrightText: 2019-2020 The PyPSA-Eur Authors

  SPDX-License-Identifier: CC-BY-4.0

.. _renewableprofiles:

Rule ``build_renewable_profiles``
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
        4	 [color="0.61 0.6 0.85",
            label=add_electricity];
        5	 [color="0.19 0.6 0.85",
            label=build_bus_regions];
        9	 [color="0.22 0.6 0.85",
            fillcolor=gray,
            label=build_renewable_profiles,
            style=filled];
        5 -> 9;
        9 -> 4;
        6	 [color="0.17 0.6 0.85",
            label=base_network];
        6 -> 9;
        8	 [color="0.00 0.6 0.85",
            label=build_shapes];
        8 -> 9;
        12	 [color="0.31 0.6 0.85",
            label=build_natura_raster];
        12 -> 9;
        13	 [color="0.56 0.6 0.85",
            label=build_cutout];
        13 -> 9;
    }

|

.. automodule:: build_renewable_profiles
