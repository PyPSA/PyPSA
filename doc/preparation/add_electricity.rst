..
  SPDX-FileCopyrightText: 2019-2020 The PyPSA-Eur Authors

  SPDX-License-Identifier: CC-BY-4.0

.. _electricity:

Rule ``add_electricity``
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
            fillcolor=gray,
            label=add_electricity,
            style=filled];
        4 -> 3;
        5	 [color="0.36 0.6 0.85",
            label=build_bus_regions];
        5 -> 4;
        6	 [color="0.58 0.6 0.85",
            label=base_network];
        6 -> 4;
        7	 [color="0.31 0.6 0.85",
            label=build_powerplants];
        7 -> 4;
        8	 [color="0.28 0.6 0.85",
            label=build_shapes];
        8 -> 4;
        9	 [color="0.22 0.6 0.85",
            label=build_renewable_profiles];
        9 -> 4;
        10	 [color="0.44 0.6 0.85",
            label=build_hydro_profile];
        10 -> 4;
    }

|

.. automodule:: add_electricity
