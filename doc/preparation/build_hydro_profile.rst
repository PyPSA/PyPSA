..
  SPDX-FileCopyrightText: 2019-2020 The PyPSA-Eur Authors

  SPDX-License-Identifier: CC-BY-4.0

.. _hydroprofiles:

Rule ``build_hydro_profile``
===============================

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
        8	 [color="0.00 0.6 0.85",
            label=build_shapes];
        10	 [color="0.11 0.6 0.85",
            fillcolor=gray,
            label=build_hydro_profile,
            style=filled];
        8 -> 10;
        10 -> 4;
        13	 [color="0.56 0.6 0.85",
            label=build_cutout];
        13 -> 10;
    }

|

.. automodule:: build_hydro_profile
