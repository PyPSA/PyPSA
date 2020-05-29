..
  SPDX-FileCopyrightText: 2019-2020 The PyPSA-Eur Authors

  SPDX-License-Identifier: CC-BY-4.0

.. _cutout:

Rule ``build_cutout``
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
        9	 [color="0.22 0.6 0.85",
            label=build_renewable_profiles];
        10	 [color="0.44 0.6 0.85",
            label=build_hydro_profile];
        13	 [color="0.17 0.6 0.85",
            fillcolor=gray,
            label=build_cutout,
            style=filled];
        13 -> 9;
        13 -> 10;
    }

|

.. automodule:: build_cutout