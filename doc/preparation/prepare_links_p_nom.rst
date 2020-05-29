..
  SPDX-FileCopyrightText: 2019-2020 The PyPSA-Eur Authors

  SPDX-License-Identifier: CC-BY-4.0

.. _links:

Rule ``prepare_links_p_nom``
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
        6	 [color="0.17 0.6 0.85",
            label=base_network];
        11	 [color="0.39 0.6 0.85",
            fillcolor=gray,
            label=prepare_links_p_nom,
            style=filled];
        11 -> 6;
    }

|

.. automodule:: prepare_links_p_nom
