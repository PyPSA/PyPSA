.. _trace_solve:

Rule ``trace_solve_network``
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
        0	 [color="0.17 0.6 0.85",
            fillcolor=gray,
            label=trace_solve_network,
            style=filled];
        1	 [color="0.58 0.6 0.85",
            label=prepare_network];
        1 -> 0;
    }

|

.. automodule:: trace_solve_network
