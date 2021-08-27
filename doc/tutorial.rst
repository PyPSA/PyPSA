..
  SPDX-FileCopyrightText: 2019-2020 The PyPSA-Eur Authors

  SPDX-License-Identifier: CC-BY-4.0

.. _tutorial:

#####################
Tutorial
#####################

.. raw:: html

    <iframe width="832" height="468" src="https://www.youtube.com/embed/mAwhQnNRIvs" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

Before getting started with **PyPSA-Eur** it makes sense to be familiar
with its general modelling framework `PyPSA <https://pypsa.readthedocs.io>`__.

Running the tutorial requires limited computational resources compared to the full model,
which allows the user to explore most of its functionalities on a local machine.
It takes approximately five minutes to complete and
requires 3 GB of memory along with 1 GB free disk space.

If not yet completed, follow the :ref:`installation` steps first.

The tutorial will cover examples on how to

- configure and customise the PyPSA-Eur model and
- run the ``snakemake`` workflow step by step from network creation to the solved network.

The configuration of the tutorial is included in the ``config.tutorial.yaml``.
To run the tutorial, use this as your configuration file ``config.yaml``.

.. code:: bash

    .../pypsa-eur % cp config.tutorial.yaml config.yaml

This configuration is set to download a reduced data set via the rules :mod:`retrieve_databundle`,
:mod:`retrieve_natura_raster`, :mod:`retrieve_cutout` totalling at less than 250 MB.
The full set of data dependencies would consume 5.3 GB.
For more information on the data dependencies of PyPSA-Eur, continue reading :ref:`data`.

How to customise PyPSA-Eur?
===========================

The model can be adapted to only include selected countries (e.g. Germany) instead of all European countries to limit the spatial scope.

.. literalinclude:: ../config.tutorial.yaml
   :language: yaml
   :lines: 20

Likewise, the example's temporal scope can be restricted (e.g. to a single month).

.. literalinclude:: ../config.tutorial.yaml
   :language: yaml
   :lines: 24-27

It is also possible to allow less or more carbon-dioxide emissions. Here, we limit the emissions of Germany 100 Megatonnes per year.

.. literalinclude:: ../config.tutorial.yaml
   :language: yaml
   :lines: 38,40

PyPSA-Eur also includes a database of existing conventional powerplants.
We can select which types of powerplants we like to be included with fixed capacities:

.. literalinclude:: ../config.tutorial.yaml
   :language: yaml
   :lines: 38,54

To accurately model the temporal and spatial availability of renewables such as wind and solar energy, we rely on historical weather data.
It is advisable to adapt the required range of coordinates to the selection of countries.

.. literalinclude:: ../config.tutorial.yaml
   :language: yaml
   :lines: 56-63

We can also decide which weather data source should be used to calculate potentials and capacity factor time-series for each carrier.
For example, we may want to use the ERA-5 dataset for solar and not the default SARAH-2 dataset.

.. literalinclude:: ../config.tutorial.yaml
   :language: yaml
   :lines: 65,108-109

Finally, it is possible to pick a solver. For instance, this tutorial uses the open-source solvers CBC and Ipopt and does not rely
on the commercial solvers Gurobi or CPLEX (for which free academic licenses are available).

.. literalinclude:: ../config.tutorial.yaml
   :language: yaml
   :lines: 171,181-182

.. note::

    To run the tutorial, either install CBC and Ipopt (see instructions for :ref:`installation`).

    Alternatively, choose another installed solver in the ``config.yaml`` at ``solving: solver:``.

Note, that we only note major changes to the provided default configuration that is comprehensibly documented in :ref:`config`.
There are many more configuration options beyond what is adapted for the tutorial!

How to use the ``snakemake`` rules?
===================================

Open a terminal, go into the PyPSA-Eur directory, and activate the ``pypsa-eur`` environment with

.. code:: bash

    .../pypsa-eur % conda activate pypsa-eur

Let's say based on the modifications above we would like to solve a very simplified model
clustered down to 6 buses and every 24 hours aggregated to one snapshot. The command

.. code:: bash

    .../pypsa-eur % snakemake -j 1 results/networks/elec_s_6_ec_lcopt_Co2L-24H.nc

orders ``snakemake`` to run the script ``solve_network`` that produces the solved network and stores it in ``.../pypsa-eur/results/networks`` with the name ``elec_s_6_ec_lcopt_Co2L-24H.nc``:

.. code::

    rule solve_network:
        input: "networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc"
        output: "results/networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc"
        [...]
        script: "scripts/solve_network.py"

.. until https://github.com/snakemake/snakemake/issues/46 closed

.. warning::
    On Windows the previous command may currently cause a ``MissingRuleException`` due to problems with output files in subfolders.
    This is an `open issue <https://github.com/snakemake/snakemake/issues/46>`_ at `snakemake <https://snakemake.readthedocs.io/>`_.
    Windows users should add the option ``--keep-target-files`` to the command or instead run ``snakemake -j 1 solve_all_networks``.

This triggers a workflow of multiple preceding jobs that depend on each rule's inputs and outputs:

.. graphviz::
    :align: center

    digraph snakemake_dag {
        graph[bgcolor=white, margin=0];
        node[shape=box, style=rounded, fontname=sans, fontsize=10, penwidth=2];
        edge[penwidth=2, color=grey];
        0[label = "solve_network", color = "0.10 0.6 0.85", style="rounded"];
        1[label = "prepare_network\nll: copt\nopts: Co2L-24H", color = "0.13 0.6 0.85", style="rounded"];
        2[label = "cluster_network\nclusters: 6", color = "0.51 0.6 0.85", style="rounded"];
        3[label = "simplify_network\nnetwork: elec\nsimpl: ", color = "0.00 0.6 0.85", style="rounded"];
        4[label = "add_electricity", color = "0.60 0.6 0.85", style="rounded"];
        5[label = "build_bus_regions", color = "0.19 0.6 0.85", style="rounded"];
        6[label = "base_network", color = "0.38 0.6 0.85", style="rounded"];
        7[label = "build_shapes", color = "0.03 0.6 0.85", style="rounded"];
        8[label = "build_renewable_profiles\ntechnology: onwind", color = "0.48 0.6 0.85", style="rounded"];
        9[label = "build_renewable_profiles\ntechnology: offwind-ac", color = "0.48 0.6 0.85", style="rounded"];
        10[label = "build_renewable_profiles\ntechnology: offwind-dc", color = "0.48 0.6 0.85", style="rounded"];
        11[label = "build_renewable_profiles\ntechnology: solar", color = "0.48 0.6 0.85", style="rounded"];
        12[label = "build_cutout\ncutout: europe-2013-era5", color = "0.35 0.6 0.85", style="rounded,dashed"];
        1 -> 0
        2 -> 1
        3 -> 2
        4 -> 3
        5 -> 3
        6 -> 4
        5 -> 4
        7 -> 4
        8 -> 4
        9 -> 4
        10 -> 4
        11 -> 4
        7 -> 5
        6 -> 5
        7 -> 6
        6 -> 8
        7 -> 8
        5 -> 8
        12 -> 8
        6 -> 9
        7 -> 9
        5 -> 9
        12 -> 9
        6 -> 10
        7 -> 10
        5 -> 10
        12 -> 10
        6 -> 11
        7 -> 11
        5 -> 11
        12 -> 11
    }

|

In the terminal, this will show up as a list of jobs to be run:

.. code:: bash

    Building DAG of jobs...
    Using shell: /bin/bash
    Provided cores: 1
    Rules claiming more threads will be scaled down.
    Unlimited resources: mem
    Job counts:
        count	jobs
        1	add_electricity
        1	base_network
        1	build_bus_regions
        4	build_renewable_profiles
        1	build_shapes
        1	cluster_network
        1	prepare_network
        1	simplify_network
        1	solve_network
        12

``snakemake`` then runs these jobs in the correct order.

A job (here ``simplify_network``) will display its attributes and normally some logs in the terminal:

.. code:: bash

    [<DATETIME>]
    rule simplify_network:
        input: networks/elec.nc, data/costs.csv, resources/regions_onshore.geojson, resources/regions_offshore.geojson
        output: networks/elec_s.nc, resources/regions_onshore_elec_s.geojson, resources/regions_offshore_elec_s.geojson, resources/clustermaps_elec_s.h5
        jobid: 3
        benchmark: benchmarks/simplify_network/elec_s
        wildcards: network=elec, simpl=
        resources: mem=4000

    INFO:pypsa.io:Imported network elec.nc has buses, carriers, generators, lines, links, loads, storage_units, transformers
    INFO:__main__:Mapping all network lines onto a single 380kV layer
    INFO:__main__:Simplifying connected link components
    INFO:__main__:Removing stubs
    INFO:__main__:Displacing offwind-ac generator(s) and adding connection costs to capital_costs: 20128 Eur/MW/a for `5718 offwind-ac`
    INFO:__main__:Displacing offwind-dc generator(s) and adding connection costs to capital_costs: 14994 Eur/MW/a for `5718 offwind-dc`, 26939 Eur/MW/a for `5724 offwind-dc`, 29621 Eur/MW/a for `5725 offwind-dc`
    INFO:pypsa.io:Exported network elec_s.nc has lines, carriers, links, storage_units, loads, buses, generators
    [<DATETIME>]
    Finished job 3.
    9 of 12 steps (75%) done

Once the whole worktree is finished, it should show state so in the terminal:

.. code:: bash

    Finished job 0.
    12 of 12 steps (100%) done
    Complete log: /home/XXXX/pypsa-eur/.snakemake/log/20XX-XX-XXTXX.snakemake.log
    snakemake results/networks/elec_s_6_ec_lcopt_Co2L-24H.nc  519,84s user 34,26s system 242% cpu 3:48,83 total

You will notice that many intermediate stages are saved, namely the outputs of each individual ``snakemake`` rule.

You can produce any output file occuring in the ``Snakefile`` by running

.. code:: bash

    .../pypsa-eur % snakemake -j 1 <output file>

For example, you can explore the evolution of the PyPSA networks by running

#. ``.../pypsa-eur % snakemake -j 1 networks/base.nc``
#. ``.../pypsa-eur % snakemake -j 1 networks/elec.nc``
#. ``.../pypsa-eur % snakemake -j 1 networks/elec_s.nc``
#. ``.../pypsa-eur % snakemake -j 1 networks/elec_s_6.nc``
#. ``.../pypsa-eur % snakemake -j 1 networks/elec_s_6_ec_lcopt_Co2L-24H.nc``

There's a special rule: If you simply run

.. code:: bash

    .../pypsa-eur % snakemake

the wildcards given in ``scenario`` in the configuration file ``config.yaml`` are used:

.. literalinclude:: ../config.tutorial.yaml
   :language: yaml
   :lines: 14-18

In this example we would not only solve a 6-node model of Germany but also a 2-node model.

How to analyse solved networks?
===============================

The solved networks can be analysed just like any other PyPSA network (e.g. in Jupyter Notebooks).

.. code:: python

    import pypsa

    network = pypsa.Network("results/networks/elec_s_6_ec_lcopt_Co2L-24H.nc")

For inspiration, read the `examples section in the PyPSA documentation <https://pypsa.readthedocs.io/en/latest/examples.html>`_.
