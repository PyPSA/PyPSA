..
  SPDX-FileCopyrightText: 2019-2020 The PyPSA-Eur Authors

  SPDX-License-Identifier: CC-BY-4.0

.. _intro:

##########################################
 Introduction
##########################################

.. raw:: html

    <iframe width="832" height="468" src="https://www.youtube.com/embed/ty47YU1_eeQ" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

Find the introductory slides `here <https://docs.google.com/presentation/d/e/2PACX-1vQGQZD7KIVdocRZzRVu8Uk-JC_ltEow5zjtIarhyws46IMJpaqGuux695yincmJA_i5bVEibEs7z2eo/pub?start=false&loop=true&delayms=3000>`_.

Workflow
=========

The generation of the model is controlled by the workflow management system
`Snakemake <https://snakemake.bitbucket.io/>`_.
In a nutshell, the ``Snakefile`` declares for each python script in the ``scripts`` directory a rule which describes which files the scripts consume and produce (their corresponding input and output files).
The ``snakemake`` tool then runs the scripts in the correct order according to the rules' input/output dependencies.
Moreover, it is able to track, what parts of the workflow have to be regenerated, when a data file or a script is modified/updated.

For instance an invocation to

.. code:: bash

    .../pypsa-eur % snakemake -j 1 networks/elec_s_128.nc

follows this dependency graph:

.. image:: img/workflow.png

The **blocks** represent the individual rules which are required to create the file ``networks/elec_s_128.nc``. The **arrows** indicate the outputs from preceding rules which a particular rule takes as input data.

.. note::
    The dependency graph shown above was generated using
    ``snakemake --dag networks/elec_s_128.nc | dot -Tpng > workflow.png``

For the use of ``snakemake``, it makes sense to familiarize oneself quickly with its `basic tutorial <https://snakemake.readthedocs.io/en/stable/tutorial/basics.html>`_ and then read carefully through the section `Executing Snakemake <https://snakemake.readthedocs.io/en/stable/executable.html>`_, noting the arguments ``-j``, ``-n``, ``-r``, but also ``--dag``, ``-R`` and ``-t``.

Scenarios, Configuration and Modification
=========================================

It is easy to run PyPSA-Eur for multiple scenarios using the `wildcards feature <https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#wildcards>`_ of ``snakemake``. Wildcards allow to generalise a rule to produce all files that follow a `regular expression <https://en.wikipedia.org/wiki/Regular_expression>`_ pattern, which e.g. defines one particular scenario. One can think of a wildcard as a parameter that shows up in the input/output file names and thereby determines which rules to run, what data to retrieve and what files to produce. **Details are explained in** :ref:`wildcards` **and** :ref:`scenario`.

The model also has several further configuration options collected in the ``config.yaml`` file
located in the root directory, which that are not part of the scenarios. **All options are explained in detail in** :ref:`config`.

Folder Structure
================

- ``data``: Includes input data that is not produced by any ``snakemake`` rule.
- ``scripts``: Includes all the Python scripts executed by the ``snakemake`` rules.
- ``resources``: Stores intermediate results of the workflow which can be picked up again by subsequent rules.
- ``networks``: Stores intermediate, unsolved stages of the PyPSA network that describes the energy system model.
- ``results``: Stores the solved PyPSA network data, summary files and plots.
- ``benchmarks``: Stores ``snakemake`` benchmarks.
- ``logs``: Stores log files about solving, including the solver output, console output and the output of a memory logger.

System Requirements
===================

Building the model with the scripts in this repository runs on a normal computer.
But computing optimal investment and operation scenarios requires a strong interior-point solver
like `Gurobi <http://www.gurobi.com/>`_ or `CPLEX <https://www.ibm.com/analytics/cplex-optimizer>`_ with more memory.
