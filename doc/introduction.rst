.. _intro:

##########################################
 Introduction
##########################################

Workflow
=========

The generation of the model is controlled by the workflow management system
`Snakemake <https://snakemake.bitbucket.io/>`_.
In a nutshell, the ``Snakefile`` declares for each python script in the ``scripts`` directory a rule which describes which files the scripts consume and produce (their corresponding input and output files).
The ``snakemake`` tool then runs the scripts in the correct order according to the rules' input/output dependencies.
Moreover, it is able to track, what parts of the workflow have to be regenerated, when a data file or a script is modified/updated.

For instance an invocation to

.. code:: bash

    .../pypsa-eur % snakemake networks/elec_s_128.nc

follows this dependency graph:

.. image:: img/workflow.png

The **blocks** represent the individual rules which are required to create the file ``networks/elec_s_128.nc``. The **arrows** indicate the outputs from preceding rules which a particular rule takes as input data.

.. note::
    The dependency graph shown above was generated using
    ``snakemake --dag networks/elec_s_128.nc | dot -Tpng > workflow.png``

For the use of ``snakemake``, it makes sense to familiarize oneself quickly with its `basic tutorial <https://snakemake.readthedocs.io/en/stable/tutorial/basics.html>`_ and then read carefully through the section `Executing Snakemake <https://snakemake.readthedocs.io/en/stable/executable.html>`_, noting the arguments ``-n``, ``-r``, but also ``--dag``, ``-R`` and ``-t``.

Modification
============

.. todo:: wildcards modification

The model has several configuration options collected in the ``config.yaml`` file
located in the root directory.

Folder Structure
================

System Requirements
===================

Building the model with the scripts in this repository uses up to 20 GB of memory. Computing optimal investment and operation scenarios requires a strong interior-point solver compatible with the modelling library `Pyomo <https://www.pyomo.org>`_ like `Gurobi <http://www.gurobi.com/>`_ or `CPLEX <https://www.ibm.com/analytics/cplex-optimizer>`_ with up to 100 GB of memory.