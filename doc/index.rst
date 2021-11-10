..
  SPDX-FileCopyrightText: 2019-2020 The PyPSA-Eur Authors

  SPDX-License-Identifier: CC-BY-4.0

PyPSA-Eur: An Open Optimisation Model of the European Transmission System
=========================================================================

.. image:: https://img.shields.io/github/v/release/pypsa/pypsa-eur?include_prereleases
    :alt: GitHub release (latest by date including pre-releases)

.. image:: https://github.com/pypsa/pypsa-eur/actions/workflows/ci.yaml/badge.svg
    :target: https://github.com/PyPSA/pypsa-eur/actions

.. image:: https://readthedocs.org/projects/pypsa-eur/badge/?version=latest
    :target: https://pypsa-eur.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/github/repo-size/pypsa/pypsa-eur
    :alt: GitHub repo size

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3520874.svg
    :target: https://doi.org/10.5281/zenodo.3520874

.. image:: https://badges.gitter.im/PyPSA/community.svg
    :target: https://gitter.im/PyPSA/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
    :alt: Chat on Gitter

.. image:: https://img.shields.io/badge/snakemake-≥5.0.0-brightgreen.svg?style=flat
    :target: https://snakemake.readthedocs.io
    :alt: Snakemake

.. image:: https://api.reuse.software/badge/github.com/pypsa/pypsa-eur
    :target: https://api.reuse.software/info/github.com/pypsa/pypsa-eur
    :alt: REUSE status

PyPSA-Eur is an open model dataset of the European power system at the
transmission network level that covers the full ENTSO-E area.

It contains alternating current lines at and above 220 kV voltage level and all high voltage direct current lines, substations, an open database of conventional power plants, time series for electrical demand and variable renewable generator availability, and geographic potentials for the expansion of wind and solar power.

The model is suitable both for operational studies and generation and transmission expansion planning studies. The continental scope and highly resolved spatial scale enables a proper description of the long-range smoothing effects for renewable power generation and their varying resource availability.

.. image:: img/base.png
    :width: 50%
    :align: center

The restriction to freely available and open data encourages the open exchange of model data developments and eases the comparison of model results. It provides a full, automated software pipeline to assemble the load-flow-ready model from the original datasets, which enables easy replacement and improvement of the individual parts.

PyPSA-Eur is designed to be imported into the open toolbox `PyPSA <https://www.pypsa.org>`_ for which `documentation <https://pypsa.org/doc>`_ is available as well.

This project is currently maintained by the `Department of Digital
Transformation in Energy Systems <https:/www.ensys.tu-berlin.de>`_ at the
`Technische Universität Berlin <https://www.tu.berlin>`_. Previous versions were
developed within the `IAI <http://www.iai.kit.edu>`_ at the `Karlsruhe Institute of
Technology (KIT) <http://www.kit.edu/english/index.php>`_ and by the `Renewable
Energy Group
<https://fias.uni-frankfurt.de/physics/schramm/renewable-energy-system-and-network-analysis/>`_
at `FIAS <https://fias.uni-frankfurt.de/>`_ to carry out simulations for the
`CoNDyNet project <http://condynet.de/>`_, financed by the `German Federal
Ministry for Education and Research (BMBF) <https://www.bmbf.de/en/index.html>`_
as part of the `Stromnetze Research Initiative
<http://forschung-stromnetze.info/projekte/grundlagen-und-konzepte-fuer-effiziente-dezentrale-stromnetze/>`_.

A version of the model that adds building heating, transport and industry sectors to the model,
as well as gas networks, is currently being developed in the `PyPSA-Eur-Sec repository <https://github.com/pypsa/pypsa-eur-sec>`_.

Documentation
=============

**Getting Started**

* :doc:`introduction`
* :doc:`installation`
* :doc:`tutorial`

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting Started

   introduction
   installation
   tutorial

**Configuration**

* :doc:`wildcards`
* :doc:`configuration`
* :doc:`costs`

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Configuration

   wildcards
   configuration
   costs

**Rules Overview**

* :doc:`preparation`
* :doc:`simplification`
* :doc:`solving`
* :doc:`plotting`

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Rules Overview

   preparation
   simplification
   solving
   plotting

**References**

* :doc:`release_notes`
* :doc:`limitations`
* :doc:`contributing`
* :doc:`cloudcomputing`

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: References

   release_notes
   limitations
   contributing
   cloudcomputing

Warnings
========

Please read the `limitations <https://pypsa-eur.readthedocs.io/en/latest/limitations.html>`_ section of the
documentation and paper carefully before using the model. We do not
recommend to use the full resolution network model for simulations. At
high granularity the assignment of loads and generators to the nearest
network node may not be a correct assumption, depending on the topology of the underlying distribution grid,
and local grid
bottlenecks may cause unrealistic load-shedding or generator
curtailment. We recommend to cluster the network to a couple of
hundred nodes to remove these local inconsistencies.

Learning Energy System Modelling
================================

If you are (relatively) new to energy system modelling and optimisation
and plan to use PyPSA-Eur, the following resources are *one way* to get started
in addition to reading this documentation.

- Documentation of `PyPSA <https://pypsa.readthedocs.io>`__, the package for
  simulating and optimising modern power systems which PyPSA-Eur uses under the hood.
- Course on `Energy System Modelling <https://nworbmot.org/courses/esm-2019/>`_,
  Karlsruhe Institute of Technology (KIT), `Dr. Tom Brown <https://nworbmot.org>`_

Citing PyPSA-Eur
================

If you use PyPSA-Eur for your research, we would appreciate it if you would cite the following paper:

- Jonas Hörsch, Fabian Hofmann, David Schlachtberger, and Tom Brown. `PyPSA-Eur: An open optimisation model of the European transmission system <https://arxiv.org/abs/1806.01613>`_. Energy Strategy Reviews, 22:207-215, 2018. `arXiv:1806.01613 <https://arxiv.org/abs/1806.01613>`_, `doi:10.1016/j.esr.2018.08.012 <https://doi.org/10.1016/j.esr.2018.08.012>`_.

Please use the following BibTeX: ::

    @article{PyPSAEur,
        author = "Jonas Hoersch and Fabian Hofmann and David Schlachtberger and Tom Brown",
        title = "PyPSA-Eur: An open optimisation model of the European transmission system",
        journal = "Energy Strategy Reviews",
        volume = "22",
        pages = "207 - 215",
        year = "2018",
        issn = "2211-467X",
        doi = "10.1016/j.esr.2018.08.012",
        eprint = "1806.01613"
    }


If you want to cite a specific PyPSA-Eur version, each release of PyPSA-Eur is stored on Zenodo with a release-specific DOI.
This can be found linked from the overall PyPSA-Eur Zenodo DOI:

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3520874.svg
   :target: https://doi.org/10.5281/zenodo.3520874

Pre-Built Networks as a Dataset
===============================

There are pre-built networks available as a dataset on Zenodo as well for every release of PyPSA-Eur.

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3601881.svg
   :target: https://doi.org/10.5281/zenodo.3601881

The included ``.nc`` files are PyPSA network files which can be imported with PyPSA via:

.. code:: python

    import pypsa

    filename = "elec_s_1024_ec.nc" # example
    n = pypsa.Network(filename)

Licence
=======

PyPSA-Eur work is released under multiple licenses:

* All original source code is licensed as free software under `MIT <LICENSES/MIT.txt>`_.
* The documentation is licensed under `CC-BY-4.0 <LICENSES/CC-BY-4.0.txt>`_.
* Configuration files are mostly licensed under `CC0-1.0 <LICENSES/CC0-1.0.txt>`_.
* Data files are licensed under `CC-BY-4.0 <LICENSES/CC-BY-4.0.txt>`_.

See the individual files and the `dep5 <.reuse/dep5>`_ file for license details.

Additionally, different licenses and terms of use also apply to the various input data, which are summarised below.
More details are included in
`the description of the data bundles on zenodo <https://zenodo.org/record/3517935#.XbGeXvzRZGo>`_.

.. csv-table::
   :header-rows: 1
   :file: configtables/licenses.csv

* *BY: Attribute Source*
* *NC: Non-Commercial Use Only*
* *SA: Share Alike*
