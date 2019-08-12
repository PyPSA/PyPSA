PyPSA-Eur: An Open Optimisation Model of the European Transmission System
=========================================================================

.. image:: https://img.shields.io/github/tag-date/pypsa/pypsa-eur
    :alt: GitHub tag

.. image:: https://readthedocs.org/projects/pypsa-eur/badge/?version=latest
    :target: https://pypsa-eur.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/github/license/pypsa/pypsa-eur
    :alt: GitHub

.. image:: https://img.shields.io/github/repo-size/pypsa/pypsa-eur
    :alt: GitHub repo size

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1246852.svg
    :target: https://doi.org/10.5281/zenodo.1246852

.. image:: https://badges.gitter.im/PyPSA/community.svg
    :target: https://gitter.im/PyPSA/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
    :alt: Chat on Gitter

PyPSA-Eur is an open model dataset of the European power system at the
transmission network level that covers the full ENTSO-E area.

It contains alternating current lines at and above 220 kV voltage level and all high voltage direct current lines, substations, an open database of conventional power plants, time series for electrical demand and variable renewable generator availability, and geographic potentials for the expansion of wind and solar power.

The model is suitable both for operational studies and generation and transmission expansion planning studies. The continental scope and highly resolved spatial scale enables a proper description of the long-range smoothing effects for renewable power generation and their varying resource availability.

.. image:: img/base.png

The restriction to freely available and open data encourages the open exchange of model data developments and eases the comparison of model results. It provides a full, automated software pipeline to assemble the load-flow-ready model from the original datasets, which enables easy replacement and improvement of the individual parts.

PyPSA-Eur is designed to be imported into the open toolbox `PyPSA <https://www.pypsa.org>`_ for which `documentation <https://pypsa.org/doc>`_ is available as well.

This project is maintained by the `Energy System Modelling group <https://www.iai.kit.edu/english/2338.php>`_ at the `Institute for Automation and Applied Informatics <https://www.iai.kit.edu/english/index.php>`_ at the `Karlsruhe Institute of Technology <http://www.kit.edu/english/index.php>`_. The group is funded by the `Helmholtz Association <https://www.helmholtz.de/en/>`_ until 2024. Previous versions were developed by the `Renewable Energy Group <https://fias.uni-frankfurt.de/physics/schramm/renewable-energy-system-and-network-analysis/>`_ at `FIAS <https://fias.uni-frankfurt.de/>`_ to carry out simulations for the `CoNDyNet project <http://condynet.de/>`_, financed by the `German Federal Ministry for Education and Research (BMBF) <https://www.bmbf.de/en/index.html>`_ as part of the `Stromnetze Research Initiative <http://forschung-stromnetze.info/projekte/grundlagen-und-konzepte-fuer-effiziente-dezentrale-stromnetze/>`_.

Documentation
=============

**Getting Started**

* :doc:`introduction`
* :doc:`installation`
* :doc:`examples`

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting Started

   introduction
   installation
   examples

**Configuration**

* :doc:`wildcards`
* :doc:`configuration`

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Configuration

   wildcards
   configuration

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
* :doc:`contributing`

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: References

   release_notes
   contributing


Citing PyPSA-Eur
================

If you use PyPSA-Eur for your research, we would appreciate it if you would cite the following paper :cite:`PyPSAEur`:

.. bibliography:: references.bib
    :cited:

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


.. todo:: Add versions to zenodo

If you want to cite a specific PyPSA-Eur version, each release of PyPSA-Eur is stored on Zenodo with a release-specific DOI.
This can be found linked from the overall PyPSA-Eur Zenodo DOI:

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1246852.svg
   :target: https://doi.org/10.5281/zenodo.1246852

Publications using/citing PyPSA-Eur
===================================

.. bibliography:: references.bib

Licence
=======

The code in PyPSA-Eur is released as free software under the `GPLv3
<http://www.gnu.org/licenses/gpl-3.0.en.html>`_, see
`LICENSE <https://github.com/PyPSA/pypsa-eur/blob/master/LICENSE.txt>`_.
