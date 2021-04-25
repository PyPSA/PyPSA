.. PyPSA documentation master file, created by
   sphinx-quickstart on Tue Jan  5 10:04:42 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyPSA: Python for Power System Analysis
=======================================

.. image:: https://img.shields.io/pypi/v/pypsa.svg
    :target: https://pypi.python.org/pypi/pypsa
    :alt: PyPI version

.. image:: https://img.shields.io/conda/vn/conda-forge/pypsa.svg
    :target: https://anaconda.org/conda-forge/pypsa
    :alt: Conda version

.. image:: https://img.shields.io/travis/PyPSA/PyPSA/master.svg
    :target: https://travis-ci.org/PyPSA/PyPSA
    :alt: Build status on Linux

.. image:: https://readthedocs.org/projects/pypsa/badge/?version=latest
    :target: https://pypsa.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/pypi/l/pypsa.svg
    :target: License

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3946412.svg
   :target: https://doi.org/10.5281/zenodo.3946412

.. image:: https://badges.gitter.im/PyPSA/community.svg
    :target: https://gitter.im/PyPSA/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
    :alt: Chat on Gitter

PyPSA stands for "Python for Power System Analysis". It is pronounced "pipes-ah".

PyPSA is a `free software
<http://www.gnu.org/philosophy/free-sw.en.html>`_ toolbox for
simulating and optimising modern power systems that include features
such as conventional generators with unit commitment, variable wind
and solar generation, storage units, coupling to other energy sectors,
and mixed alternating and direct current networks.  PyPSA is designed
to scale well with large networks and long time series.

This project is maintained by the `Energy System Modelling
group <https://www.iai.kit.edu/english/2338.php>`_ at the `Institute for
Automation and Applied
Informatics <https://www.iai.kit.edu/english/index.php>`_ at the
`Karlsruhe Institute of
Technology <http://www.kit.edu/english/index.php>`_. The group is funded by the
`Helmholtz Association <https://www.helmholtz.de/en/>`_ until 2024.
Previous versions were developed by the `Renewable Energy Group
<https://fias.uni-frankfurt.de/physics/schramm/renewable-energy-system-and-network-analysis/>`_
at `FIAS <https://fias.uni-frankfurt.de/>`_ to carry out simulations
for the `CoNDyNet project <http://condynet.de/>`_, financed by the
`German Federal Ministry for Education and Research (BMBF) <https://www.bmbf.de/en/index.html>`_ as part of the `Stromnetze Research Initiative <http://forschung-stromnetze.info/projekte/grundlagen-und-konzepte-fuer-effiziente-dezentrale-stromnetze/>`_.


Documentation
=============

**Getting Started**

* :doc:`introduction`
* :doc:`installation`
* :doc:`quick_start`
* :doc:`examples`

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting Started

   introduction
   installation
   quick_start
   examples

**User Guide**

* :doc:`design`
* :doc:`components`
* :doc:`import_export`
* :doc:`power_flow`
* :doc:`optimal_power_flow`
* :doc:`contingency_analysis`
* :doc:`plotting`

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: User Guide

   design
   components
   import_export
   power_flow
   optimal_power_flow
   contingency_analysis
   plotting


**Help & References**

* :doc:`release_notes`
* :doc:`api_reference`
* :doc:`troubleshooting`
* :doc:`comparable_software`
* :doc:`contributing`
* :doc:`citing`
* :doc:`unit_testing`
* :doc:`mailing_list`
* :doc:`users`

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Help & References

   release_notes
   api_reference
   troubleshooting
   comparable_software
   contributing
   citing
   unit_testing
   mailing_list
   users
