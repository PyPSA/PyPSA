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

.. image:: https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FPyPSA%2FPyPSA%2Fmaster%2Fpyproject.toml
    :alt: Python Version from PEP 621 TOML

.. image:: https://github.com/PyPSA/PyPSA/actions/workflows/test.yml/badge.svg
    :target: https://github.com/PyPSA/PyPSA/actions/workflows/test.yml
    :alt: Tests

.. image:: https://readthedocs.org/projects/pypsa/badge/?version=latest
    :target: https://pypsa.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://results.pre-commit.ci/badge/github/PyPSA/PyPSA/master.svg
    :target: https://results.pre-commit.ci/latest/github/PyPSA/PyPSA/master
    :alt: pre-commit.ci status

.. image:: https://codecov.io/gh/PyPSA/PyPSA/branch/master/graph/badge.svg?token=kCpwJiV6Jr
    :target: https://codecov.io/gh/PyPSA/PyPSA
    :alt: Code coverage

.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff
    :alt: Ruff

.. image:: https://img.shields.io/pypi/l/pypsa.svg
    :target: LICENSE.txt
    :alt: License

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3946412.svg
    :target: https://doi.org/10.5281/zenodo.3946412
    :alt: Zenodo

.. image:: https://img.shields.io/discord/911692131440148490?logo=discord
    :target: https://discord.gg/AnuJBk23FU
    :alt: Discord

.. image:: https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg
    :target: CODE_OF_CONDUCT.md
    :alt: Contributor Covenant


| PyPSA stands for "Python for Power System Analysis". It is pronounced "pipes-ah".

PyPSA is an open source toolbox for simulating and optimising modern power and
energy systems that include features such as conventional generators with unit
commitment, variable wind and solar generation, storage units, coupling to other
energy sectors, and mixed alternating and direct current networks. PyPSA is
designed to scale well with large networks and long time series.

This project is maintained by the `Department of Digital Transformation in
Energy Systems <https://tub-ensys.github.io>`_ at the `Technical University of
Berlin <https://www.tu.berlin>`_. Previous versions were developed by the Energy
System Modelling group at the `Institute for Automation and Applied
Informatics <https://www.iai.kit.edu/english/index.php>`_ at the `Karlsruhe
Institute of Technology <http://www.kit.edu/english/index.php>`_ funded by the
`Helmholtz Association <https://www.helmholtz.de/en/>`_, and by the `Renewable
Energy
Group <https://fias.uni-frankfurt.de/physics/schramm/renewable-energy-system-and-network-analysis/>`_
at `FIAS <https://fias.uni-frankfurt.de/>`_ to carry out simulations for the
`CoNDyNet project <https://fias.institute/en/projects/condynet/>`_, financed by the `German Federal
Ministry for Education and Research (BMBF) <https://www.bmbf.de/bmbf/en/>`_
as part of the `Stromnetze Research
Initiative <http://forschung-stromnetze.info/projekte/grundlagen-und-konzepte-fuer-effiziente-dezentrale-stromnetze/>`_.


Documentation
=============

**Getting Started**

* :doc:`/getting-started/introduction`
* :doc:`/getting-started/installation`
* :doc:`/getting-started/quick-start`

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting Started

   getting-started/introduction
   getting-started/installation
   getting-started/quick-start

**User Guide**

* :doc:`/user-guide/design`
* :doc:`/user-guide/components`
* :doc:`/user-guide/import-export`
* :doc:`/user-guide/power-flow`
* :doc:`/user-guide/optimal-power-flow`
* :doc:`/user-guide/contingency-analysis`
* :doc:`/user-guide/statistics`
* :doc:`/user-guide/plotting`

.. toctree::
   :hidden:
   :maxdepth: 0
   :caption: User Guide

   user-guide/design
   user-guide/components
   user-guide/import-export
   user-guide/power-flow
   user-guide/optimal-power-flow
   user-guide/contingency-analysis
   user-guide/statistics
   user-guide/plotting

**Examples**

* :doc:`/examples-index/lopf`
* :doc:`/examples-index/sector-coupling`
* :doc:`/examples-index/other`
* :doc:`/examples-index/models`

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Examples

   examples-index/lopf
   examples-index/sector-coupling
   examples-index/other
   examples-index/models

**Contributing & Support**

* :doc:`/contributing/contributing`
* :doc:`/contributing/support`
* :doc:`/contributing/troubleshooting`

.. toctree::
   :hidden:
   :maxdepth: 0
   :caption: Contributing & Support

   contributing/contributing
   contributing/support
   contributing/troubleshooting


**References**

* :doc:`/references/api-reference`
* :doc:`/references/release-notes`
* :doc:`/references/citing`
* :doc:`/references/users`
* :doc:`/references/developers`

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: References

   references/api-reference
   references/release-notes
   references/citing
   references/users
   references/developers
