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

.. image:: https://readthedocs.org/projects/pypsa-readthedocs/badge/?version=readthedocs
    :target: https://pypsa-readthedocs.readthedocs.io/en/readthedocs/?badge=readthedocs
    :alt: Documentation Status

.. image:: https://img.shields.io/pypi/l/pypsa.svg
    :target: License

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.786605.svg
   :target: https://doi.org/10.5281/zenodo.786605

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
* :doc:`unit_testing`

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Help & References

   release_notes
   api_reference
   troubleshooting
   comparable_software
   contributing
   unit_testing


Target user group
=================

PyPSA is intended for researchers, planners and utilities who need a
fast, easy-to-use and transparent tool for power system
analysis. PyPSA is free software and can be arbitrarily extended.


Mailing list
============

PyPSA has a Google Group `forum / mailing list
<https://groups.google.com/group/pypsa>`_.

Anyone can join and anyone can read the posts; only members of the
group can post to the list.

The intention is to have a place where announcements of new releases
can be made and questions can be asked.

To discuss issues and suggest/contribute features
for future development we prefer ticketing through the `PyPSA Github Issues page
<https://github.com/PyPSA/PyPSA/issues>`_.


Citing PyPSA
============

If you use PyPSA for your research, we would appreciate it if you
would cite the following paper:

* T. Brown, J. Hörsch, D. Schlachtberger, `PyPSA: Python for Power
  System Analysis <https://arxiv.org/abs/1707.09913>`_, 2018,
  `Journal of Open Research Software
  <https://openresearchsoftware.metajnl.com/>`_, 6(1),
  `arXiv:1707.09913 <https://arxiv.org/abs/1707.09913>`_,
  `DOI:10.5334/jors.188 <https://doi.org/10.5334/jors.188>`_

Please use the following BibTeX: ::

   @article{PyPSA,
      author = {T. Brown and J. H\"orsch and D. Schlachtberger},
      title = {{PyPSA: Python for Power System Analysis}},
      journal = {Journal of Open Research Software},
      volume = {6},
      issue = {1},
      number = {4},
      year = {2018},
      eprint = {1707.09913},
      url = {https://doi.org/10.5334/jors.188},
      doi = {10.5334/jors.188}
   }


If you want to cite a specific PyPSA version, each release of PyPSA is
stored on `Zenodo <https://zenodo.org/>`_ with a release-specific DOI.
This can be found linked from the overall PyPSA Zenodo DOI:

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.786605.svg
   :target: https://doi.org/10.5281/zenodo.786605



Licence
=======

PyPSA is released under the `GPLv3
<http://www.gnu.org/licenses/gpl-3.0.en.html>`_.

