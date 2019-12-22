##########################################
Release Notes
##########################################

PyPSA-Eur 0.1.0 (DATE)
======================

This is the first release of PyPSA-Eur:

* Documentation on installation, workflows and configuration settings is now available online at `pypsa-eur.readthedocs.io <pypsa-eur.readthedocs.io>`_ (`#65 <https://github.com/PyPSA/pypsa-eur/pull/65>`_).

* The ``conda`` environment files were updated and extended (`#81 <https://github.com/PyPSA/pypsa-eur/pull/81>`_).

* The power plant database was updated with extensive filtering options via ``pandas.query`` functionality (`#84 <https://github.com/PyPSA/pypsa-eur/pull/84>`_ and `#94 <https://github.com/PyPSA/pypsa-eur/pull/94>`_).

* Continuous integration testing with `Travis CI <https://travis-ci.org>`_ is now included for Linux, Mac and Windows (`#82 <https://github.com/PyPSA/pypsa-eur/pull/82>`_).

* Data dependencies were moved to `zenodo <https://zenodo.org/>`_ and are now versioned (`#60 <https://github.com/PyPSA/pypsa-eur/issues/60>`_).

* Data dependencies are now retrieved directly from within the snakemake workflow (`#86 <https://github.com/PyPSA/pypsa-eur/pull/86>`_).

* Emission prices can be added to marginal costs of generators through the keyworks ``Ep`` in the ``{opts}`` wildcard (`#100 <https://github.com/PyPSA/pypsa-eur/pull/100>`_).

* An option is introduced to add extendable nuclear power plants to the network (`#98 <https://github.com/PyPSA/pypsa-eur/pull/98>`_).

* Focus weights can now be specified for particular countries for the network clustering, which allows to set a proportion of the total number of clusters for particular countries (`#87 <https://github.com/PyPSA/pypsa-eur/pull/87>`_).

* A new rule :mod:`add_extra_components` allows to add additional components to the network only after clustering. It is thereby possible to model storage units (e.g. battery and hydrogen) in more detail via a combination of ``Store``, ``Link`` and ``Bus`` elements (`#97 <https://github.com/PyPSA/pypsa-eur/pull/97>`_).

* Hydrogen pipelines (including cost assumptions) can now be added alongside clustered network connections in the rule :mod:`add_extra_components` . Set ``electricity: extendable_carriers: Link: [H2 pipeline]`` and ensure hydrogen storage is modelled as a ``Store``. This is a first simplified stage (`#108 <https://github.com/PyPSA/pypsa-eur/pull/108>`_).

* Logfiles for all rules of the ``snakemake`` workflow are now written in the folder ``log/`` (`#102 <https://github.com/PyPSA/pypsa-eur/pull/102>`_). 

* The new function ``_helpers.mock_snakemake`` creates a ``snakemake`` object which mimics the actual ``snakemake`` object produced by workflow by parsing the ``Snakefile`` and setting all paths for inputs, outputs, and logs. This allows running all scripts within a (I)python terminal (or just by calling ``python <script-name>``) and thereby facilitates developing and debugging scripts significantly (`#107 <https://github.com/PyPSA/pypsa-eur/pull/107>`_).
