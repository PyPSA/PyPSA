#######################
Developing
#######################

Contributing
============

We strongly welcome anyone interested in contributing to this project.

Submit pull requests / issues to the `github repository <https://github.com/PyPSA/PyPSA>`_.

Join the `mailing list
<https://groups.google.com/group/pypsa>`_.


Release Process
===============


* Update ``release_notes.rst``
* Update version in ``setup.py``, ``doc/conf.py``, ``pypsa/__init__.py``
* ``git commit`` and put release notes in commit message
* ``git tag v0.x.0``
* ``git push`` and  ``git push --tags``
* To upload to `PyPI <https://pypi.org/>`_, run ``python setup.py
  sdist``, then ``twine check dist/pypsa-0.x.0.tar.gz`` and ``twine
  upload dist/pypsa-0.x.0.tar.gz``
* To update to conda-forge, check the pull request generated at the `feedstock repository
  <https://github.com/conda-forge/pypsa-feedstock>`_.
* Making a `GitHub release <https://github.com/PyPSA/PyPSA/releases>`_
  will trigger `zenodo <https://zenodo.org/>`_ to archive the release
  with its own DOI.
* Inform the PyPSA mailing list.

Unit Testing
============

Unit testing is performed with ``py.test``.

Tests can be found in ``pypsa/test/``.

Power flow is tested against Pypower (the Python implementation of MATPOWER) using its built-in cases.

Unit testing of new GitHub commits is automated with `Travis CI <https://travis-ci.org/PyPSA/PyPSA>`_.


