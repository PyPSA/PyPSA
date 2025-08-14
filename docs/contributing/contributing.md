#######################
Contributing
#######################

First of all, thank you for thinking about contributing to PyPSA! 

We strongly welcome anyone interested in PyPSA and all its related projects, be it
with new ideas, suggestions, submitting bug reports or contributing code changes.

**How to contribute**

* :ref:`code-contributions`: Implement new features, fix bugs, or improve the performance.
* :ref:`documentation-contributions`: Improve the documentation by adding new sections, fixing typos, or clarifying existing content.
* :ref:`example-contributions`: Showcase your work, which could be useful for others.

**Where to go for help**

* To **discuss** with other PyPSA users, organise projects, share news, and get in touch with the community you can use the `Discord server <https://discord.gg/AnuJBk23FU>`_.
* For **troubleshooting**, please check the `troubleshooting <https://pypsa.readthedocs.io/en/latest/contributing/troubleshooting.html>`_ in the documentation.
* For **guidelines to contribute** to PyPSA, stay right here.


.. _code-contributions:

Code
=====


**Contribution workflow in a nutshell**

#. Fork the repository on GitHub
#. Clone your fork: ``git clone https://github.com/<your-username>/PyPSA.git``
#. Fetch the upstream tags ``git fetch --tags https://github.com/PyPSA/PyPSA.git``
#. Install with dependencies in editable mode: ``pip install -e .[dev]``
#. Setup linter and formatter, e.g ``pre-commit install`` (see :ref:`linting-and-formatting`)
#. Write your code (preferably on a new branch)
#. Run tests: ``pytest`` (see :ref:`testing`)
#. Push your changes to your fork and create a pull request on GitHub

.. TODO: What to work on, TODO, which issues, labeling etc. 

.. _linting-and-formatting:

Code Style
----------------------

**pre-commit**

We run a couple of tools via `pre-commit <https://pre-commit.com>`_ to ensure a 
consistent code style and to catch common programming errors or bad practices before
they are committed. Don't worry, you can just start coding and the pre-commit will 
tell you if something is not right.

It is already installed with the development dependencies, but you can also install it
manually via ``pip install pre-commit`` or ``conda install -c conda-forge pre-commit``.

To use it automatically before every commit (recommended), just run once:

.. code::

    pre-commit install

This will automatically check the changes which are staged before you commit them.

To manually run it, use:

.. code::

    pre-commit run --all

This will check all files in the repository.

**Ruff**

One of the tools that is run by pre-commit is `Ruff <https://docs.astral.sh/ruff>`_,
which is our linter and formatter. It combines common tools like Flake8, Black, etc. 
Besides pre-commit, you can also run it via your CLI (see `Ruff installation) <https://docs.astral.sh/ruff/installation/>`_) 
or IDE (e.g. VSCode `plugin <https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff>`_).
This will help you to keep your code clean and consistent already during development.

Ruff is also already installed with the development dependencies, but you can also install it
manually using ``pip install ruff``.

To use the linter in your CLI, run:

.. code::

    ruff check . --fix

This will check all files in the repository and give you hints on what to improve. The 
``--fix`` flag will also automatically fix some of the issues, if possible. Some 
issues need to be fixed manually.

And to run the formatter, use:

.. code::

    ruff format .

This will format all the files in the repository and immediately apply the changes to 
them. It is basically `the same <https://docs.astral.sh/ruff/faq/#how-does-ruffs-formatter-compare-to-black>`_
as Black. 

.. note::

 It is not mandatory to use either Ruff or pre-commit. We will also be running it in 
 our CI/CD pipeline. But it's highly recommended, to make everyone's life easier.


.. _testing:

Testing
-------

Unit testing is performed with pytest which is installed with the development dependencies.

The tests can be found in the :file:`test/` folder and can be run with:

.. code::

    pytest

Or to run individual tests:

.. code::

    pytest test_lpf_against_pypower.py

Power flow is tested against PYPOWER (the Python implementation of MATPOWER)
and pandapower.

.. warning::

    Note that PYPOWER 5.0 has a bug in the linear load flow, which was fixed in the github version in January 2016.

.. note::

    Note also that the test results against which everything is tested
    were generated with the free software LP solver GLPK; other solver may
    give other results (e.g. Gurobi can give a slightly better result).


Unit testing is also performed in the CI/CD pipeline, similar to the linting and formatting.


.. _documentation-contributions:

Documentation
==============

We strive to keep documentation useful and up to date for all PyPSA users. If you 
encounter an area where documentation is not available or insufficient, we very much 
welcome your contribution.

For bigger changes, we recommend to make them locally. Just follow the steps in 
:ref:`code-contributions` to set up your local environment. In addition you can:

#. Also install the documentation dependencies via ``pip install -e .[docs]``.
#. Make your changes in the corresponding .rst file under the :file:`doc` folder.
#. Compile your changes by running the following command in your terminal in the :file:`doc` folder: ``make html``
   
   * You may encounter some warnings, but end up with a message such as ``build succeeded, XX warnings.``. html files to review your changes can then be found under :file:`doc/_build/html`.

For simple changes, you can also edit the documentation directly on GitHub:

#. If you are on the documentation page, click on the little book icon on the bottom 
   left with :guilabel:`v: latest`, which indicates the version/ branch. :guilabel:`Edit`
   under "On GitHub" will bring you straight to the source file.
#. Make your changes in the file.
#. Commit your changes and create a pull request. 

.. note::

    If you are not familiar with reStructuredText, you can find a quick guide `here <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_.
    It is quite simple and you should be fine with just keeping the structure of 
    the existing files.

.. _example-contributions:

Examples
=========

Nice examples are always welcome.

You can even submit your Jupyter notebook (``.ipynb``) directly
as an example. Please run the linter (see :ref:`linting-and-formatting`) to ensure
that the notebook is clean and metadata is removed.

Then for every notebook:

#. Write the notebook (let's call it :file:`foo.ipynb`) and place it
   in :file:`examples/notebooks/foo.ipynb`.

#. Provide a link to the documentation:
   Include a file :file:`foo.nblink` located in :file:`doc/examples/`
   with the following content:

       {'path' : '../../examples/foo.ipynb'}
    
   Adjust the path for your file's name.
   This ``nblink`` allows us to link your notebook into the documentation.

#. Link your file in the documentation:

   * Include your :file:`examples/foo.nblink` directly into one of the documentations 
     toctrees
   * or tell us where in the documentation you want your example to show up

#. Commit your changes and create a pull request.

The support for the ``.ipynb`` notebook format in our documentation
is realised via the extensions `nbsphinx <https://nbsphinx.readthedocs.io/en/0.4.2/installation.html>`_ 
and `nbsphinx_link <https://nbsphinx.readthedocs.io/en/latest/>`_.
