<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Installation

!!! hint

    If it is your first time using Python, we recommend [conda](https://docs.conda.io/en/latest/miniconda.html), [mamba](https://github.com/mamba-org/mamba), [pip](https://pip.pypa.io/en/stable/) or [uv](https://docs.astral.sh/uv/) as easy-to-use package managers. They are available for Windows, macOS, and GNU/Linux. It is always helpful to use dedicated environments.

You can install PyPSA via all common package managers:

=== "pip"

    ``` bash
    pip install pypsa
    ```

=== "conda/mamba"

    ``` bash
    conda install -c conda-forge pypsa
    ```

=== "uv"

    ``` bash
    uv add pypsa
    ```

PyPSA is written and tested to be compatible with Python 3.11 and above. We recommend to use the latest version with active support (see [endoflife.date](https://endoflife.date/python)).

## Solvers

PyPSA passes optimisation problems (see [overview](overview.md)) to an external solver and is deeply integrated with the optimisation framework [linopy](https://github.com/PyPSA/linopy) to do so. Some examples of available solvers:

| Free & open source | Commercial & proprietary |
| ------------- | ------------------- |
| [HiGHS](https://highs.dev/) | [Gurobi](https://www.gurobi.com/documentation/quickstart.html) |
| [Cbc](https://projects.coin-or.org/Cbc#DownloadandInstall) | [CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio) |
| [GLPK](https://www.gnu.org/software/glpk/) | [FICO Xpress](https://www.fico.com/en/products/fico-xpress-optimization) |
| [SCIP](https://scip.zib.de/) | [MOSEK](https://www.mosek.com/) |
| | [COPT](https://www.shanshu.ai/copt) |

PyPSA ships with the open-source solver HiGHS by default. For installation instructions of further solvers for your operating system, follow the links above.

!!! note

    Commercial solvers currently significantly outperform open-source solvers for large-scale problems.
    It might be the case that you can only retrieve solutions by using a commercial solver.
    Many commercial solvers provide free academic licenses.


## Upgrading

We recommend always keeping your PyPSA installation up-to-date, since bugs get
fixed and new features are added. PyPSA is also only tested with the latest
stable versions of all the dependent packages for the respective Python
versions.

To upgrade PyPSA, run:

=== "pip"

    ``` bash
    pip install --upgrade pypsa
    # Or upgrade to a specific version:
    pip install pypsa==0.35.2
    ```

=== "conda/mamba"

    ``` bash
    conda update pypsa
    # Or upgrade to a specific version:
    conda install -c conda-forge pypsa==0.35.2
    ```

=== "uv"
    
    ``` bash
    uv add --upgrade pypsa 
    # Or upgrade to a specific version:
    uv add pypsa==0.35.2
    ```

Check the [release notes](../release-notes.md) for API changes that may require you to update your code. PyPSA releases new versions according to the [semantic versioning](https://semver.org/) scheme. Any breaking changes are always announced via deprecation warnings in the code and in the release notes, including a version when they are going to be removed (always the next major version, e.g. `v2.0.0`). That way you can be sure that your code will continue to work at least until the next major version. But this does not include bug fixes, which you only get when upgrading to the latest version.
If you are upgrading from a pre <!-- md:version v1.0.0 --->, we recommend you upgrade in small steps and fix any deprecation warnings before upgrading to `v1.0.0`.

## Dependencies

PyPSA relies heavily on other open-source Python packages. Some of them are:

* [pandas](http://pandas.pydata.org/) for storing data about components and time series
* [numpy](http://www.numpy.org/) and [scipy](http://scipy.org/) for calculations, such as linear algebra and sparse matrix calculations
* [linopy](https://github.com/PyPSA/linopy) for preparing optimisation problems (LP, QP, MILP)
* [matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/) and [plotly](https://plotly.com/python/) for static and interactive plotting
* [networkx](https://networkx.github.io/) for some network calculations
* [pytest](http://pytest.org/) for unit testing

Find the full list of dependencies in the [`pyproject.toml`](https://github.com/PyPSA/PyPSA/blob/master/pyproject.toml) file.

### Optional dependencies

Besides the mandatory dependencies that are installed by default, PyPSA has a number of optional dependencies that are not installed by default and are only needed for certain features. You can install them by executing the command:

=== "pip"

    ``` bash
    pip install "pypsa[<feature>]"
    ```

=== "conda/mamba"

    ``` bash
    conda install -c conda-forge "pypsa[<feature>]"
    ```

=== "uv"

    ``` bash
    uv add "pypsa[<feature>]"
    ```

where `<feature>` can be one of the following:

**IO**

- `pypsa[hdf5]`: for reading and writing HDF5 files
- `pypsa[excel]`: for reading and writing Excel files
- `pypsa[cloudpath]`: for reading and writing files from cloud storage

**Plots**

- `pypsa[cartopy]`: for plotting geographical maps

**Solvers**

- `pypsa[gurobipy]`: for installing the Gurobi Python API

**Development**

- `pypsa[dev]`: for installing all development dependencies, including linopy and pytest
- `pypsa[docs]`: for installing all dependencies needed to build the documentation
