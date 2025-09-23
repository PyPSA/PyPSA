# Installation

!!! hint

    If this is your first time with Python, check out the [First Time Users¶](first-time-users.md) guide.

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

PyPSA is written and tested to be compatible with Python 3.10 and above. We recommend to use the latest version with active support (see [endoflife.date](https://endoflife.date/python)).

## Getting a solver

PyPSA passes optimisation problems (see [overview](optimal-power-flow/#overview)) to an external solver and is deeply integrated with [linopy](https://github.com/PyPSA/linopy) to do so.

Free software:

- [HiGHS](https://highs.dev/)
- [Cbc](https://projects.coin-or.org/Cbc#DownloadandInstall)
- [GLPK](https://www.gnu.org/software/glpk/) ([WinGLKP](http://winglpk.sourceforge.net/))
- [SCIP](https://scip.zib.de/)

Commercial software (for some of which free academic licenses are available):

- [Gurobi](https://www.gurobi.com/documentation/quickstart.html)
- [CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio)
- [FICO Xpress](https://www.fico.com/en/products/fico-xpress-optimization)
- [MOSEK](https://www.mosek.com/)
- [COPT](https://www.shanshu.ai/copt)
- [MindOpt](https://solver.damo.alibaba.com/doc/en/html/index.html)

An installation of PyPSA will automatically install the default solver HiGHS. For installation instructions of further solvers for your operating system, follow the links above.

!!! note

    Commercial solvers such as Gurobi, CPLEX, and Xpress currently significantly outperform open-source solvers for large-scale problems.
    It might be the case that you can only retrieve solutions by using a commercial solver.


## Upgrading PyPSA

We recommend always keeping your PyPSA installation up-to-date, since bugs get
fixed and new features are added. PyPSA is also only tested with the latest
stable versions of all the dependent packages for the respective Python
versions.

To upgrade PyPSA, run:

=== "pip"

    ``` bash
    pip install -U pypsa
    ```

=== "conda/mamba"

    ``` bash
    conda update pypsa
    ```

=== "uv"

    todo

Check the [release notes](https://pypsa.readthedocs.io/en/latest/release-notes.html) for API changes that may require you to update your code. PyPSA releases new versions according to the [semantic versioning](https://semver.org/) scheme. Any breaking changes are always announced via deprecation warnings in the code and in the release notes. If you are coming from a very old version (< `v1.0.0`), it makes sense to update gradually and fix any deprecation warnings before updating to the latest version.

## Dependencies

PyPSA relies heavily on other open-source Python packages. Some of them are:

* [pandas](http://pandas.pydata.org/) for storing data about components and time series
* [numpy](http://www.numpy.org/) and [scipy](http://scipy.org/) for calculations, such as linear algebra and sparse matrix calculations
* [linopy](https://github.com/PyPSA/linopy) for preparing optimisation problems (currently only linear and mixed-integer linear)
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
