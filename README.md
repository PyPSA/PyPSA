<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: MIT
-->

<picture align="center">
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/PyPSA/PyPSA/refs/heads/master/docs/assets/logo/logo-primary-dark.svg">
  <img alt="PyPSA Banner" src="https://raw.githubusercontent.com/PyPSA/PyPSA/refs/heads/master/docs/assets/logo/logo-primary-light.svg">
</picture>

# PyPSA - Python for Power System Analysis


[![PyPI version](https://img.shields.io/pypi/v/pypsa.svg)](https://pypi.python.org/pypi/pypsa)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/pypsa.svg)](https://anaconda.org/conda-forge/pypsa)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FPyPSA%2FPyPSA%2Fmaster%2Fpyproject.toml)
[![Tests](https://github.com/PyPSA/PyPSA/actions/workflows/test.yml/badge.svg)](https://github.com/PyPSA/PyPSA/actions/workflows/test.yml)
[![Documentation Status](https://readthedocs.org/projects/pypsa/badge/?version=latest)](https://docs.pypsa.org/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/PyPSA/PyPSA/master.svg)](https://results.pre-commit.ci/latest/github/PyPSA/PyPSA/master)
[![Code coverage](https://codecov.io/gh/PyPSA/PyPSA/branch/master/graph/badge.svg?token=kCpwJiV6Jr)](https://codecov.io/gh/PyPSA/PyPSA)
[![REUSE status](https://api.reuse.software/badge/github.com/pypsa/pypsa)](https://api.reuse.software/info/github.com/pypsa/pypsa)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/pypi/l/pypsa.svg)](LICENSE)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.3946412.svg)](https://doi.org/10.5281/zenodo.3946412)
[![Discord](https://img.shields.io/discord/911692131440148490?logo=discord)](https://discord.gg/AnuJBk23FU)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

PyPSA stands for **Python for Power System Analysis**. It is pronounced
**pipes-ah**. 

PyPSA is an open-source Python framework for optimising and simulating modern
power and energy systems that include features such as conventional generators
with unit commitment, variable wind and solar generation, hydro-electricity,
inter-temporal storage, coupling to other energy sectors, elastic demands, and
linearised power flow with loss approximations in DC and AC networks. PyPSA is
designed to scale well with large networks and long time series. It is made for
researchers, planners and utilities with basic coding aptitude who need a fast,
easy-to-use and transparent tool for power and energy system analysis.

PyPSA is maintained by the [Department of Digital Transformation in Energy
Systems](https://tu.berlin/en/ensys) at the [Technical University of
Berlin](https://www.tu.berlin). Previous versions were developed at the
[Karlsruhe Institute of Technology](http://www.kit.edu/english/index.php) funded
by the [Helmholtz Association](https://www.helmholtz.de/en/), and at
[FIAS](https://fias.uni-frankfurt.de/) funded by the [German Federal Ministry
for Education and Research (BMBF)](https://www.bmbf.de/bmbf/en/).

## Features

- **Economic Dispatch (ED):** Models short-term market-based dispatch including
unit commitment, renewable availability, short-duration and seasonal storage
including hydro reservoirs with inflow and spillage dynamics, elastic demands,
load shedding and conversion between energy carriers, using either perfect
operational foresight or rolling horizon time resolution.

- **Linear Optimal Power Flow (LOPF):** Extends economic dispatch to determine
the least-cost dispatch while respecting network constraints in meshed AC-DC
networks, using a linearised representation of power flow (KVL, KCL) with
optional loss approximations.

- **Security-Constrained LOPF (SCLOPF):** Extends LOPF by accounting for line
outage contingencies to ensure system reliability under $N-1$ conditions.

- **Capacity Expansion Planning (CEP):** Supports least-cost
long-term system planning with investment decisions for generation, storage,
conversion, and transmission infrastructure. Handles both single and multiple
investment periods. Continuous and discrete investments are supported.

- **Pathway Planning:** Supports co-optimisation of multiple investment periods to
plan energy system transitions over time with perfect planning foresight.

- **Stochastic Optimisation:** Implements two-stage stochastic programming
framework with scenario-weighted uncertain inputs, with investments as
first-stage decisions and dispatch as recourse decisions.

- **Modelling-to-Generate-Alternatives (MGA):** Explores near-optimal decision
spaces to provide insight into the range of feasible system configurations with
similar costs.

- **Sector-Coupling:** Modelling integrated energy systems with multiple energy
  carriers (electricity, heat, hydrogen, etc.) and conversion between them.
  Flexible representation of technologies such as heat pumps, electrolysers,
  battery electric vehicles (BEVs), direct air capture (DAC), and synthetic
  fuels production.

- **Static Power Flow Analysis:** Computes both full non-linear and linearised
  load flows for meshed AC and DC grids using Newton-Raphson method.

## Documentation

PyPSA has extensive [documentation](https://docs.pypsa.org) with tutorials, user guides, examples and an API reference.

## Installation

pip:

``` bash
pip install pypsa
```

conda/mamba:

``` bash
conda install -c conda-forge pypsa
```

uv:

``` bash
uv add pypsa
```

## Usage

``` py
import pypsa

# create a new network
n = pypsa.Network()
n.add("Bus", "mybus")
n.add("Load", "myload", bus="mybus", p_set=100)
n.add("Generator", "mygen", bus="mybus", p_nom=100, marginal_cost=20)

# load an example network
n = pypsa.examples.ac_dc_meshed()

# run the optimisation
n.optimize()

# plot results
n.generators_t.p.plot()
n.plot()

# get statistics
n.statistics()
n.statistics.energy_balance()
```

## Dependencies

PyPSA relies heavily on other open-source Python packages. Some of them are:

* [pandas](http://pandas.pydata.org/) for storing data about components and time series
* [numpy](http://www.numpy.org/) and [scipy](http://scipy.org/) for linear algebra and matrix calculations
* [linopy](https://github.com/PyPSA/linopy) for preparing optimisation problems
* [matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/) and [plotly](https://plotly.com/python/) for static and interactive plotting
* [networkx](https://networkx.github.io/) for network calculations
* [pytest](http://pytest.org/) for unit testing

Find the full list of dependencies in the [`pyproject.toml`](https://github.com/PyPSA/PyPSA/blob/master/pyproject.toml) file.

PyPSA can be used with different solvers. For instance, the free solvers
[HiGHS](https://highs.dev/) (installed by default), [GLPK](https://www.gnu.org/software/glpk/) and
[CBC](https://github.com/coin-or/Cbc/) or commercial solvers like
[Gurobi](http://www.gurobi.com/) or [FICO Xpress](https://www.fico.com/en/products/fico-xpress-optimization) for which free academic licenses are available.

## Contributing and Support

We strongly welcome anyone interested in contributing to this project. If you have any ideas, suggestions or encounter problems, feel invited to file issues or make pull requests on GitHub.

-   To **discuss** with other PyPSA users, organise projects, share news, and get in touch with the community you can use the [Discord server](https://discord.gg/AnuJBk23FU).
-   For **bugs and feature requests**, please [open an issue](https://github.com/PyPSA/PyPSA/issues).
-   For **troubleshooting and support**, please check the [troubleshooting](https://docs.pypsa.org/latest/user-guide/support/) and [support](https://docs.pypsa.org/latest/user-guide/support/) sectionsin the documentation.

Detailed guidelines can be found in the [Contributing](https://docs.pypsa.org/latest/contributing/contributing/) guidelines of our documentation.

## Code of Conduct

Please respect our [Code of Conduct](https://docs.pypsa.org/latest/contributing/code-of-conduct/).

## Citing PyPSA

If you use PyPSA for your research, we would appreciate it if you would
cite the following paper:

-   T. Brown, J. Hörsch, D. Schlachtberger, [PyPSA: Python for Power
    System Analysis](https://arxiv.org/abs/1707.09913), 2018, [Journal
    of Open Research
    Software](https://openresearchsoftware.metajnl.com/), 6(1),
    [arXiv:1707.09913](https://arxiv.org/abs/1707.09913),
    [DOI:10.5334/jors.188](https://doi.org/10.5334/jors.188)

Please use the following BibTeX:

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

If you want to cite a specific PyPSA version, each release of PyPSA is archived
on [Zenodo](https://zenodo.org/) with a release-specific DOI:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3946412.svg)](https://doi.org/10.5281/zenodo.3946412)

## Licence

Copyright [PyPSA Contributors](https://docs.pypsa.org/latest/developers.html)

PyPSA is licensed under the open source [MIT License](LICENSES/MIT.txt).
The documentation is licensed under [CC-BY-4.0](LICENSES/CC-BY-4.0.txt).

The repository uses [REUSE](https://reuse.software/) to expose the licenses of its files.
