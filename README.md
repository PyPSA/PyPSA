<picture align="center">
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/PyPSA/PyPSA/refs/heads/new-docs/docs/assets/logo/logo-primary-dark.svg">
  <img alt="PyPSA Banner" src="https://raw.githubusercontent.com/PyPSA/PyPSA/refs/heads/new-docs/docs/assets/logo/logo-primary-light.svg">
</picture>

# PyPSA - Python for Power System Analysis


[![PyPI version](https://img.shields.io/pypi/v/pypsa.svg)](https://pypi.python.org/pypi/pypsa)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/pypsa.svg)](https://anaconda.org/conda-forge/pypsa)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FPyPSA%2FPyPSA%2Fmaster%2Fpyproject.toml)
![Static Badge](https://img.shields.io/badge/latest-%23d10949?logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyBpZD0iTG9nbyIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB2ZXJzaW9uPSIxLjEiIHZpZXdCb3g9IjAgMCA0NTAgNDUwIj4KICA8IS0tIEdlbmVyYXRvcjogQWRvYmUgSWxsdXN0cmF0b3IgMjkuNC4wLCBTVkcgRXhwb3J0IFBsdWctSW4gLiBTVkcgVmVyc2lvbjogMi4xLjAgQnVpbGQgMTUyKSAgLS0%2BCiAgPGRlZnM%2BCiAgICA8c3R5bGU%2BCiAgICAgIC5zdDAgewogICAgICAgIGZpbGw6ICMyOTkzYjU7CiAgICAgIH0KCiAgICAgIC5zdDEgewogICAgICAgIGZpbGw6ICMwYTg3NTQ7CiAgICAgIH0KCiAgICAgIC5zdDIgewogICAgICAgIGZpbGw6ICNmZmY7CiAgICAgIH0KCiAgICAgIC5zdDMgewogICAgICAgIGZpbGw6ICNkMTBhNDk7CiAgICAgIH0KCiAgICAgIC5zdDQgewogICAgICAgIGZpbGw6ICNmZmJmMDA7CiAgICAgIH0KCiAgICAgIC5zdDUgewogICAgICAgIGRpc3BsYXk6IG5vbmU7CiAgICAgIH0KICAgIDwvc3R5bGU%2BCiAgPC9kZWZzPgogIDxnIGlkPSJMZWZ0X1RyaWFuZ2xlIj4KICAgIDxwYXRoIGNsYXNzPSJzdDMiIGQ9Ik0yMTkuMzgsMTQyLjQ5bC0xMTEuNTQsMTg5LjYyYy05Ljg0LTEzLjkyLTI1LjYzLTIzLjM1LTQzLjY3LTI0LjU5bDExMi4wMi0xOTAuNDRjOS41OSwxNC4xMywyNS4yMiwyMy44Myw0My4xOCwyNS40MWgwWiIvPgogIDwvZz4KICA8ZyBpZD0iUmlnaHRfVHJpYW5nbGUiPgogICAgPHBhdGggY2xhc3M9InN0MyIgZD0iTTM4NC43MSwzMDcuMTFjLTEzLjE5LDEuMTctMjYuMDQsNi43OS0zNi4xNCwxNi44OS0yLjY2LDIuNjYtNS4wMSw1LjUxLTcuMDQsOC41MWwtMTExLjc4LTE5MC4wMmMxNy45Ni0xLjU3LDMzLjU5LTExLjI4LDQzLjE4LTI1LjQxbDExMS43NywxOTAuMDNoMFoiLz4KICA8L2c%2BCiAgPGcgaWQ9IkJvdHRvbV9UcmlhbmdsZSI%2BCiAgICA8cGF0aCBjbGFzcz0ic3QzIiBkPSJNMzM3LjA4LDM5MC4zN0gxMTMuMTZjMy40Ny03LjQ2LDUuMzktMTUuNzcsNS4zOS0yNC41NHMtMi4xMS0xNy44My01Ljg3LTI1LjU2aDIyNC4zOGMtNy41LDE1LjgxLTcuNSwzNC4yOSwwLDUwLjFoMFoiLz4KICA8L2c%2BCiAgPGcgaWQ9IlRvcF9NYXNrIiBjbGFzcz0ic3Q1Ij4KICAgIDxwYXRoIGNsYXNzPSJzdDIiIGQ9Ik0yODMuMjUsODQuMjZjMCwxMi4yMS0zLjczLDIzLjU1LTEwLjEyLDMyLjk0LTkuNjMsMTQuMTgtMjUuMzIsMjMuOTItNDMuMzUsMjUuNTEtMS43Mi4xNS0zLjQ1LjIzLTUuMjEuMjNzLTMuNDktLjA4LTUuMjEtLjIzYy0xOC4wMy0xLjU4LTMzLjcyLTExLjMyLTQzLjM1LTI1LjUxLTYuMzktOS4zOS0xMC4xMi0yMC43My0xMC4xMi0zMi45NCwwLTMyLjQsMjYuMjctNTguNjcsNTguNjctNTguNjdzNTguNjcsMjYuMjcsNTguNjcsNTguNjdoLjAyWiIvPgogIDwvZz4KICA8ZyBpZD0iTGVmdF9NYXNrIiBjbGFzcz0ic3Q1Ij4KICAgIDxwYXRoIGNsYXNzPSJzdDIiIGQ9Ik0xMTguNzgsMzY1LjMyYzAsOC44LTEuOTQsMTcuMTQtNS40MSwyNC42My05LjMsMjAuMS0yOS42NiwzNC4wNC01My4yNiwzNC4wNC0zMi40LDAtNTguNjgtMjYuMjctNTguNjgtNTguNjdzMjYuMjctNTguNjcsNTguNjctNTguNjdjMS4zNywwLDIuNzMuMDUsNC4wOC4xNCwxOC4xMSwxLjI0LDMzLjk2LDEwLjcsNDMuODQsMjQuNjgsMS44NCwyLjU4LDMuNDYsNS4zMiw0Ljg2LDguMiwzLjc3LDcuNzUsNS44OSwxNi40NSw1Ljg5LDI1LjY2aC4wMVoiLz4KICA8L2c%2BCiAgPGcgaWQ9IlJpZ2h0X01hc2siIGNsYXNzPSJzdDUiPgogICAgPHBhdGggY2xhc3M9InN0MiIgZD0iTTQzMS4zNyw0MDYuODFjLTIyLjkyLDIyLjkyLTYwLjA3LDIyLjkyLTgyLjk3LDAtNC45LTQuOS04Ljc0LTEwLjQ0LTExLjU0LTE2LjM1LTcuNTMtMTUuODgtNy41My0zNC40MywwLTUwLjI5LDEuMjctMi42OCwyLjc3LTUuMyw0LjQ4LTcuOCwyLjA0LTMuMDEsNC40LTUuODgsNy4wNy04LjU0LDEwLjEzLTEwLjEzLDIzLjA0LTE1Ljc5LDM2LjI4LTE2Ljk2LDE2LjctMS40OCwzMy45MSw0LjE3LDQ2LjcsMTYuOTYsMjIuOTIsMjIuOTEsMjIuOTIsNjAuMDYsMCw4Mi45N2gtLjAyWiIvPgogIDwvZz4KICA8cGF0aCBpZD0iUmlnaHRfQ2lyY2xlIiBjbGFzcz0ic3QwIiBkPSJNMzg5Ljg5LDQxNS40M2MxMy4yOSwwLDI2LjAzLTUuMjgsMzUuNDMtMTQuNjcsOS4zOS05LjM5LDE0LjY3LTIyLjEzLDE0LjY3LTM1LjQzcy01LjI4LTI2LjAzLTE0LjY3LTM1LjQzYy05LjM5LTkuMzktMjIuMTMtMTQuNjctMzUuNDMtMTQuNjdzLTI2LjAzLDUuMjgtMzUuNDMsMTQuNjdjLTkuNCw5LjM5LTE0LjY3LDIyLjEzLTE0LjY3LDM1LjQzczUuMjgsMjYuMDMsMTQuNjcsMzUuNDNjOS4zOSw5LjM5LDIyLjEzLDE0LjY3LDM1LjQzLDE0LjY3aDBaIi8%2BCiAgPHBhdGggaWQ9IlRvcF9DaXJjbGUiIGNsYXNzPSJzdDQiIGQ9Ik0yMjQuNTcsMTM0LjM3YzEzLjI5LDAsMjYuMDMtNS4yOCwzNS40My0xNC42Nyw5LjM5LTkuMzksMTQuNjctMjIuMTMsMTQuNjctMzUuNDNzLTUuMjgtMjYuMDMtMTQuNjctMzUuNDNjLTkuMzktOS4zOS0yMi4xMy0xNC42Ny0zNS40My0xNC42N3MtMjYuMDMsNS4yOC0zNS40MywxNC42N2MtOS4zOSw5LjM5LTE0LjY3LDIyLjEzLTE0LjY3LDM1LjQzczUuMjgsMjYuMDMsMTQuNjcsMzUuNDNjOS4zOSw5LjM5LDIyLjEzLDE0LjY3LDM1LjQzLDE0LjY3aDBaIi8%2BCiAgPHBhdGggaWQ9IkxlZnRfQ2lyY2xlIiBjbGFzcz0ic3QxIiBkPSJNNjAuMTEsNDE1LjQzYzEzLjI5LDAsMjYuMDMtNS4yOCwzNS40My0xNC42Nyw5LjM5LTkuMzksMTQuNjctMjIuMTMsMTQuNjctMzUuNDNzLTUuMjgtMjYuMDMtMTQuNjctMzUuNDNjLTkuMzktOS4zOS0yMi4xMy0xNC42Ny0zNS40My0xNC42N3MtMjYuMDMsNS4yOC0zNS40MywxNC42N2MtOS4zOSw5LjM5LTE0LjY3LDIyLjEzLTE0LjY3LDM1LjQzczUuMjgsMjYuMDMsMTQuNjcsMzUuNDNjOS4zOSw5LjM5LDIyLjEzLDE0LjY3LDM1LjQzLDE0LjY3WiIvPgo8L3N2Zz4%3D&label=PyPSA&labelColor=%23293036&link=https%3A%2F%2Fpypsa.readthedocs.io)
[![Tests](https://github.com/PyPSA/PyPSA/actions/workflows/test.yml/badge.svg)](https://github.com/PyPSA/PyPSA/actions/workflows/test.yml)
[![Documentation Status](https://readthedocs.org/projects/pypsa/badge/?version=latest)](https://pypsa.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/PyPSA/PyPSA/master.svg)](https://results.pre-commit.ci/latest/github/PyPSA/PyPSA/master)
[![Code coverage](https://codecov.io/gh/PyPSA/PyPSA/branch/master/graph/badge.svg?token=kCpwJiV6Jr)](https://codecov.io/gh/PyPSA/PyPSA)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/pypi/l/pypsa.svg)](LICENSE.txt)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.3946412.svg)](https://doi.org/10.5281/zenodo.3946412)
[![Discord](https://img.shields.io/discord/911692131440148490?logo=discord)](https://discord.gg/AnuJBk23FU)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

PyPSA stands for "Python for Power System Analysis". It is pronounced
"pipes-ah".

PyPSA is an open source toolbox for simulating and optimising modern power and
energy systems that include features such as conventional generators with unit
commitment, variable wind and solar generation, storage units, coupling to other
energy sectors, and mixed alternating and direct current networks. PyPSA is
designed to scale well with large networks and long time series.

This project is maintained by the [Department of Digital Transformation in
Energy Systems](https://www.tu.berlin/ensys) at the [Technical University of
Berlin](https://www.tu.berlin). Previous versions were developed by the Energy
System Modelling group at the [Institute for Automation and Applied
Informatics](https://www.iai.kit.edu/english/index.php) at the [Karlsruhe
Institute of Technology](http://www.kit.edu/english/index.php) funded by the
[Helmholtz Association](https://www.helmholtz.de/en/), and by the [Renewable
Energy
Group](https://fias.uni-frankfurt.de/physics/schramm/renewable-energy-system-and-network-analysis/)
at [FIAS](https://fias.uni-frankfurt.de/en/) to carry out simulations for the
[CoNDyNet project](https://fias.institute/en/projects/condynet/), financed by the [German Federal
Ministry for Education and Research (BMBF)](https://www.bmbf.de/bmbf/en/)
as part of the [Stromnetze Research
Initiative](http://forschung-stromnetze.info/projekte/grundlagen-und-konzepte-fuer-effiziente-dezentrale-stromnetze/).

## Functionality

PyPSA can calculate:

-   static power flow (using both the full non-linear network equations and the
    linearised network equations)
-   linear optimal power flow (least-cost optimisation of power plant and
    storage dispatch within network constraints, using the linear network
    equations, over several snapshots)
-   security-constrained linear optimal power flow
-   total electricity/energy system least-cost investment optimisation (using
    linear network equations, over several snapshots and investment periods
    simultaneously for optimisation of generation and storage dispatch and
    investment in the capacities of generation, storage, transmission and other
    infrastructure)

It has models for:

-   meshed multiply-connected AC and DC networks, with controllable converters
    between AC and DC networks
-   standard types for lines and transformers following the implementation in
    [pandapower](https://www.pandapower.org/)
-   conventional dispatchable generators and links with unit commitment
-   generators with time-varying power availability, such as wind and solar
    generators
-   storage units with efficiency losses
-   simple hydroelectricity with inflow and spillage
-   coupling with other energy carriers (e.g. resistive Power-to-Heat (P2H),
    Power-to-Gas (P2G), battery electric vehicles (BEVs), Fischer-Tropsch,
    direct air capture (DAC))
-   basic components out of which more complicated assets can be built, such as
    Combined Heat and Power (CHP) units and heat pumps.

## Documentation

* [Documentation](https://pypsa.readthedocs.io/en/latest/index.html)

    * [Quick start](https://pypsa.readthedocs.io/en/latest/quick_start.html)

    * [Examples](https://pypsa.readthedocs.io/en/latest/examples-index/lopf.html)

    * [Known users of PyPSA](https://pypsa.readthedocs.io/en/latest/users.html)

## Installation

pip:

```pip install pypsa```

conda/mamba:

```conda install -c conda-forge pypsa```

Additionally, install a solver (see [here](https://pypsa.readthedocs.io/en/latest/getting-started/installation.html#getting-a-solver)).

## Usage

```py
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

There are [more extensive
examples](https://pypsa.readthedocs.io/en/latest/examples-basic.html) available
as [Jupyter notebooks](https://jupyter.org/). They are also available as Python scripts in
[examples/notebooks/](examples/notebooks/) directory.

## Screenshots

[PyPSA-Eur](https://github.com/PyPSA/pypsa-eur) optimising capacities of
generation, storage and transmission lines (9% line volume expansion allowed)
for a 95% reduction in CO2 emissions in Europe compared to 1990 levels

![image](doc/img/elec_s_256_lv1.09_Co2L-3H.png)

[SciGRID model](https://power.scigrid.de/) simulating the German power system
for 2015.

![image](doc/img/stacked-gen_and_storage-scigrid.png)

![image](doc/img/lmp_and_line-loading.png)

## Dependencies

PyPSA is written and tested to be compatible with Python 3.10 and above.
The last release supporting Python 2.7 was PyPSA 0.15.0.

It leans heavily on the following Python packages:

-   [pandas](http://pandas.pydata.org/) for storing data about
    components and time series
-   [numpy](http://www.numpy.org/) and [scipy](http://scipy.org/) for
    calculations, such as linear algebra and sparse matrix calculations
-   [networkx](https://networkx.org/) for some network
    calculations
-   [matplotlib](https://matplotlib.org/) for static plotting
-   [linopy](https://github.com/PyPSA/linopy) for preparing optimisation problems
    (currently only linear and mixed integer linear optimisation)
-   [cartopy](https://scitools.org.uk/cartopy) for plotting the
    baselayer map
-   [pytest](https://docs.pytest.org/) for unit testing
-   [logging](https://docs.python.org/3/library/logging.html) for
    managing messages

Find the full list of dependencies in the 
[dependency graph](https://github.com/PyPSA/PyPSA/network/dependencies).

The optimisation uses interface libraries like `linopy` which are independent of
the preferred solver. You can use e.g. one of the free solvers
[HiGHS](https://highs.dev/), [GLPK](https://www.gnu.org/software/glpk/) and
[CLP/CBC](https://github.com/coin-or/Cbc/) or the commercial solver
[Gurobi](http://www.gurobi.com/) for which free academic licenses are available.

## Contributing and Support

We strongly welcome anyone interested in contributing to this project. If you have any ideas, suggestions or encounter problems, feel invited to file issues or make pull requests on GitHub.

-   To **discuss** with other PyPSA users, organise projects, share news, and get in touch with the community you can use the [Discord server](https://discord.gg/AnuJBk23FU).
-   For **bugs and feature requests**, please use the [PyPSA Github Issues page](https://github.com/PyPSA/PyPSA/issues).
-   For **troubleshooting**, please check the [troubleshooting](https://pypsa.readthedocs.io/en/latest/troubleshooting.html) in the documentation.

Detailed guidelines can be found in the [Contributing](https://pypsa.readthedocs.io/en/latest/contributing.html) section of our documentation.

## Code of Conduct

Please respect our [code of conduct](CODE_OF_CONDUCT.md).

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

If you want to cite a specific PyPSA version, each release of PyPSA is
stored on [Zenodo](https://zenodo.org/) with a release-specific DOI. The
release-specific DOIs can be found linked from the overall PyPSA Zenodo
DOI for Version 0.17.1 and onwards:

[![image](https://zenodo.org/badge/DOI/10.5281/zenodo.3946412.svg)](https://doi.org/10.5281/zenodo.3946412)

or from the overall PyPSA Zenodo DOI for Versions up to 0.17.0:

[![image](https://zenodo.org/badge/DOI/10.5281/zenodo.786605.svg)](https://doi.org/10.5281/zenodo.786605)

# Licence

Copyright 2015-2025 [PyPSA
Developers](https://pypsa.readthedocs.io/en/latest/developers.html)

PyPSA is licensed under the open source [MIT
License](https://github.com/PyPSA/PyPSA/blob/master/LICENSE.txt).
