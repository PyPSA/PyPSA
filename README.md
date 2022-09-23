# PyPSA - Python for Power System Analysis


[![PyPI version](https://img.shields.io/pypi/v/pypsa.svg)](https://pypi.python.org/pypi/pypsa)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/pypsa.svg)](https://anaconda.org/conda-forge/pypsa)
[![CI](https://github.com/pypsa/pypsa/actions/workflows/CI.yml/badge.svg)](https://github.com/pypsa/pypsa/actions/workflows/CI.yml)
[![CI with conda](https://github.com/pypsa/pypsa/actions/workflows/CI-conda.yml/badge.svg)](https://github.com/pypsa/pypsa/actions/workflows/CI-conda.yml)
[![Code coverage](https://codecov.io/gh/PyPSA/PyPSA/branch/master/graph/badge.svg?token=kCpwJiV6Jr)](https://codecov.io/gh/PyPSA/PyPSA)
[![Documentation Status](https://readthedocs.org/projects/pypsa/badge/?version=latest)](https://pypsa.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/pypi/l/pypsa.svg)](LICENSE.txt)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.3946412.svg)](https://doi.org/10.5281/zenodo.3946412)
[![Examples of use](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/PyPSA/PyPSA/master?filepath=examples%2Fnotebooks)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/PyPSA/PyPSA/master.svg)](https://results.pre-commit.ci/latest/github/PyPSA/PyPSA/master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

PyPSA stands for "Python for Power System Analysis". It is pronounced
"pipes-ah".

PyPSA is an open source toolbox for simulating and optimising modern power and
energy systems that include features such as conventional generators with unit
commitment, variable wind and solar generation, storage units, coupling to other
energy sectors, and mixed alternating and direct current networks. PyPSA is
designed to scale well with large networks and long time series.

This project is maintained by the [Department of Digital Transformation in
Energy Systems](https://tub-ensys.github.io) at the [Technical University of
Berlin](https://www.tu.berlin). Previous versions were developed by the Energy
System Modelling group at the [Institute for Automation and Applied
Informatics](https://www.iai.kit.edu/english/index.php) at the [Karlsruhe
Institute of Technology](http://www.kit.edu/english/index.php) funded by the
[Helmholtz Association](https://www.helmholtz.de/en/), and by the [Renewable
Energy
Group](https://fias.uni-frankfurt.de/physics/schramm/renewable-energy-system-and-network-analysis/)
at [FIAS](https://fias.uni-frankfurt.de/) to carry out simulations for the
[CoNDyNet project](http://condynet.de/), financed by the [German Federal
Ministry for Education and Research (BMBF)](https://www.bmbf.de/en/index.html)
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
-   conventional dispatchable generators with unit commitment
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

[Documentation](https://pypsa.readthedocs.io/en/latest/index.html)

[Quick start](https://pypsa.readthedocs.io/en/latest/quick_start.html)

[Examples](https://pypsa.readthedocs.io/en/latest/examples-basic.html)

[Known users of
PyPSA](https://pypsa.readthedocs.io/en/latest/users.html)

## Installation

pip:

```pip install pypsa```

conda/mamba:

```conda install -c conda-forge pypsa```

Additionally, install a solver.

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
n.lopf()

# plot results
n.generators_t.p.plot()
n.plot()
```

There are [more extensive
examples](https://pypsa.readthedocs.io/en/latest/examples-basic.html) available
as [Jupyter notebooks](https://jupyter.org/). They are also described in the
[doc/examples.rst](doc/examples.rst) and are available as Python scripts in
[examples/](examples/).

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

PyPSA is written and tested to be compatible with Python 3.7 and above.
The last release supporting Python 2.7 was PyPSA 0.15.0.

It leans heavily on the following Python packages:

-   [pandas](http://pandas.pydata.org/) for storing data about
    components and time series
-   [numpy](http://www.numpy.org/) and [scipy](http://scipy.org/) for
    calculations, such as linear algebra and sparse matrix calculations
-   [networkx](https://networkx.github.io/) for some network
    calculations
-   [matplotlib](https://matplotlib.org/) for static plotting
-   [pyomo](http://www.pyomo.org/) for preparing optimisation problems
    (currently only linear)
-   [cartopy](https://scitools.org.uk/cartopy) for plotting the
    baselayer map
-   [pytest](http://pytest.org/) for unit testing
-   [logging](https://docs.python.org/3/library/logging.html) for
    managing messages

The optimisation uses interface libraries like `pyomo` which are
independent of the preferred solver. You can use e.g. one of the free
solvers [GLPK](https://www.gnu.org/software/glpk/) and
[CLP/CBC](https://github.com/coin-or/Cbc/) or the commercial solver
[Gurobi](http://www.gurobi.com/) for which free academic licenses are
available.

## Mailing list

PyPSA has a Google Group [forum / mailing
list](https://groups.google.com/group/pypsa) where announcements of new
releases can be made and questions can be asked.

To discuss issues and suggest/contribute features for future development
we prefer ticketing through the [PyPSA Github Issues
page](https://github.com/PyPSA/PyPSA/issues).

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

Copyright 2015-2022 [PyPSA
Developers](https://pypsa.readthedocs.io/en/latest/developers.html)

PyPSA is licensed under the open source [MIT
License](https://github.com/PyPSA/PyPSA/blob/master/LICENSE.txt).
