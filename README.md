# PyPSA-Eur: An Open Optimisation Model of the European Transmission System

PyPSA-Eur is an open model dataset of the European power system at the
transmission network level that covers the full ENTSO-E area.

![PyPSA-Eur Grid Model](https://raw.githubusercontent.com/PyPSA/pypsa-eur/master/img/pypsa-eur-grid.png)


The model is described and partially validated in the paper
[PyPSA-Eur: An Open Optimisation Model of the European Transmission
System](https://arxiv.org/abs/1806.01613), 2018,
[arXiv:1806.01613](https://arxiv.org/abs/1806.01613).

This repository contains the scripts and some of the data required to
automatically build the dataset from openly-available sources.

Already-built versions of the model can be found in the accompanying [Zenodo
repository](https://zenodo.org/record/1246851).

The model is designed to be imported into the open toolbox
[PyPSA](https://github.com/PyPSA/PyPSA) for operational studies as
well as generation and transmission expansion planning studies.

The dataset consists of:

- A grid model based on a modified [GridKit](https://github.com/bdw/GridKit)
  extraction of the [ENTSO-E Transmission System
  Map](https://www.entsoe.eu/data/map/). The grid model contains 6001 lines
  (alternating current lines at and above 220kV voltage level and all high
  voltage direct current lines) and 3657 substations.
- The open power plant database
  [powerplantmatching](https://github.com/FRESNA/powerplantmatching).
- Electrical demand time series from the
  [OPSD project](https://open-power-system-data.org/).
- Renewable time series based on ERA5 and SARAH, assembled using the [atlite tool](https://github.com/FRESNA/atlite).
- Geographical potentials for wind and solar generators based on land use (CORINE) and excluding nature reserves (Natura2000) are computed with the [vresutils library](https://github.com/FRESNA/vresutils).

Building the model with the scripts in this repository uses up to 20GB of
memory. Computing optimal investment and operation scenarios requires a strong
interior-point solver compatible with the modelling library
[PYOMO](https://github.com/Pyomo/pyomo) like Gurobi or CPLEX with up to 100GB of
memory (for the 356-bus approximation).

This project is maintained by the [Energy System Modelling
group](https://www.iai.kit.edu/english/2338.php) at the [Institute for
Automation and Applied
Informatics](https://www.iai.kit.edu/english/index.php) at the
[Karlsruhe Institute of
Technology](http://www.kit.edu/english/index.php). It is currently
funded by the [Helmholtz
Association](https://www.helmholtz.de/en/). Previous versions were
developed by the [Renewable Energy
Group](https://fias.uni-frankfurt.de/physics/schramm/renewable-energy-system-and-network-analysis/)
at [FIAS](https://fias.uni-frankfurt.de/) to carry out simulations for
the [CoNDyNet project](http://condynet.de/), financed by the [German
Federal Ministry for Education and Research
(BMBF)](https://www.bmbf.de/en/index.html) as part of the [Stromnetze
Research
Initiative](http://forschung-stromnetze.info/projekte/grundlagen-und-konzepte-fuer-effiziente-dezentrale-stromnetze/).

# Installation

The steps are demonstrated as shell commands, where the path before the `%` sign denotes the
directory in which the commands following the `%` should be entered.

Clone the repository using `git` (**to a directory without any spaces in the path**)
```shell
/some/other/path % cd /some/path/without/spaces
/some/path/without/spaces % git clone https://github.com/PyPSA/pypsa-eur.git
```

## Python dependencies
The python package requirements are curated in the conda [environment.yaml](environment.yaml) file.
The environment can be installed and activated using
```shell
.../pypsa-eur % conda env create -f environment.yaml
.../pypsa-eur % conda activate pypsa-eur   # or source activate pypsa-eur on older linux installations
```

**Note that activation is local to the currently open shell! After opening a new terminal window, one needs to reissue the second command!**

## Data dependencies
Not all data dependencies are shipped with the git repository (since git is not suited for handling large changing files). Instead we provide two separate data bundles:
1. [pypsa-eur-data-bundle.tar.xz](https://vfs.fias.science/d/0a0ca1e2fb/files/?p=/pypsa-eur-data-bundle.tar.xz) contains common GIS datasets like NUTS3 shapes, EEZ shapes, CORINE Landcover, Natura 2000 and also electricity specific summary statistics like historic per country yearly totals of hydro generation, GDP and POP on NUTS3 levels and per-country load time-series. It should be extracted in the `data` subdirectory (so that all files are in the `data/bundle` subdirectory)
```shell
.../pypsa-eur/data % curl -OL "https://vfs.fias.science/d/0a0ca1e2fb/files/?dl=1&p=/pypsa-eur-data-bundle.tar.xz"
.../pypsa-eur/data % tar xJf pypsa-eur-data-bundle.tar.xz
```
2. [pypsa-eur-cutouts.tar.xz](https://vfs.fias.science/d/0a0ca1e2fb/files/?p=/pypsa-eur-cutouts.tar.xz) are spatiotemporal subsets of the European weather data from the [ECMWF ERA5](https://software.ecmwf.int/wiki/display/CKB/ERA5+data+documentation) reanalysis dataset and the [CMSAF SARAH-2](https://wui.cmsaf.eu/safira/action/viewDoiDetails?acronym=SARAH_V002) solar surface radiation dataset for the year 2013. They have been prepared by and are for use with the [atlite](https://github.com/FRESNA/atlite) tool. You can either generate them yourself using the `build_cutouts` snakemake rule or extract them directly in the `pypsa-eur` directory (extracting the bundle is recommended, since procuring the source weather data files for atlite is not properly documented at the moment):
```shell
.../pypsa-eur % curl -OL "https://vfs.fias.science/d/0a0ca1e2fb/files/?dl=1&p=/pypsa-eur-cutouts.tar.xz"
.../pypsa-eur % tar xJf pypsa-eur-cutouts.tar.xz
```

3. Optionally, you can download a rasterized version of the NATURA dataset [natura.tiff](https://vfs.fias.science/d/0a0ca1e2fb/files/?p=/natura.tiff&dl=1) and put it into the `resources` sub-directory. If you don't, it will be generated automatically, which takes several hours.

```shell
.../pypsa-eur % curl -L "https://vfs.fias.science/d/0a0ca1e2fb/files/?p=/natura.tiff&dl=1" -o "resources/natura.tiff"
```

4. Optionally, if you want to save disk space, you can delete `data/pypsa-eur-data-bundle.tar.xz` and `pypsa-eur-cutouts.tar.xz` once extracting the bundles is complete. E.g.

```shell
.../pypsa-eur % rm -rf data/pypsa-eur-data-bundle.tar.xz pypsa-eur-cutouts.tar.xz
```

# Script overview

The model has several configuration options collected in the [config.yaml](config.yaml) file
located in the root directory.

## Model workflow
The generation of the model is controlled by the workflow management system
[Snakemake](https://snakemake.bitbucket.io/). In a nutshell, one declares in the
`Snakefile` for each python script in the `scripts` directory a rule which
describes which files the scripts consume and produce. `snakemake` then runs the
scripts in the correct order and is able to track, what parts of the workflow
have to be regenerated, when a data file or script is updated. For instance,
with the [Snakefile of pypsa-eur](Snakefile), an invocation to
```shell
snakemake networks/elec_s_128.nc
```
follows the dependency graph
![Dependency graph for network elec_s_128](img/dependencies-elec_s_128.png)

## Building the network
In detail this means it has to run the independent scripts,
- `build_shapes` to generate GeoJSON files with country, exclusive economic zones and nuts3 shapes
- `build_cutout` to prepare smaller weather data portions from ERA5 for cutout `europe-2013-era5` and SARAH for cutout `europe-2013-sarah`.

With these and the externally extracted `ENTSO-E online map topology`, it can build the PyPSA basis model
- `base_network` stored at `networks/base.nc` with all `buses`, HVAC `lines` and HVDC `links`, and in
- `build_bus_regions` determine the Voronoi cell of each substation.

Then it hands these over to the scripts for generating renewable and hydro feedin data,
- `build_hydro_profile` for the hourly hydro energy availability,
- `build_renewable_potentials` for the landuse/natura2000 constrained installation potentials for PV and wind,
- `build_renewable_profiles` for the PV and wind hourly capacity factors in each Voronoi cell.
- `build_powerplants` uses [powerplantmatching](https://github.com/FRESNA/powerplantmatching) to determine today's thermal power plant capacities and then locates the closest substation for each powerplant.

The central rule `add_electricity` then ties all the different data inputs together to a detailed PyPSA model stored in `networks/elec.nc`, containing:

- Today's transmission topology and capacities (optionally including lines which are under construction according to the config settings `lines: under_construction` and `links: under_construction`)
- Today's thermal and hydro generation capacities (for the technologies listed in the config setting `electricity: conventional_carriers`)
- Today's load time-series (upsampled according to population and gross domestic product)

It further adds extendable `generators` and `storage_units` with *zero* capacity for
- wind and pv installations with today's locational, hourly wind and solar pv capacity factors (but **no** capacities)
- long-term hydrogen and short-term battery storage units (if listed in `electricity: extendable_carriers`)
- additional open-cycle gas turbines (if `OCGT` is listed in `electricity: extendable_carriers`)

The additional rules prepare approximations of the full model, in which generation, storage and transmission capacities can be co-optimized
- `simplify_network` transforms the transmission grid to a 380 kV-only equivalent network, while
- `cluster_network` uses a kmeans based clustering technique to partition the network into a certain number of zones and then reduce the network to a representation with one bus per zone.

The simplification and clustering steps are described in detail in the paper
[The role of spatial scale in joint optimisations of generation and transmission for European highly renewable scenarios](https://arxiv.org/abs/1705.07617), 2017, [arXiv:1705.07617](https://arxiv.org/abs/1705.07617), [doi:10.1109/EEM.2017.7982024](https://doi.org/10.1109/EEM.2017.7982024).

## Solving the network
After generating the network it can be solved by using 'solve_all_elec_networks'. This runs the following rules:
- 'cluster_network'
- 'prepare_network'
- 'solve_all_elec_networks'
- 'solve_network'

## Summarising the results and making plots
The following rule can be used to summarize the results in seperate .csv files:
```
snakemake results/summaries/elec_s_all_lall_Co2L-3H_all
                                      ^ clusters
                                          ^ line volume or cost cap
                                               ^- options
                                                       ^- all countries
```
the line volume/cost cap field can be set to one of the following:
* `lv1.25` for a particular line volume extension by 25%
* `lc1.25` for a line cost extension by 25 %
* `lall` for all evalutated caps
* `lvall` for all line volume caps
* `lcall` for all line cost caps

Replacing '/summaries/' with '/plots/' creates nice colored maps of the results.

# Solver choice
Default choice for the solver is Gurobi (freely available under academic license) or CPLEX. If you want to go fully opensource the CBC solver (https://projects.coin-or.org/Cbc) can be used. To install CBC run 'conda install -c conda-forge coincbc'.

# Hints

For the use of `snakemake`, it makes sense to familiarize oneself quickly with its [basic tutorial](https://snakemake.readthedocs.io/en/stable/tutorial/basics.html) and then read carefully through the section [Executing Snakemake](https://snakemake.readthedocs.io/en/stable/executable.html), noting the arguments `-n`, `-r`, but also `--dag`, `-R` and `-t`.

The dependency graph shown above was generated using
```shell
snakemake --dag networks/elec_s_128.nc | dot -Tpng > dependency-graph-elec_s_128.png
```

# License

The code in PyPSA-Eur is released as free software under the
[GPLv3](http://www.gnu.org/licenses/gpl-3.0.en.html), see
[LICENSE](LICENSE.txt).
