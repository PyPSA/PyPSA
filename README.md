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

- A grid model based on the [GridKit](https://github.com/bdw/GridKit)
  extraction of the [ENTSO-E Transmission System
  Map](https://www.entsoe.eu/data/map/). The grid model contains 6001
  lines (alternating current lines at and above 220kV voltage level
  and all high voltage direct current lines) and 3657 substations.
- The open power plant database
  [powerplantmatching](https://github.com/FRESNA/powerplantmatching).
- Electrical demand time series from the
  [OPSD project](https://open-power-system-data.org/).
- Renewable time series based on ERA5 and SARAH, assembled using the [atlite tool](https://github.com/FRESNA/atlite).
- Geographical potentials for wind and solar generators based on land use (CORINE) and excluding nature reserves (Natura2000) are computed with the [vresutils library](https://github.com/FRESNA/vresutils).


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

Change to an directory of your choosing (***the directory and all of it parent directories may not contain any spaces***) and clone the pypsa-eur git repository
```shell
cd /home/quack/vres
git clone https://github.com/PyPSA/pypsa-eur.git
```

## Python dependencies
The python package requirements are curated in the conda [environment.yaml](https://github.com/FRESNA/pypsa-eur/blob/master/environment.yaml) file.
The environment can be installed and activated using
```shell
cd pypsa-eur
conda env create -f environment.yaml
source activate pypsa-eur   # or conda activate pypsa-eur on windows
```

## Data dependencies
Not all data dependencies are shipped with the git repository (since git is not suited for handling large changing files). Instead we provide two separate data bundles:
1. [pypsa-eur-data-bundle.tar.xz](https://vfs.fias.science/d/0a0ca1e2fb/files/?p=/pypsa-eur-data-bundle.tar.xz) contains common GIS datasets like NUTS3 shapes, EEZ shapes, CORINE Landcover, Natura 2000 and also electricity specific summary statistics like historic per country yearly totals of hydro generation, GDP and POP on NUTS3 levels and per-country load time-series. It should be extracted in the `data` subdirectory (so that all files are in the `data/bundle` subdirectory)
```shell
cd data
wget https://vfs.fias.science/d/0a0ca1e2fb/files/?p=/pypsa-eur-data-bundle.tar.xz
tar xJf pypsa-eur-data-bundle.tar.xz
cd ..
```
2. [pypsa-eur-cuouts.tar.xz](https://vfs.fias.science/d/0a0ca1e2fb/files/?p=/pypsa-eur-cutouts.tar.xz) are spatiotemporal subsets of the European weather data from the [ECMWF ERA5](https://software.ecmwf.int/wiki/display/CKB/ERA5+data+documentation) reanalysis dataset and the [CMSAF SARAH-2](https://wui.cmsaf.eu/safira/action/viewDoiDetails?acronym=SARAH_V002) solar surface radiation dataset for the year 2013. They have been prepared by and are for use with the [atlite](https://github.com/FRESNA/atlite) package. You can either generate them yourself using the `build_cutouts` snakemake rule or extract them directly in the `pypsa-eur` directory (extracting the bundle is recommended, since procuring the source weather data files for atlite is not properly documented at the moment):
```shell
wget https://vfs.fias.science/d/0a0ca1e2fb/files/?p=/pypsa-eur-cutouts.tar.xz
tar xJf pypsa-eur-cutouts.tar.xz
```

# Script overview

See the description in the [Zenodo repository](https://zenodo.org/record/1246851).


# License

The code in PyPSA-Eur is released as free software under the
[GPLv3](http://www.gnu.org/licenses/gpl-3.0.en.html), see
[LICENSE](LICENSE.txt).
