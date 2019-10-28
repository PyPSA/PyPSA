![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/pypsa/pypsa-eur?include_prereleases)
[![Build Status](https://travis-ci.org/PyPSA/pypsa-eur.svg?branch=master)](https://travis-ci.org/PyPSA/pypsa-eur)
[![Documentation](https://readthedocs.org/projects/pypsa-eur/badge/?version=latest)](https://pypsa-eur.readthedocs.io/en/latest/?badge=latest)
![GitHub](https://img.shields.io/github/license/pypsa/pypsa-eur)
![Size](https://img.shields.io/github/repo-size/pypsa/pypsa-eur)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.1246852.svg)](https://doi.org/10.5281/zenodo.1246852)
[![Gitter](https://badges.gitter.im/PyPSA/community.svg)](https://gitter.im/PyPSA/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

# PyPSA-Eur: An Open Optimisation Model of the European Transmission System


PyPSA-Eur is an open model dataset of the European power system at the
transmission network level that covers the full ENTSO-E area.
The model is suitable both for operational studies and generation and transmission expansion planning studies.
The continental scope and highly resolved spatial scale enables a proper description of the long-range
smoothing effects for renewable power generation and their varying resource availability.

The model is described in the [documentation](https://pypsa-eur.readthedocs.io)
and in the paper
[PyPSA-Eur: An Open Optimisation Model of the European Transmission
System](https://arxiv.org/abs/1806.01613), 2018,
[arXiv:1806.01613](https://arxiv.org/abs/1806.01613).

![PyPSA-Eur Grid Model](doc/img/base.png)

![PyPSA-Eur Grid Model Simplified](doc/img/elec_s_X.png)

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
- Geographical potentials for wind and solar generators based on land use (CORINE) and excluding nature reserves (Natura2000) are computed with the [vresutils library](https://github.com/FRESNA/vresutils) and the [glaes library](https://github.com/FZJ-IEK3-VSA/glaes).

Already-built versions of the model can be found in the accompanying [Zenodo
repository](https://zenodo.org/record/1246851).
