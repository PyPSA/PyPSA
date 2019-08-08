##########################################
Preparing Networks
##########################################

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

.. each rule description should have a list of parameters
.. from the config.yaml that affect this rule.

Build Shapes
=============================

.. automodule:: build_shapes

Build Cutout
=============================

.. automodule:: build_cutout

Prepare HVDC Links
=============================

.. automodule:: prepare_links_p_nom

Base Network
=============================

.. automodule:: base_network

Build Bus Regions
=============================

.. automodule:: build_bus_regions

Build Country Full Load Hours
=============================

.. automodule:: build_country_flh

Build Hydro Profile
=============================

.. automodule:: build_hydro_profile

Build Natura Raster
=============================

.. automodule:: build_natura_raster

Build Renewable Profiles
========================

.. automodule:: build_renewable_profiles

Build Power Plants
=============================

.. automodule:: build_powerplants

Add Electricity
=============================

.. automodule:: add_electricity
