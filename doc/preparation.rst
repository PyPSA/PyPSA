##########################################
Preparing Networks
##########################################

The preparation process of the PyPSA-Eur energy system model consists of a group of ``snakemake`` rules which are briefly outlined and explained in detail in the sections below:

- ``build_shapes`` generates GeoJSON files with shapes of the countries, exclusive economic zones and `NUTS3 <https://en.wikipedia.org/wiki/Nomenclature_of_Territorial_Units_for_Statistics>`_ areas.
- ``build_cutout`` prepares smaller weather data portions from `ERA5 <https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5>`_ for cutout ``europe-2013-era5`` and SARAH for cutout ``europe-2013-sarah``.

With these and the externally extracted ENTSO-E online map topology (``data/entsoegridkit``), it can build a base PyPSA network with the following rules:

- ``base_network`` builds and stores the base network with all buses, HVAC lines and HVDC links, while
- ``build_bus_regions`` determines `Voronoi cells <https://en.wikipedia.org/wiki/Voronoi_diagram>`_ for all substations.

Then the process continues by calculating conventional power plant capacities, potentials, and per-unit availability time series for variable renewable energy carriers and hydro power plants with the following rules:

- ``build_powerplants`` for today's thermal power plant capacities using `powerplantmatching <https://github.com/FRESNA/powerplantmatching>`_ allocating these to the closest substation for each powerplant,
- ``build_renewable_potentials`` for the installation potentials for solar panels, onshore and offshore wind turbines constrained by landuse restrictions and natural protection areas,
- ``build_renewable_profiles`` for the hourly capacity factors in each substation's Voronoi cell for PV, onshore and offshore wind, and
- ``build_hydro_profile`` for the hourly per-unit hydro power availability time series.

The central rule ``add_electricity`` then ties all the different data inputs together into a detailed PyPSA network stored in ``networks/elec.nc``.

.. _shapes:

Build Shapes
=============================

.. automodule:: build_shapes

.. _cutout:

Build Cutout
=============================

.. automodule:: build_cutout

.. _links:

Prepare HVDC Links
=============================

.. automodule:: prepare_links_p_nom

.. _base:

Base Network
=============================

.. automodule:: base_network

.. _busregions:

Build Bus Regions
=============================

.. automodule:: build_bus_regions

.. _natura:

Build Natura Raster
=============================

.. automodule:: build_natura_raster

.. _flh:

Build Country Full Load Hours
=============================

.. automodule:: build_country_flh

.. _powerplants:

Build Power Plants
=============================

.. automodule:: build_powerplants

.. _renewableprofiles:

Build Renewable Profiles
========================

.. automodule:: build_renewable_profiles

.. _hydroprofiles:

Build Hydro Profile
=============================

.. automodule:: build_hydro_profile

.. _electricity:

Add Electricity
=============================

.. automodule:: add_electricity
