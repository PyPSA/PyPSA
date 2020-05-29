..
  SPDX-FileCopyrightText: 2019-2020 The PyPSA-Eur Authors

  SPDX-License-Identifier: CC-BY-4.0

##########################################
Limitations
##########################################


While the benefit of an openly available, functional and partially validated
model of the European transmission system is high, many approximations have
been made due to missing data.
The limitations of the dataset are listed below,
both as a warning to the user and as an encouragement to assist in
improving the approximations.

- **Network topology:**
  The grid data is based on a map of the ENTSO-E area that is known
  to contain small distortions to improve readability. Since the exact impedances
  of the lines are unknown, approximations based on line lengths and standard
  line parameters were made that ignore specific conductoring choices for
  particular lines. There is no openly available data on busbar configurations, switch
  locations, transformers or reactive power compensation assets.

- **Distribution networks:**
  Using Voronoi cells to aggregate load and generator data to transmission
  network substations ignores the topology of the underlying distribution network,
  meaning that assets may be connected to the wrong substation.

- **Power Demand:**
  Assumptions
  have been made about the distribution of load in each country proportional to
  population and GDP that may not reflect local circumstances.
  Openly available
  data on load time series may not correspond to the true vertical load and is
  not spatially disaggregated; assuming, as we have done, that the load time series
  shape is the same at each node within each country ignores local differences.

- **Currently installed renewable capacities:** 
  Information on existing wind, solar and small hydro, geothermal, marine and
  biomass power plants are excluded from the dataset because of a lack of data
  availability in many countries. Approximate distributions of wind and solar
  plants in each country can be generated that are proportional to the capacity
  factor at each location.

- **Hydro-electric power plants:**
  The database of hydro-electric power plants does not include plant-specific
  energy storage information, so that blanket values based on country storage
  totals have been used. Inflow time series are based on country-wide approximations,
  ignoring local topography and basin drainage; in principle a full
  hydrological model should be used.

- **International interactions:**
  Border connections and power flows to Russia,
  Belarus, Ukraine, Turkey and Morocco have not been taken into account;
  islands which are not connected to the main European system, such as Malta,
  Crete and Cyprus, are also excluded from the model.
  