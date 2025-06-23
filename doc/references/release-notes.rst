#######################
Release Notes
#######################

.. Upcoming Release
.. ================

.. .. warning:: 
  
..    The features listed below are not released yet, but will be part of the next release! 
..    To use the features already you have to install the ``master`` branch, e.g. 
..    ``pip install git+https://github.com/pypsa/pypsa``.

`v0.35.0 <https://github.com/PyPSA/PyPSA/releases/tag/v0.35.0>`__ (22th June 2025)
=======================================================================================

* New **interactive** plotting library

  * You can now create plots on any PyPSA statistic. (https://github.com/PyPSA/PyPSA/pull/1189)

    * :meth:`n.statistics.energy_balance.iplot() <pypsa.iplot.statistics.plotter.StatisticInteractivePlotter.__call__>` to get the pre defined default plot
    * :meth:`n.statistics.energy_balance.iplot.bar() <pypsa.plot.statistics.plotter.StatisticInteractivePlotter.bar>` to get a bar plot. replace `bar` with `line`, `area`, `map` or `scatter` to get the respective plot.

* The function ``n.statistics.opex()`` now includes all operational cost
  components: marginal costs, quadratic marginal costs, storage costs, spill
  costs, start-up costs, shut-down costs, and stand-by costs. Previously, only
  marginal costs were considered. A new parameter `cost_types` allows selecting
  which cost components to include. (https://github.com/PyPSA/PyPSA/pull/1195)

* New method `n.equals() <pypsa.Network.equals>` to compare two networks for equality. 
  This is similar to the equality operator `==` but allows for more flexibility in the
  comparison which is useful for testing and debugging.
  (https://github.com/PyPSA/PyPSA/pull/1194, https://github.com/PyPSA/PyPSA/pull/1205)

* The components subpackage was further restructured. The known API remains untouched.
  (https://github.com/PyPSA/PyPSA/pull/1223)

* New experimental **NetworkCollection** (https://github.com/PyPSA/PyPSA/pull/1212)

  * You can now create a container for multiple `Network` objects. Use is with
    ``pypsa.NetworkCollection()`` and pass a list of networks. The feature is
    experimental and might change with the next release. Documentation and API
    reference will follow with a stable version of it.

Bug Fixes
--------

* Bugfix: The function ``n.statistics.opex()`` now considers the correct
  snapshot weightings ``n.snapshot_weightings.objective``.
  (https://github.com/PyPSA/PyPSA/pull/1247) 
  
* Fixed unaligned statistics index names when ``groupby=False``
  (https://github.com/PyPSA/PyPSA/pull/1205)

* Fixed interactive area plots in stacked more with `facet_row` and `facet_col`.
  (https://github.com/PyPSA/PyPSA/pull/1212)

* The docstrings of the statistics function are now properly displayed again, ie. the output of `n.statistics.energy_balance?`.
  (https://github.com/PyPSA/PyPSA/pull/1212)

* Fixed various some I/O edge cases for better data preservation during import/export
  (https://github.com/PyPSA/PyPSA/pull/1255, https://github.com/PyPSA/PyPSA/pull/1256, 
  https://github.com/PyPSA/PyPSA/pull/1258)

`v0.34.1 <https://github.com/PyPSA/PyPSA/releases/tag/v0.34.1>`__ (7th April 2025)
=======================================================================================

Bug Fixes
---------

* The static map plots for statistics are fixed, e.g. `n.statistics.energy_balance.map()`. 
  (https://github.com/PyPSA/PyPSA/pull/1201)

* The previous maps module under `pypsa/plot` is now modularized. Instead of a 
  monolithic module, the maps module is now split into several submodules. The
  submodules are: `.maps.common`, `.maps.interactive`, and `.maps.static`.
  (https://github.com/PyPSA/PyPSA/pull/1190)


* Added new single node capacity expansion example in style of model.energy.
  It can be loaded with ``pypsa.examples.model_energy()``.

* Add new example for how to run MGA ('modelling-to-generate-alternatives') optimisation.

* Added demand elasticity example.

`v0.34.0 <https://github.com/PyPSA/PyPSA/releases/tag/v0.34.0>`__ (25th March 2025)
=======================================================================================

Features
--------

* New supported file formats for import and export: Excel

  * Use :meth:`n.import_from_excel <pypsa.io.import_from_excel>` and 
    :meth:`n.export_to_excel <pypsa.io.export_to_excel>` to import and export Networks
    from and to Excel files.
     
  * `openpyxl` and `python-calamine` are required dependencies for this feature, but
    different engines can be passed. By default they are not installed, but can be
    installed via ``pip install pypsa[excel]``.

* New plotting library

  * You can now create plots on any PyPSA statistic. Try them with:

    * :meth:`n.statistics.energy_balance.plot() <pypsa.plot.statistics.plotter.StatisticPlotter.__call__>` to get the pre defined default plot
    * :meth:`n.statistics.energy_balance.plot.bar() <pypsa.plot.statistics.plotter.StatisticPlotter.bar>` to get a bar plot
    * :meth:`n.statistics.energy_balance.plot.line() <pypsa.plot.statistics.plotter.StatisticPlotter.line>` to get a line plot
    * :meth:`n.statistics.energy_balance.plot.area() <pypsa.plot.statistics.plotter.StatisticPlotter.area>` to get a area plot
    * :meth:`n.statistics.energy_balance.plot.map() <pypsa.plot.statistics.plotter.StatisticPlotter.map>` to get a map plot

  * ``n.plot()``  was moved to ``n.plot.map()``

  * ``n.explore()`` was moved to ``n.plot.explore()`` and ``n.iplot()`` was moved to ``n.plot.iplot()``

* Statistics module

  * All statistics functions now interpret the bus_carrier argument as a regular 
    expression (regex), enabling more flexible filtering options. 
    (https://github.com/PyPSA/PyPSA/pull/1155)

  * All statistics functions have a new argument ``carrier`` to filter by carriers.
    (https://github.com/PyPSA/PyPSA/pull/1176)

  * All statistics functions have two new arguments ``drop_zero`` and ``round`` to
    control the output. ``drop_zero`` drops all rows with zero values and ``round``
    rounds the output to the specified number of decimal places. Those settings have been
    used before already via the statistics parameters, but are deprecated now. Use the
    new arguments or the module level settings instead (to set them globally). E.g. 
    ``pypsa.options.params.statistics.nice_names = False``. List all available parameter 
    settings via ``pypsa.options.params.describe()``. 
    (https://github.com/PyPSA/PyPSA/pull/1173)

Minor improvements
------------------

* Ensuring that the created lp/mps file is deterministic by sorting the strongly meshed 
  buses. (https://github.com/PyPSA/PyPSA/pull/1174)

* Added warning for consistent legend circle and semicirle sizes when combining plots 
  on a geographical axis.

* Add new statistic ``n.statistics.system_cost()`` to calculate the total system cost from capital and operational expenditures.

* Added descriptive attribute "location" to Buses. This attribute does not influence the optimisation model but can be used for aggregation in the statistics module.

* Added descriptive attribute "location" to Buses. This attribute does not influence
  the optimisation model but can be used for aggregation in the statistics module.
  (https://github.com/PyPSA/PyPSA/pull/1182)


Bug fixes
---------

* Fixed ``pypsa.plot.add_legend_semicircles()`` circle sizing to be consistent with 
  ``n.plot(bus_sizes=..., bus_split_circles=True)`` argument. 
  (https://github.com/PyPSA/PyPSA/pull/1179)

`v0.33.2 <https://github.com/PyPSA/PyPSA/releases/tag/v0.33.2>`__ (12th March 2025)
=======================================================================================

Bug fixes
---------

* **Regression hotfix**: Fixed a critical bug in statistics functions for 
  multi-investment networks where built years and lifetimes were not being correctly 
  considered. In version ``v0.32.0``, only components active in the first time period were
  being included in statistics calculations. The fix ensures all components are properly
  represented according to their respective built years and lifetimes across all 
  investment periods. This issue was patched in version ``0.33.2``. We also backported the 
  fix to version ``0.32.2``. (https://github.com/PyPSA/PyPSA/pull/1172)

* The expressions function `n.optimize.expressions.capacity` now uses the absolute 
  efficiency to calculate the capacity at link ports, unless a `bus_carrier` is defined
  or `at_port` is set to True. This is in line with the behavior of the statistics 
  functions (`statistics.installed_capacity`, `statistics.optimal_capacity`). 
  Before, the efficiency was allowed to be negative, which lead to inconsistent results.


`v0.33.1 <https://github.com/PyPSA/PyPSA/releases/tag/v0.33.1>`__ (3rd March 2025)
=======================================================================================

Minor improvements
------------------

* Added a ``quotechar`` parameter to :func:`io.import_from_csv_folder` and
  :func:`io.export_to_csv_folder` to handle non-standard field quoting in CSV
  import/export, aligning with :func:`pandas.read_csv` and
  :func:`pandas.to_csv`. (https://github.com/PyPSA/PyPSA/pull/1143)

Bug fixes
---------

* `pypsa[cloudpath]` optional dependency will now only install `cloudpathlib` without 
  extra cloud storage provider client libraries, these will be left to the user to 
  install. (https://github.com/PyPSA/PyPSA/pull/1139)

* :func:`import_from_netcdf` and :func:`import_from_hdf5` now work when a URI is
  passed as a string instead of a CloudPath object.
  (https://github.com/PyPSA/PyPSA/pull/1139)

* Linearized unit commitment with equal startup and shutdown costs.
  (https://github.com/PyPSA/PyPSA/pull/1157)

* Fix pandas dtype warning. (https://github.com/PyPSA/PyPSA/pull/1151)

`v0.33.0 <https://github.com/PyPSA/PyPSA/releases/tag/v0.33.0>`__ (7th February 2025)
=======================================================================================

Features
--------

* New component class structure 
  (https://github.com/PyPSA/PyPSA/pull/1075, https://github.com/PyPSA/PyPSA/pull/1130)

  * Major structural refactoring of how component data is stored and accessed. The new 
    structure adds an extra layer to move all component-specific data from the network 
    class to a new component class.

  * This is an experimental feature, will be developed further and is not yet 
    recommended for general use. More features, documentation and examples will 
    follow. Most users will not notice any changes.

  * The new additional layer makes it easy to add new features. If you wanna play around
    with the new components class, see the 
    `Components class example <https://pypsa.readthedocs.io/en/latest/examples/experimental-components-class.html>`_ 
    in the documentation. You will find an short introduction and some simple examples 
    to show which other features could be added in the future. If you have any ideas, 
    wishes, feedback or suggestions, please let us know via the 
    `issue tracker <https://www.github.com/PyPSA/PyPSA/issues>`_.

* Breaking: Deprecation of custom components (https://github.com/PyPSA/PyPSA/pull/1130)

  * This version of PyPSA deprecates custom components. While we don't see many use 
    cases for them, they might be added in an improved way in future again. For a 
    potential reimplementation we would be happy to hear your use case and 
    requirements via the `issue tracker <https://www.github.com/PyPSA/PyPSA/issues>`_.
  
  * If you don't know what this is or have never used the ``override_components``
    and ``override_component_attrs`` arguments during Network initialisation, you can
    safely ignore this deprecation.

* Breaking: Behavior of ``n.components``

  * Iterating over `n.components` now yields the values instead of keys. Use 
    `n.components.keys()` to keep iterating over keys.

  * Checking if a component is in `n.components` using the 'in' operator is deprecated. 
    With the deprecation of custom components keys in `n.components` also ever change.
            
* PyPSA `0.33` provides support for the recent Python 3.13 release and drops support 
  for Python 3.9. While Python 3.9 still gets security updates until October 2025,
  core dependencies of PyPSA are dropping support for Python 3.9 (e.g. `numpy`) and
  active support is only provided for the most recent versions 
  (see `endoflife.date <https://endoflife.date/python>`_). It is recommended to upgrade 
  to the latest Python version if possible. Note that there might be some issues with
  Windows and Python 3.13, which are not yet resolved. 
  (https://github.com/PyPSA/PyPSA/pull/1099)

* Added PyPSA options architecture via :meth:`pypsa.get_option`, :meth:`pypsa.set_option`, 
  :meth:`pypsa.describe_options` and :meth:`pypsa.option_context`.
  This allows to set and get global options for PyPSA and
  mimics the options setting behavior of pandas. Currently there are not many options
  available, but this will be extended in future. 
  (https://github.com/PyPSA/PyPSA/pull/1134)

* New network attributes :meth:`n.timesteps <pypsa.networks.Network.timesteps>`, 
  :meth:`n.periods <pypsa.networks.Network.periods>` and 
  :meth:`n.has_periods <pypsa.networks.Network.has_periods>` to simplified level access
  of the snapshots dimension. (https://github.com/PyPSA/PyPSA/pull/1113)

* Consistency checks can now be run with the parameter ``strict``, which will raise 
  them as ``ConsistenyError``. Pass checks which should be strict in 
  :meth:`n.consistency_check <pypsa.consistency.consistency_check>` as e.g.
  ``strict=['unknown_buses']``. :meth:`n.optimize <pypsa.optimization.optimize.optimize>`
  will run some strict checks by default now. (https://github.com/PyPSA/PyPSA/pull/1120, 
  https://github.com/PyPSA/PyPSA/pull/1112)

* New example in the documentation showing how to implement reserve power constraints.
  (https://github.com/PyPSA/PyPSA/pull/1133)

* Doctests are now run with the unit tests. They allow to test the documentation 
  examples, which will improve the quality of docstrings and documentation in future 
  releases. (https://github.com/PyPSA/PyPSA/pull/1114)
  
Bug fixes
---------

* The parameter threshold in function get_strong_meshed_buses was not considered
  in the function it self. A kwargs check has been added for providing a own threshold.
  E.g., get_strongly_meshed_buses (network, threshold=10)


`v0.32.2 <https://github.com/PyPSA/PyPSA/releases/tag/v0.32.2>`__ (12th March 2025)
=======================================================================================

Bug fixes
---------

* Backported from version ``v0.33.2``: Fixed a critical bug in statistics functions for 
  multi-investment networks where built years and lifetimes were not being correctly 
  considered. In version ``v0.32.0``, only components active in the first time period were
  being included in statistics calculations. The fix ensures all components are properly
  represented according to their respective built years and lifetimes across all 
  investment periods. (https://github.com/PyPSA/PyPSA/pull/1172)

`v0.32.1 <https://github.com/PyPSA/PyPSA/releases/tag/v0.32.1>`__ (23th Januarary 2025)
=======================================================================================

Bug fixes
---------

* The expression module now correctly includes the "Load" component in the
  energy balance calculation. Before the fix, the "Load" component was not
  considered. (https://github.com/PyPSA/PyPSA/pull/1110)

* The optimize/expression module now correctly assigns contributions from branch 
  components in the `withdrawal` and `supply` functions. Before, there was a wrong 
  multiplication by -1 for branch components. (https://github.com/PyPSA/PyPSA/pull/1123)

`v0.32.0 <https://github.com/PyPSA/PyPSA/releases/tag/v0.32.0>`__ (5th December 2024)
=======================================================================================

Features
--------

* Improvements to groupers in the statistics module 
  (https://github.com/PyPSA/PyPSA/pull/1093, https://github.com/PyPSA/PyPSA/pull/1078)

  * The ``groupby`` argument now accepts keys to allow for more granular and flexible 
    grouping.
    For example,
    :meth:`n.statistics.energy_balance(groupby=["bus_carrier", "carrier"]) <pypsa.statistics.StatisticsAccessor.energy_balance>`
    groups the energy balance by bus carrier and carrier.

    * Build in groupers include: 

      * :meth:`pypsa.statistics.groupers.carrier <pypsa.statistics.grouping.Groupers.carrier>`
      * :meth:`pypsa.statistics.groupers.bus_carrier <pypsa.statistics.grouping.Groupers.bus_carrier>`
      * :meth:`pypsa.statistics.groupers.name <pypsa.statistics.grouping.Groupers.name>`
      * :meth:`pypsa.statistics.groupers.bus <pypsa.statistics.grouping.Groupers.bus>`
      * :meth:`pypsa.statistics.groupers.country <pypsa.statistics.grouping.Groupers.country>`
      * :meth:`pypsa.statistics.groupers.unit <pypsa.statistics.grouping.Groupers.unit>`
      * A list of registered groupers can be accessed via
        :meth:`pypsa.statistics.groupers.list_groupers <pypsa.statistics.grouping.Groupers.list_groupers>`
  
  * Custom groupers can be registered on module level via
    :meth:`pypsa.statistics.groupers.add_grouper <pypsa.statistics.grouping.Groupers.add_grouper>`.
    The key will be used as identifier in the ``groupby`` argument. Check the API reference
    for more information.

  * Accessing default groupers was moved to module level and an improved API was 
    introduced. ``n.statistics.get_carrier`` can now be accessed as 
    :meth:`pypsa.statistics.groupers.carrier <pypsa.statistics.grouping.Groupers.carrier>`
    and a combination of groupers can be accessed as 
    :meth:`pypsa.statistics.groupers['bus', 'carrier'] <pypsa.statistics.grouping.Groupers.__call__>`
    instead of ``n.statistics.groupers.get_bus_and_carrier``.

* A new module ``pypsa.optimize.expressions`` was added. It contains functions to quickly 
  create expressions for the optimization model. The behavior of the functions is 
  mirroring the behavior of the ``statistics``` module and allows for similar complexity 
  in grouping and filtering. Use it with e.g. 
  :meth:`n.optimize.expressions.energy_balance() <pypsa.Network.expressions.energy_balance>`.
  (https://github.com/PyPSA/PyPSA/pull/1044)

* ``pytables`` is now an optional dependency for using the HDF5 format. Install 
  it via ``pip install pypsa[hdf5]``. Otherwise it is not installed by default 
  anymore. (https://github.com/PyPSA/PyPSA/pull/1100)

`v0.31.2 <https://github.com/PyPSA/PyPSA/releases/tag/v0.31.2>`__ (27th November 2024)
=======================================================================================

Bug fixes
---------

* The constraint to account for ``e_sum_max``/ ``e_sum_min`` is now skipped if not applied 
  to any asset 
  (https://github.com/PyPSA/PyPSA/pull/1069, https://github.com/PyPSA/PyPSA/pull/1074)


`v0.31.1 <https://github.com/PyPSA/PyPSA/releases/tag/v0.31.1>`__ (1st November 2024)
======================================================================================

Bug fixes
---------

* Abolishing ``min_units`` in the post discretization. If the maximum capacity of a 
  component is smaller than the specified unit size, the maximum capacity is built as 
  soon as the threshold is passed (https://github.com/PyPSA/PyPSA/pull/1052)

* Less verbose logging when using :meth:`n.add <pypsa.Network.add>` 
  (https://github.com/PyPSA/PyPSA/pull/1067)

`v0.31.0 <https://github.com/PyPSA/PyPSA/releases/tag/v0.31.0>`__ (1st October 2024)
=====================================================================================

Features
--------

* New ``active`` attribute (https://github.com/PyPSA/PyPSA/pull/1038)

  - A new attribute for one-port and branch components `active` was added. If set to 
    true (default), the asset is considered active for all functionality, including 
    optimization and power flow calculation. If set to false, the asset is considered 
    inactive and is excluded from the optimization, power flow and statistics modules. 

  - The active attribute can be thought of as a global filter on the components. When 
    running a multi-horizon optimization, the active attribute is considered a global 
    condition for each horizon. Then assets are considered active only if `active` is 
    true and the investment period falls within the lifetime of the asset.

* New attributes for the :ref:`generator <component-generator>` component 
  (https://github.com/PyPSA/PyPSA/pull/1047)
  
  - ``e_sum_min`` and ``e_sum_max`` add a new constraint and allow to set the minimum 
    and maximum total energy that can be generated by the generator over one 
    optimization horizon. 

* New :meth:`n.add <pypsa.Network.add>` method (https://github.com/PyPSA/PyPSA/pull/896)
  
  - :meth:`n.add <pypsa.Network.add>` now handles the addition of a single or multiple
    components, has more robust index alignment checks allows to overwrite existing
    components using the new argument ``overwrite``. Because of the more strict 
    alignment checks, this might be a **breaking change** for some users.
  
  - Therefore the methods :meth:`n.madd <pypsa.Network.madd>` and 
    :meth:`n.mremove <pypsa.Network.mremove>` are now deprecated and will point to
    their generalised counterparts.

* New function :meth:`n.optimize_and_run_non_linear_powerflow <pypsa.optimization.optimize.OptimizationAccessor.optimize_and_run_non_linear_powerflow>`
  was added to the set of abstract optimize functions. This function optimizes the 
  network and runs a non-linear power flow calculation afterwards. (https://github.com/PyPSA/PyPSA/pull/1038)

* API and structural changes:

  - The :class:`Component <pypsa.definitions.components.Component>` object is now a refactored 
    stand-alone class. This is ongoing work and will change further in future 
    releases. (https://github.com/PyPSA/PyPSA/pull/1038)
  - The :class:`pypsa.SubNetwork` class has new methods `df`, `pnl`, `component` 
    to ease the access of component data for a subnetwork. Use it with e.g.
    `subnetwork.df("Generator")` and alike. (https://github.com/PyPSA/PyPSA/pull/1038)
  - :meth:`n.df <pypsa.Network.df>` and :meth:`n.pnl <pypsa.Network.pnl>` 
    have been renamed to :meth:`n.static <pypsa.Network.static>` and 
    :meth:`n.dynamic <pypsa.Network.dynamic>`. But `n.df` and `n.pnl` are still available 
    and can be used as aliases without any deprecation warning for now. (https://github.com/PyPSA/PyPSA/pull/1028)

`v0.30.3 <https://github.com/PyPSA/PyPSA/releases/tag/v0.30.3>`__ (24th September 2024)
========================================================================================

* Bugfix in the post discretization for ``Links`` with a maximum capacity.
  Furthermore, giving the option to build out only multiples of the specified unit_size
  or allowing to use the full maximum capacity. (https://github.com/PyPSA/PyPSA/pull/1039)

`v0.30.2 <https://github.com/PyPSA/PyPSA/releases/tag/v0.30.2>`__ (11th September 2024)
========================================================================================

* Bugfix in operational limit global constraints, which now directly uses the
  carrier of the ``Store`` rather than the carrier of the bus it is attached to.
  (https://github.com/PyPSA/PyPSA/pull/1029)

`v0.30.1 <https://github.com/PyPSA/PyPSA/releases/tag/v0.30.1>`__ (9th September 2024)
=======================================================================================

* Added option for importing and exporting CSV, netCDF and HDF5 files in cloud
  object storage. This requires the installation of the optional dependency
  ``cloudpathlib``, e.g. via ``pip install pypsa[cloudpath]``.

* Bugfix of ``n.plot()`` when single buses have no coordinates.

`v0.30.0 <https://github.com/PyPSA/PyPSA/releases/tag/v0.30.0>`__ (30th August 2024)
=====================================================================================

* Added ``n.explore()`` function based on ``folium`` and ``geopandas`` to
  interactively explore networks. (https://github.com/PyPSA/PyPSA/pull/1009)

* Added new ``spill_cost`` input parameter for storage units which penalizes
  spilling excess energy. (https://github.com/PyPSA/PyPSA/pull/1012)

* Added new ``marginal_cost_storage`` input parameter for stores and storage
  units to represent the cost of storing energy in currency/MWh/h.
  (https://github.com/PyPSA/PyPSA/pull/603)

* Added type annotations to all functions. (https://github.com/PyPSA/PyPSA/pull/1010)

* Updated documentation. (https://github.com/PyPSA/PyPSA/pull/1004)

`v0.29.0 <https://github.com/PyPSA/PyPSA/releases/tag/v0.29.0>`__ (31st July 2024)
===================================================================================

* Removed ``n.lopf()`` pyomo-based and nomopyomo-based optimisation modules. Use
  linopy-based optimization with ``n.optimize()`` instead.
  (https://github.com/PyPSA/PyPSA/pull/884)

* HiGHS becomes the new default solver for ``n.optimize()``.
  (https://github.com/PyPSA/PyPSA/pull/884)

* Changes to the ``statistics`` module:

  - The statistics functions ``n.statistics.capex()``,
    ``n.statistics.installed_capex()``, and ``expanded_capex`` now have an
    optional ``cost_attribute`` argument, which defaults to `capital_cost`. The
    default behavior of the functions is not changed.
    (https://github.com/PyPSA/PyPSA/pull/989)

  - The functions ``n.statistics.optimal_capacity()`` and
    ``n.statistics.expanded_capacity()`` now return positive and negative
    capacity values if a ``bus_carrier`` is selected. Positive values correspond
    to production capacities, negative values to consumption capacities.
    (https://github.com/PyPSA/PyPSA/pull/885)

  - The statistics module now supports the ``nice_name`` argument for bus
    carriers. Previously, nice names were only supported for components
    carriers. (https://github.com/PyPSA/PyPSA/pull/991)

  - The statistics module now features functionality to set global style
    parameters (e.g. ``nice_names``, ``drop_zero`` and ``round``) which is then
    applied to all statistics methods without the need to set them individually.
    To set parameters one can run
    ``n.statistics.set_parameters(nice_names=False, round=2)`` and to view
    current parameters setting ``n.statistics.parameters``.
    (https://github.com/PyPSA/PyPSA/pull/886)

* Changes to the ``clustering`` module:

  - Add attribute-based exemptions for clustering lines and links. With the
    argument ``custom_line_groupers`` in the function ``aggregatelines()`` one
    can specify additional columns besides ``bus0`` and ``bus1`` to consider as
    unique criteria for clustering. This is useful, for example, to avoid the
    aggregation of lines/links with different ``build_year`` or ``carrier``.
    (https://github.com/PyPSA/PyPSA/pull/982)

* Changes to the ``plot`` module:

  - Add option to add semicircle legends by running
    ``pypsa.plot.add_legend_semicircle(ax, sizes=[1000/scaling_factor],
    labels=["1 GWh"])``. (https://github.com/PyPSA/PyPSA/pull/986)

  - Add functionality to provide list of colors in ``add_legend_lines()``.
    (https://github.com/PyPSA/PyPSA/pull/902)

* Bugfixes:
  
  - The security-constrained optimization via
    ``n.optimize.optimize_security_constrained()`` was fixed to correctly handle
    multiple subnetworks. (https://github.com/PyPSA/PyPSA/pull/946)

  - The global constraint on the total transmission costs now includes the
    weight of the investment periods and persistence of investment costs of
    active assets in multi-horizon optimisations.

  - Retain investment periods and weightings when clustering networks.
    (https://github.com/PyPSA/PyPSA/pull/891)

  - Removed performance regression of ``statistics`` module.
    (https://github.com/PyPSA/PyPSA/pull/990)

  - When adding bus ports on the fly with `add` methods, the dtype of the
    freshly created column is now fixed to `string`. (https://github.com/PyPSA/PyPSA/pull/893)

  - Using timezone information in `n.snapshots` raises an error now, since it
    leads to issues with `numpy`/ `xarray`. (https://github.com/PyPSA/PyPSA/pull/976)

* Improvements to consistency checks and model debugging:

  - When adding components with bus ports greater than 1, e.g. `bus2`, pypsa
    checks if the bus exists and prints a warning if it does not.
    (https://github.com/PyPSA/PyPSA/pull/893)

  - Also check for missing values of default attributes in the
    `n.consistency_check()` function. (https://github.com/PyPSA/PyPSA/pull/903)

  - Restructure ``n.consistency_check()``.
    (https://github.com/PyPSA/PyPSA/pull/903,https://github.com/PyPSA/PyPSA/pull/918, https://github.com/PyPSA/PyPSA/pull/920)

  - Add option `n.optimize(compute_infeasibilities=True)` to compute Irreducible
    Inconsistent Subset (IIS) in case an infeasibility was encountered and Gurobi
    is installed. (https://github.com/PyPSA/PyPSA/pull/978)

  - Improved error messages. (https://github.com/PyPSA/PyPSA/pull/897)

* Add functionality to compare two networks for equality via equality operator
  (``==``). (https://github.com/PyPSA/PyPSA/pull/924)

* Add single-node electricity-only and sector-coupled capacity expansion
  example. (https://github.com/PyPSA/PyPSA/pull/904)

* Added new line type "Al/St 490/64 4-bundle 380.0".
  (https://github.com/PyPSA/PyPSA/pull/887)

* Use ``ruff``. (https://github.com/PyPSA/PyPSA/pull/900,
  https://github.com/PyPSA/PyPSA/pull/901)

* Improve CI and auto-release process. (https://github.com/PyPSA/PyPSA/pull/907,
  https://github.com/PyPSA/PyPSA/pull/921)

* Restructured API reference. (https://github.com/PyPSA/PyPSA/pull/960)

* Compatibility with ``numpy>=2.0``. (https://github.com/PyPSA/PyPSA/pull/932)

`v0.28.0 <https://github.com/PyPSA/PyPSA/releases/tag/v0.28.0>`__ (8th May 2024)
=================================================================================

* When using iterative optimisation functionality
  ``n.optimize_transmission_expansion_iteratively()``, add option to discretize
  optimised line and link capacities in the final iteration based on new keyword
  arguments ``line_unit_size``, ``link_unit_size``, ``line_threshold`` and
  ``link_threshold``. This allows to round the optimised capacities to a
  multiple of the unit size based on the threshold.
  (https://github.com/PyPSA/PyPSA/pull/871)

* A new function ``n.merge()`` was added allowing the components and
  time-dependent data of one network to be added to another network. The
  function is also available via ``n + m`` with default settings. The function
  requires disjunct component indices and identical snapshots and snapshot
  weightings. (https://github.com/PyPSA/PyPSA/pull/783)

* New features in the statistics module (https://github.com/PyPSA/PyPSA/pull/860):

  - The statistics module introduces a new keyword argument ``at_port`` to all
    functions. This allows considering the port of a component when calculating
    statistics. Depending on the function, the default of ``at_port`` is set to
    ``True`` or ``False``, for example for the dispatch all ports are
    considered.

  - The statistics module now supports an optional ``port`` argument in
    ``groupby`` functions. This allows to group statistics while considering the
    port of a component.

  - The ``statistics.revenue`` function introduces a new keyword argument
    ``kind`` to optionally calculate the revenue based on the ``input``
    commodity or the ``output`` commodity of a component.

  - The ``statistics.energy_balance`` function introduces a new keyword argument
    ``kind`` to optionally calculate the ``supply`` and ``withdrawal`` of a
    component.

  - Deprecation warnings are added to the statistics module for the
    functionalities that will be removed in the next major release.

* Updated ``environment_doc.yml`` to include the latest required ``pip``
  dependencies for the documentation environment. (https://github.com/PyPSA/PyPSA/pull/862)

* Bugfix: calling ``n.create_model()`` or ``n.optimize()`` when a global
  operational limit is defined will no longer set the carrier attribute of
  stores to the carrier of the bus they are attached to.
  (https://github.com/PyPSA/PyPSA/pull/880)

* Added warning to ``plot.py`` with instructions to handle the case where the
  ``requests`` dependency is missing. (https://github.com/PyPSA/PyPSA/pull/882)

* Bugfix: calling ``n.optimize.*`` functions (e.g. ``n.optimize.optimize_mga``)
  now correctly returns each functions return values. (https://github.com/PyPSA/PyPSA/pull/871)


`v0.27.1 <https://github.com/PyPSA/PyPSA/releases/tag/v0.27.1>`__ (22nd March 2024)
====================================================================================

* Fixed sometimes-faulty total budget calculation for single-horizon MGA optimisations.

* Fixed assignment of active assets in multi-horizon optimisation with ``n.optimize``.

* Fixed setting of investment periods when copying a multi-horizon network.

* Always use name and mask keys in variable and constraint assignment to protect against future changes in argument order.

* Rewrite function ``get_switchable_as_dense`` so that it consumes less memory when calling it with large dataframes.

* Fix of the capex description in the attribute CSV files.

`v0.27.0 <https://github.com/PyPSA/PyPSA/releases/tag/v0.27.0>`__ (18th February 2024)
=======================================================================================

* Bugfix: If plotting a network map with split buses
  (``n.plot(bus_split_circles=True)``), the bus sizes are now scaled by factor 2
  to account for the fact that the bus sizes are split into half circles. This
  makes the area scaling of the buses consistent with the area of non-split
  buses.

* The global constraint ``define_tech_capacity_expansion_limit`` now also takes
  branch components into account. If defined per bus, the ``bus0`` of the branch
  is considered as a reference bus.

* Bugfixes in building of global constraints in multi-horizon optimisations.

* Fixed total budget calculation for MGA on multi-horizon optimisations.

* The ``extra_functionality`` argument is now also supported in ``solve_model``
  accessor.

* ``optimize_mga`` now returns the solver termination status and condition.

* The deprecated functions ``_make_consense``, ``aggregategenerators``,
  ``get_buses_linemap_and_lines`` and ``get_clustering_from_busmap`` were
  removed.

* The minimum ``networkx`` version was bumped from ``1.10`` to ``2``.

* ``pyomo`` is no longer supported for Python 3.12 or higher.


`v0.26.3 <https://github.com/PyPSA/PyPSA/releases/tag/v0.26.3>`__ (25th January 2024)
======================================================================================

* Bugfix: With line transmission losses there was a sign error in the
  calculation of the line capacity constraints.

* Approximated transmission losses of lines are now stored after optimisation as
  the difference between ``n.lines_t.p0`` and ``n.lines_t.p1`` so that they
  appear in the energy balance (e.g. ``n.statistics.energy_balance()``) and when
  calculating losses with ``n.lines_t.p0 + n.lines_t.p1``.

`v0.26.2 <https://github.com/PyPSA/PyPSA/releases/tag/v0.26.2>`__ (31st December 2023)
=======================================================================================

* Bugfix in the definition of spillage variables for storage units. Previously,
  the spillage variable creation was skipped in some cases due to a wrong
  condition check even though there was a positive inflow.

`v0.26.1 <https://github.com/PyPSA/PyPSA/releases/tag/v0.26.1>`__ (29th December 2023)
=======================================================================================

* The output attribute ``n_mod`` introduced in the previous version was removed
  since it contains duplicate information. Calculate the number of expanded
  modules with ``p_nom_opt / p_nom_mod`` instead.

* Bugfix in MGA function to correctly parse the ``sense`` keyword argument.

* Fix strict type compatibility issues with ``pandas>=2.1`` causing problems for
  clustering.

* Removed ``numexpr`` version constraint.

`v0.26.0 <https://github.com/PyPSA/PyPSA/releases/tag/v0.26.0>`__ (4th December 2023)
======================================================================================

**New Features**

* The ``Network`` class has a **new component for geometric shapes** under
  ``n.shapes``. It consists of a ``geopandas`` dataframe which can be used to
  store network related geographical data (for plotting, calculating potentials,
  etc). The dataframe has the columns `geometry`, `component`, `idx` and `type`.
  The columns `component`, `idx` and `type` do not require specific values, but
  allow for storing information about which components the shapes belong to. The
  coordinate reference system (CRS) of the shapes can be accessed and set via a
  new attribute ``n.crs``. For a transition period, the attribute ``n.srid``,
  which independently refers to the projection of the bus coordinates, is kept.

* Improvements to the network **statistics module**:

  * The statistics module now supports the consideration of multi-port links. An
    additional argument `bus_carrier` was added to the statistics functions to
    select the components that are attached to buses of a certain carrier.

  * The statistics module now supports the consideration of multiple investment
    periods. As soon as ``n.snapshots`` is a MultiIndex, the network statistics
    are calculated separately for each investment period.

  * A new function ``transmission`` was added to the statistics accessor. This
    function considers all lines and links that connect buses of the same carrier.

  * The statistics functions now support the selection of single components in
    the ``comps`` argument.

* The plotting function ``n.plot()`` now supports **plotting of only a subset of
  network components** by allowing that arguments like ``bus_sizes``,
  ``link_widths`` or ``link_colors`` do no longer require to contain the full
  set of indices of a component.

* Add option to specify **time-varying ramp rates** for generators and links
  (``ramp_limit_up`` and ``ramp_limit_down``, e.g. under
  ``n.links_t.ramp_limit_up``).

* Added attributes ``p_nom_mod``, ``e_nom_mod``, and ``s_nom_mod`` to components
  to consider capacity modularity. When this attribute is non-zero and the
  component is extendable, the component's capacity can only be extended in
  multiples of the capacity modularity. The optimal number of components is
  stored as ``n_mod`` (such that ``p_nom_mod * n_mod == p_nom_opt``). The
  default is kept such that extendable components can be expanded continuously.

**Bugfixes and Compatibiliity**

* Bugfix: In rolling horizon optimisation with unit commitment constraints, the
  generator status of the previously optimized time step is now considered.

* Bugfix: Allow optimising the network for just subset of investment periods by
  using ``n.optimize(multi_investment_periods=True, snapshots=...)``.

* Bugfix: The function ``n.import_from_netcdf()`` failed when trying to import
  data from an ``xarray`` object.

* Bugfix: Fix global constraints for primary energy and transmission volume
  limits for networks with multiple investment periods.

* Bugfix: Fix stand-by-costs optimization for latest ``linopy`` version.

* Resolve performance regression for multi-decade optimisation in highly meshed
  networks.

* Compatibility with ``pandas==2.1``.

* Added Python 3.12 to CI and supported Python versions.


`v0.25.2 <https://github.com/PyPSA/PyPSA/releases/tag/v0.25.2>`__ (30th September 2023)
========================================================================================

* Add option to enable or disable nice carrier name in the statistics module,
  e.g. ``n.statistics(nice_name=False)``.

* Add example in documentation for the statistics module.

* Add example for stochastic optimization with PyPSA to the documentation.

* Extended documentation for multi-decade optimization.

* Bugfix: Use of ``nice_names`` keyword argument in
  ``n.statistics.energy_balance()``.

* Bugfix: Correctly handle ``p_nom`` or ``p_nom_opt`` in power flow distributed
  slack.

* Bugfix: After the optimization the right-hand side and sign of global
  constraints were previously overwritten by altered values.

* Bugfix: In netCDF export, typecasting to float32 after setting the compression
  encoding led to ignored compression encodings.

* Bugfix: Handle solver options for CBC and GLPK for ``n.lopf(pyomo=False)``.

* Bugfix: Handle cases with multi-decade optimisation, activated transmission
  limit and an empty list of lines or DC links.

`v0.25.1 <https://github.com/PyPSA/PyPSA/releases/tag/v0.25.1>`__ (27th July 2023)
===================================================================================

**New Features**

* The function ``get_clustering_from_busmap`` has a new argument
  ``line_strategies``.

* The ``n.optimize()`` function gets a new keyword argument
  ``assign_all_duals=False`` which controls whether all dual values or only
  those that already have a designated place in the network are assigned.
  (https://github.com/PyPSA/PyPSA/pull/635)

**Changes**

* The function ``get_buses_linemap_and_lines`` was deprecated, in favor of
  direct use of ``aggregatebuses`` and ``aggregate_lines``.

* Improve logging printout for rolling horizon optimization.
  (https://github.com/PyPSA/PyPSA/pull/697,
  https://github.com/PyPSA/PyPSA/pull/699)

* The CI environment handling was migrated to ``micromamba``
  (https://github.com/PyPSA/PyPSA/pull/688).

**Bugfixes**

* The aggregation functions in the clustering module were adjusted to correctly
  handle infinity values (see https://github.com/pandas-dev/pandas/issues/54161
  for more details). (https://github.com/PyPSA/PyPSA/pull/684)

* The unit commitment formulation with a rolling horizon horizon was fixed in
  case of non-committable and committable generators with ramp limits.
  (https://github.com/PyPSA/PyPSA/pull/686)

* The clustering functionality was fixed in case of passing a subset of carriers
  that should be aggregated. (https://github.com/PyPSA/PyPSA/pull/696)

* When clustering, allow safe clustering of component attributes which are both
  static and dynamic. (https://github.com/PyPSA/PyPSA/pull/700)

* When assigning a new user-defined variable to the underlying optimization
  model, the assignment of the solution resulted in an error if the variable
  name did not match the pattern ``{Component}-{Varname}``. This has been fixed
  by ignoring variables that do not match the pattern during solution
  assignment. (https://github.com/PyPSA/PyPSA/pull/693)

* Multilinks are now also handled automatically when importing a network from
  file. (https://github.com/PyPSA/PyPSA/pull/702)

* Multilink default efficiencies are always set to 1.0.
  (https://github.com/PyPSA/PyPSA/pull/701)

* For linearized unit commitment relaxation, some tightening additional
  constraints are only valid if start-up and shut-down costs are equal. These
  constraints are now skipped if this is not the case and a warning message is
  printed. (https://github.com/PyPSA/PyPSA/pull/690)

* Fix division in capacity factor calculation in statistics module when not
  aggregating in the time dimension. (https://github.com/PyPSA/PyPSA/pull/687)


`v0.25.0 <https://github.com/PyPSA/PyPSA/releases/tag/v0.25.0>`__ (13th July 2023)
===================================================================================

**New Features**

* **Stand-by costs:** PyPSA now supports stand-by cost terms. A new column
  ``stand_by_cost`` was added to generators and links. The stand-by cost is
  added to the objective function when calling ``n.optimize()``.
  (https://github.com/PyPSA/PyPSA/pull/659)

* **Rolling horizon function:** The ``n.optimize`` accessor now provides
  functionality for rolling horizon optimisation using
  ``n.optimize.optimize_with_rolling_horizon()`` which splits whole optimization
  of the whole time span into multiple subproblems which are solved
  consecutively. This is useful for operational optimizations with a high
  spatial resolution. (https://github.com/PyPSA/PyPSA/pull/668)

* **Modelling-to-generate-alternatives (MGA) function** The ``n.optimize``
  accessor now provides functionality for running
  modelling-to-generate-alternatives (MGA) on previously solved networks using
  ``n.optimize.optimize_mga(slack=..., weights=...)``. This is useful for
  exploring the near-optimal feasible space of the network.
  (https://github.com/PyPSA/PyPSA/pull/672)

**Changes**

* **Multilinks by default:** Links with multiple inputs/outputs are now
  supported by default. The Link component attributes are automatically extended
  if a link with ``bus2``, ``bus3``, etc. are added to the network. Overriding
  component attributes at network initialisation is no longer required.
  (https://github.com/PyPSA/PyPSA/pull/669)

* **Spatial clustering refactored:** The spatial clustering module was
  refactored. The changes lead to performance improvements and a more consistent
  clustering API. (https://github.com/PyPSA/PyPSA/pull/673)

  * The network object has a new accessor ``cluster`` which allows accessing
    clustering routines from the network itself. For example,
    ``n.cluster.cluster_spatially_by_kmeans`` returns a spatially clustered
    version of the network.

  * The default clustering strategies were refined. Per default, columns like
    ``efficiency`` and ``p_max_pu`` are now aggregated by the capacity weighted
    mean.

  * The clustering module now applies the custom strategies to time-dependant
    data.

  * The function ``pypsa.clustering.spatial.get_clustering_from_busmap`` and
    ``pypsa.clustering.spatial.aggregategenerators`` now allows the passing of a
    list of buses for which aggregation of all carriers is desired. Generation
    from a carrier at a bus is aggregated now if: It is either in the passed
    list of aggregated carriers, or in the list of aggregated buses.

  * Take generator strategies for time-series into account. Before, time-series
    would always be aggregated by summing.
    (https://github.com/PyPSA/PyPSA/pull/670)

  * The deprecated ``networkclustering`` module was removed.
    (https://github.com/PyPSA/PyPSA/pull/675)

* A new function `get_country_and_carrier` was added to the statistics module in
  order to group statistics by country and carrier.
  (https://github.com/PyPSA/PyPSA/pull/678)

* NetCDF file compression is now disabled by default when exporting networks.
  (https://github.com/PyPSA/PyPSA/pull/679)

**Breaking Changes**

* The ``Clustering`` class no longer contains a positive and negative linemap.

* Outdated examples were removed. (https://github.com/PyPSA/PyPSA/pull/674)

**Bugfixes**

* In the statistics module, the calculation of operational costs of storage
  units was corrected. (https://github.com/PyPSA/PyPSA/pull/671)


`v0.24.0 <https://github.com/PyPSA/PyPSA/releases/tag/v0.24.0>`__ (27th June 2023)
===================================================================================

* PyPSA now supports quadratic marginal cost terms. A new column
  `marginal_cost_quadratic` was added to generators, links, stores and storage
  units. The quadratic marginal cost is added to the objective function when
  calling ``n.optimize()``. This requires a solver that is able to solve quadratic problems, for instance,
  HiGHS, Gurobi, Xpress, or CPLEX.
* The statistics function now allows calculating energy balances
  ``n.statistics.energy_balance()`` and dispatch ``n.statistics.dispatch()``, as
  well as time series (e.g. ``n.statistics.curtailment(aggregate_time=False)``).
  The energy balance can be configured to yield energy balance time series for
  each bus.
* The statistics function ``n.statistics()`` now also supports the calculation
  of the market values of components.
* The function ``n.set_snapshots()`` now takes two optional keyword arguments; ``default_snapshot_weightings``
  to change the default snapshot weightings, and ``weightings_from_timedelta``
  to compute the weights if snapshots are of type ``pd.DatetimeIndex``.
* The function ``n.lopf()`` is deprecated in favour of the linopy-based
  implementation ``n.optimize()`` and will be removed in PyPSA v1.0. We will
  have a generous transition period, but please start migrating your
  ``extra_functionality`` functions, e.g. by following our `migration guide
  <https://pypsa.readthedocs.io/en/latest/examples/optimization-with-linopy-migrate-extra-functionalities.html>`_.
* The module ``pypsa.networkclustering`` was moved to
  ``pypsa.clustering.spatial``. The module ``pypsa.networkclustering`` is now
  deprecated but all functionality will continue to be accessible until PyPSA v0.25.
* Bug fix in linearized unit commitment implementation correcting sign.
* The minimum required version of ``linopy`` is now ``0.2.1``.
* Dropped support for Python 3.8. The minimum required version of Python is now 3.9.


`v0.23.0 <https://github.com/PyPSA/PyPSA/releases/tag/v0.23.0>`__ (10th May 2023)
==================================================================================

* Transmission losses can now be represented during optimisation with
  ``n.optimize()`` or ``n.lopf()`` using a piecewise linear approximation of the
  loss parabola as presented in `this paper
  <https://doi.org/10.1016/j.apenergy.2022.118859>`_. The number of segments can
  be chosen with the argument ``n.optimize(transmission_losses=3)``. The default
  remains that transmission losses are neglected with
  ``n.optimize(transmission_losses=0)``, and analogously for
  ``n.lopf(pyomo=True)`` and ``n.lopf(pyomo=False)``. [`#462
  <https://github.com/PyPSA/PyPSA/pull/462>`_]

* Efficiencies and standing losses of stores, storage units and generators can
  now be specified as time-varying attributes (``efficiency``,
  ``efficiency_dispatch``, ``efficiency_store``, ``standing_loss``). For
  example, this allows specifying temperature-dependent generator efficiencies
  or evaporation in hydro reservoirs. [`#572
  <https://github.com/PyPSA/PyPSA/pull/572>`_]

* Unit commitment constraints (ramp limits, start up and shut down costs) can
  now also be applied to links in addition to generators. This is useful to
  model the operational restrictions of fuel synthesis plants. [`#582
  <https://github.com/PyPSA/PyPSA/pull/582>`_]

* Added implementation for a linearized unit commitment approximation (LP-based)
  that can be activated when calling
  ``n.optimize(linearized_unit_commitment=True)``. The implementation follows
  Hua et al. (2017), `10.1109/TPWRS.2017.2735026
  <https://doi.org/10.1109/TPWRS.2017.2735026>`_. This functionality is not
  implemented for ``n.lopf()``. [`#472
  <https://github.com/PyPSA/PyPSA/pull/472>`_]

* NetCDF (``.nc``) and HDF5 (``.h5``) network files can now be read directly
  from URL:
  ``pypsa.Network("https://github.com/PyPSA/PyPSA/raw/master/examples/scigrid-de/scigrid-with-load-gen-trafos.nc")``
  [`#569
  <https://github.com/PyPSA/PyPSA/pull/569>`_]

* Networks are now compressed when exporting the NetCDF
  ``n.export_to_netcdf(...)`` step using the native compression feature of
  netCDF files. Additionally, a typecasting option from float64 to float 32 was
  added. Existing network files are not affected. To also compress existing
  networks, load and save them using ``xarray`` with compression specified, see
  `the xarray documentation
  <https://docs.xarray.dev/en/stable/generated/xarray.Dataset.to_netcdf.html>`_
  for details. The compression can be disabled with
  ``n.export_to_netcdf(compression=None)``. Use
  ``n.export_to_netcdf(float32=True, compression={'zlib': True, 'complevel': 9, 'least_significant_digit': 5})``
  for high compression. [`#583
  <https://github.com/PyPSA/PyPSA/pull/583>`_, `#614
  <https://github.com/PyPSA/PyPSA/pull/614>`_]

* Time aggregation for OPEX, curtailment, supply, withdrawal, and revenue now
  default to 'sum' rather than 'mean'.

* A new type of ``GlobalConstraint`` called `operational_limit` is now supported
  through the ``n.optimize()`` function. It allows to limit the total
  production of a carrier analogous to `primary_energy_limit` with the
  difference that it applies directly to the production of a carrier rather than
  to an attribute of the primary energy use. [`#618
  <https://github.com/PyPSA/PyPSA/pull/618>`_]

* The attributes ``lifetime`` and ``build_year`` are now aggregated with a
  capacity-weighted mean when clustering the network. Previously, these
  attributes had to carry identical values for components that were to be
  merged. [`#571
  <https://github.com/PyPSA/PyPSA/pull/571>`_]

* To enable better backwards compatibility with the ``n.lopf()`` function, the
  ``n.optimize()`` functions has now the explicit keyword argument
  ``solver_options``. It takes a dictionary of options passed to the solver.
  Before, these were passed as keyword arguments to the ``n.optimize()``
  function. Note that both functionalities are supported. [`#595
  <https://github.com/PyPSA/PyPSA/pull/595>`_]

* Fixed interference of io routines with linopy optimisation [`#564
  <https://github.com/PyPSA/PyPSA/pull/564>`_, `#567
  <https://github.com/PyPSA/PyPSA/pull/567>`_]

* Fix a bug where time-dependant generator variables could be forgotten during
  aggregation in a particular case. [`#576
  <https://github.com/PyPSA/PyPSA/pull/576>`_]

* A new type of ``GlobalConstraint`` called `operational_limit` is now supported through the `Network.optimize` function. It allows to limit the total production of a carrier analogous to `primary_energy_limit` with the difference that it applies directly to the production of a carrier rather than to an attribute of the primary energy use.

* Fix an issue appeared when processing networks which were reduced to a set of
  isolated nodes in course of clustering. Previously, an empty ``Line``
  component has lead to problems when processing empty lines-related dataframes.
  That has been fixed by introducing special treatment in case a lines dataframe
  is empty. [`#599
  <https://github.com/PyPSA/PyPSA/pull/599>`_]


`v0.22.1 <https://github.com/PyPSA/PyPSA/releases/tag/v0.22.1>`__ (15th February 2023)
=======================================================================================

* The model creation for large, sector-coupled models is now much quicker.
* The FICO Xpress solver interface now skips loading a basis if there is an
  error associated with the basis function and continues without it.
* The colors of borders and coastlines can now be controlled with
  ``n.plot(color_geomap=dict(border='b', coastline='r'))``.
* Plotting multiple legends was fixed for applying a tight layout with ``matplotlib>=3.6``.
* The plotting function now supports plotting negative and positive values
  separately per bus using the argument ```n.plot(bus_split_circles=...)``. This
  results in drawing separate half circles for positive and negative values.


`v0.22.0 <https://github.com/PyPSA/PyPSA/releases/tag/v0.22.0>`__ (3rd February 2023)
======================================================================================

* Python 3.11 is now tested. The support of Python 3.7 was dropped. The minimum supported python version is now 3.8.
* The linopy based optimization (``n.optimize()``) now allows to limit the carrier's growth by an additional linear term, so that one can limit an expansion growth by multiples of what was installed in the preceding investment period.
* The linopy based optimization now requires ``linopy`` v0.1.1 or higher. The new version eases the creation of custom constraint through a better display of linear expression and variables.
* Wrapped functions defined by the ``Network.optimize`` accessor are now wrapping meta information of the original functions more coherently. This enables better feedback in interactive sessions.
* Checking of datatypes in the ``consistency_check`` is now deactivated by default. Set ``n.consistency_check(check_dtypes=True)`` to activate it.
* The plotting functionality ``n.plot()`` now supports setting alpha values on the branch components individually.
* The plotting functionality ``n.plot()`` now allows independent control of arrow size and branch width using ``line_widths`` and ``flow`` in conjunction.
* The documentation shines in a new look using the ``sphinx-book-theme``. Limit ``sphinx`` to versions below 6.
* Address various deprecation warnings.

v0.21.3 (16th December 2022)
=================================

* Bugfix: Time-varying marginal cost of a component were removed if at least one of its value was zero.
* Bugfix: Due to xarray's ``groupby`` operation not fully supporting multi-indexes in recent version (see https://github.com/pydata/xarray/issues/6836), parts of the multi investment optimization code was adjusted.
* Update HiGHS parsing function in linopt for HiGHS version 1.4.0. Minimum version of HiGHS is v1.3.0. Older versions have not been tested.
* Update of gas boiler example to ``linopy``.
* New standard line types for DC lines.
* Included code of conduct.

v0.21.2 (30th November 2022)
=================================

* Compatibility with ``pyomo>=6.4.3``.

v0.21.1 (10th November 2022)
=================================

* Default of ``n.lopf()`` changed to ``n.lopf(pyomo=False)``.
* Bugfix in calculating statistics of curtailment.
* Bugfix in IO of netCDF network files for datetime indices.
* Bugfix for warning about imports from different PyPSA versions.
* Add linopy and statistics module to API reference.

v0.21.0 (7th November 2022)
================================

* A new optimization module `optimization` based on `Linopy <https://github.com/PyPSA/linopy>`_ was introduced. It aims at being as fast as the in-house optimization code and as flexible as the optimization with ``Pyomo``. A introduction to the optimization can be found at the `examples section
  <https://pypsa.readthedocs.io/en/latest/examples/optimization-with-linopy.html>`_ a migration guide for extra functionalities can be found at `here
  <https://pypsa.readthedocs.io/en/latest/examples/optimization-with-linopy-migrate-extra-functionalities.html>`_
* A new module for a quick calculation of system relevant quantities was introduced. It is directly accessible via the new accessor `Network.statistics` which returns a table of values often calculated manually. At the same time `Network.statistics` allows to call individual functions, as `capex`, `opex`, `capacity_factor` etc.
* Add reference to `Discord server <https://discord.gg/AnuJBk23FU>`_ for support and discussion.
* Restore import of pandapower networks. Issues regarding the transformer component and indexing as well as missing imports for shunts are fixed. [`#332 <https://github.com/PyPSA/PyPSA/pull/332>`_]
* The import performance of networks was improved. With the changes, the import time for standard netcdf imports decreased by roughly 70%.

v0.20.1 (6th October 2022)
===============================

* The representation of networks was modified to show the number of components and snapshots.
* The performance of the consistency check function was improved. The consistency check was extended by validating the capacity expansion limits as well as global constraint attributes.
* When applying network clustering algorithms, per unit time series are now aggregated using a capacity-weighted average and default aggregation strategies were adjusted.
* The value of ``n.objective`` is now set to NaN for failed optimisation runs.
* Added example notebook on how to model redispatch with PyPSA.
* Added new network plotting example.
* Bugfix for non-pyomo version of ``n.sclopf()``.
* Accept ``pathlib.Path`` objects when importing networks with ``pypsa.Network()``.
* Addressed ``.iteritems()`` deprecations.


v0.20.0 (26th July 2022)
==============================

This release contains new features for plotting and storing metadata with Network objects.

* A new attribute ``n.meta`` was added to the Network object. This can be an arbitrary dictionary, and is used to store meta data about the network.

* Improved support for individually normed colorbars in ``n.plot()`` for buses, lines, links, transformers with keyword arguments ``bus_norm``, ``line_norm``, ``link_norm``, ``transformer_norm``.

  .. code-block:: python
    :caption: Colorbar plotting example

    import pypsa
    import matplotlib.pyplot as plt
    n = pypsa.examples.ac_dc_meshed()
    norm = plt.Normalize(vmin=0, vmax=10)
    n.plot(
        bus_colors=n.buses.x,
        bus_cmap='viridis',
        bus_norm=norm
    )
    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=norm))

* New utility functions to add legends for line widths (:func:`pypsa.plot.add_legend_lines`), circles and pie chart areas (:func:`pypsa.plot.add_legend_circles`), and patch colors (:func:`pypsa.plot.add_legend_patches`).
  See the following example:

  .. code-block:: python
    :caption: Legend plotting example

    import pypsa
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from pypsa.plot import add_legend_circles

    n = pypsa.examples.ac_dc_meshed()

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    n.plot(ax=ax, bus_sizes=1)

    add_legend_circles(
        ax,
        [1, 0.5],
        ["reference size", "reference size 2"],
        legend_kw=dict(frameon=False, bbox_to_anchor=(1,0.1))
    )

* When iterating over components of a Subnetwork, only a those assets are included in the dataframes which are included in the subnetwork.

* In ``n.plot()``, compute boundaries in all cases for consistent circle sizes. This is realised by setting a new default margin of 0.05.

* Compatibility with pyomo 6.4.1.

* Removed ``pypsa.stats`` module.

* Extended defaults for the clustering of attributes in ``pypsa.networkclustering``.

* Removed deprecated clustering algorithms in ``pypsa.networkclustering``.

* Improved documentation and README.

* Fix a few deprecations.

* Improved test coverage, e.g. when copying networks.

* Testing: ``pypower`` is not importable with newest numpy versions. Skip test if import fails.

Special thanks for this release to @Cellophil,
@txelldm and @rockstaedt for improving test coverage and documentation.


v0.19.3 (22nd April 2022)
==============================

* Apply pre-commit formats to support development (incl. black formatting,
  jupyter cleanup, import sorting, preventing large file uploads). This will
  distort ``git blame`` functionality, which can be fixed by running ``git
  config blame.ignoreRevsFile .git-blame-ignore-revs`` inside the PyPSA
  repository. Run ``pre-commit install`` to set up locally.
* Change message when exporting and importing networks without a set ``network_name``.
  Fixes [`#381 <https://github.com/PyPSA/PyPSA/issues/381>`_].
* Greedy Modularity Maximisation was introduced as new spatial
  clustering method [`#377 <https://github.com/PyPSA/PyPSA/pull/377>`_].

v0.19.2 (7th March 2022)
=============================

* Add standard line type for 750 kV transmission line.

v0.19.1 (18th February 2022)
=================================

* When setting ramp limits for links and calling ``Network.lopf`` with ``pyomo=False``, an unexpected KeyError was raised. This was fixed by correctly accessing the data frame referring to the power dispatch of links.


v0.19.0 (11th February 2022)
=================================

This release contains new features for ramping constraints in link components,
hierarchical network clustering functionality, and an interface to the
open-source HiGHS solver.

**New Features**

* Ramp limits for ``Links``. The ``Link`` component has two new attributes, :code:`ramp_limit_up` and
  :code:`ramp_limit_down`, which limits the marginal power increase equivalent to the
  implementation for generators. The new attributes are only considered when
  running ``network.lopf(pyomo=False)``.

* Hierarchical Agglomerative Clustering (HAC) was introduced as new spatial
  clustering method [`#289 <https://github.com/PyPSA/PyPSA/pull/289>`_].

* Clustering networks now also supports the clustering of time-series associated
  to lines.

* Add open-source `HiGHS solver <https://github.com/ERGO-Code/HiGHS>`_.

* A new convenience function ``Network.get_committable_i`` was added. This returns
  an index containing all committable assets of component ``c``. In case that
  component ``c`` does not support committable assets, it returns an empty
  dataframe.

* A warning message is shown if a network contains one or more links with an
  :code:`efficiency` smaller than 1 and a negative value for :code:`p_min_pu`
  [`#320 <https://github.com/PyPSA/PyPSA/pull/320>`_].

* New example for spatial clustering.

* Speed-up of ``network.plot()`` by only plotting buses with non-zero size.

* Increased test coverage.

**Changes**

* The names of the indexes in static dataframes are now set to the component
  names. So, the index of ``n.generators`` has the name 'Generator'. The same
  accounts for the columns of the timeseries.

* The snapshot levels of a multi-indexed snapshot were renamed to ['period',
  'timestep'], the name of the index was set to 'snapshot'. This makes the
  snapshot name coherent for single and multi-indexed snapshots.

**Bugs and Compatibility**

* Compatibility with ``pandas>=1.4``.

* Drop support for Python 3.6 in accordance with its
  [end-of-life](https://endoflife.date/python).

* Use ``nx.Graph`` instead of ``nx.OrderedGraph`` which guarantees order is
  preserved for Python 3.7 and above.

* Add assert: CBC solver does not work with '>' and '<'.

* When running ``network.lopf(pyomo=False)``, the ramp limits did not take
  the time step right before the optimization horizon into account (relevant for
  rolling horizon optimization). This is now fixed.

* Fix bug when multi-links are defined but the network has no links.

Special thanks for this release to Samuel Matthew Dumlao (@smdumlao) for
implementing the ramp limits for Links in PyPSA, Martha Frysztacki (@martacki) for
implementing the hierarchical network clustering, and Max Parzen (@pz-max) for
implementing the HiGHS solver interface.

v0.18.1 (15th October 2021)
================================

* Compatibility with ``pyomo>=6.1``.

* Bugfix: specifying the ``solver_logfile`` is no longer mandatory with CPLEX for
  ``n.lopf(pyomo=False)``.

* The distance measures for the network clustering functions ``busmap_by_spectral()``
  and ``busmap_by_louvain()`` were adapted to use electrical distance
  (``s_nom/|r+i*x|``) (before: ``num_parallel``).

* Deprecations: The functions ``busmap_by_linemask()``, ``busmap_by_length()``, ``length_clustering()``,
  ``busmap_by_spectral_clustering()``, ``spectral_clustering()``, ``busmap_by_louvain()``,
  ``louvain_clustering()``, ``busmap_by_rectangular_grid()``, ``rectangular_grid_clustering()``
  and ``stubs_clustering()`` were deprecated and will be removed in v0.20.

* Distance measures for function ``busmap_by_spectral()`` and ``busmap_by_louvain()``
  were adapted to electrical distance (``s_nom/|r+i*x|``) (before: ``num_parallel``)

* In ``pypsa.networkclustering``, strip the string of the clustered
  component name. Not doing this had caused troubles for components with an
  empty carrier column.

* Various documentation updates.


v0.18.0 (12th August 2021)
===============================

This release contains new features for pathway optimisation, improvements of the
documentation's examples section as well as compatibility and bug fixes.

**Licensing**

* With this release, we have changed the licence from the copyleft GPLv3
  to the more liberal MIT licence with the consent of all contributors
  (for the reasoning why, see `#274 <https://github.com/PyPSA/PyPSA/pull/274>`_).

**New features**

* Added support for the optimisation of multiple investment periods, also known
  as pathway optimization. With this feature, snapshots can span over multiple
  years or decades which are divided into investment periods. Within each
  investment period, assets can be added to the network. The optimization only
  works with ``pyomo=False``. For more information see the documentation at :ref:`multi-horizon` and the `example notebook
  <https://pypsa.readthedocs.io/en/latest/examples/multi-investment-optimisation.html>`_. Endogenous learning curves can be applied as ``extra_functionality``.

* ``n.snapshot_weightings`` is now a ``pandas.DataFrame`` rather than
  a ``pandas.Series`` with weightings now subdivided into weightings
  for the objective function, generators and stores/storage
  units. This separation of weightings is relevant for temporal
  snapshot clustering, where the weight in the objective function may
  differ from the number of hours represented by each snapshot for
  storage purposes.

  * Objective weightings determine the multiplier of the marginal costs in the
    objective function of the LOPF.

  * Generator weightings specify the impact of generators in a
    ``GlobalConstraint`` (e.g. in a carbon dioxide emission constraint).

  * Store weightings define the elapsed hours for the charge, discharge,
    standing loss and spillage of storage units and stores in order to determine
    the current state of charge.

  PyPSA still supports setting ``n.snapshot_weightings`` with a ``pandas.Series``.
  In this case, the weightings are uniformly applied to all columns of the new
  ``n.snapshot_weightings`` ``pandas.DataFrame``.

* All functionalities except for optimisation with ``pyomo=True`` now work
  with multi-indexed snapshots.

* Many example notebooks are now also integrated in the
  documentation. See :doc:`/getting-started/quick-start`, :doc:`/examples-index/lopf`,
  :doc:`/examples-index/sector-coupling` and :doc:`/examples-index/other`.


* A new module ``examples`` was added which contains frontend functions for
  retrieving/loading example networks provided by the PyPSA project.

* When solving ``n.lopf(pyomo=False)``, PyPSA now supports setting lower and
  upper capacity bounds per bus and carrier. These are specified in the columns
  ``n.buses['nom_min_{carrier}']`` and ``n.buses['nom_max_{carrier}']``
  respectively. For example, if multiple generators of carrier ``wind`` are at bus
  ``bus1``, the combined capacity is limited to 1000 MW by setting
  ``n.buses.loc['bus1', 'nom_max_wind'] = 1000`` (a minimal capacity is forced by
  setting ``n.buses.loc['bus1', 'nom_min_wind']``). In the same manner the
  combined ``p_nom`` of components ``StorageUnit`` and ``e_nom`` of components
  ``Store`` can be limited.

* Add new attribute ``carrier`` to the components ``Line``, ``Link``, ``Store``
  and ``Load``, defining the energy carrier of the components. Its default is an
  empty string. When calling ``n.calculate_dependent_values()``, empty carriers
  are replaced by the carriers of the buses to which the components are attached.

* Add new descriptive attribute ``unit`` to ``bus`` component.

* Automated upload of code coverage reports for pull requests.

**Changes**

* When using iterative LOPF with ``n.ilopf()`` to consider impedance updates of
  reinforced transmission lines, the attributes ``p_nom`` and ``s_nom`` of lines
  and links are reset to their original values after final iteration.

* ``n.snapshots`` are now a property, hence assigning values with
  ``n.snapshots = values`` is the same as ``n.set_snapshots(values)``.

* Remove deprecated function ``geo.area_from_lon_lat_poly``.

**Deprecations**

* The function ``geo.area_from_lon_lat_poly()`` was deprecated and will be removed in v0.19.

* The deprecated argument ``csv_folder_name`` in ``pypsa.Network`` was removed.

* The deprecated column names ``source``, ``dispatch``, ``p_max_pu_fixed``,
  ``p_min_pu_fixed`` for the class ``Generator``, ``current_type`` for the class
  ``Bus`` and ``s_nom`` for the class ``Link`` were removed.

**Bugs and Compatibility**

* Added support for ``pandas`` version 1.3.

* Adjust log file creation for CPLEX version 12.10 and higher.

* ``n.snapshot_weightings`` is no longer copied for ``n.copy(with_time=False)``.

* Bugfix in ``n.ilopf()`` where previously all links were fixed in the final
  iteration when it should only be the HVDC links.

* Fix setting ``margin`` and ``boundaries`` when plotting a network with  ``geomap=False``.

Special thanks for this release to Lisa Zeyen (@lisazeyen) for implementing the
multi-horizon investment in PyPSA and to Fabian Hofmann (@FabianHofmann) for
thoroughly reviewing it and adding the example notebooks to the documentation.


v0.17.1 (15th July 2020)
=============================

This release contains bug fixes and extensions to the features for optimization when not using Pyomo.

* N-1 security-constrained linear optimal power flow is now also supported without pyomo by running ``network.sclopf(pyomo=False)``.

* Added support for the FICO Xpress commercial solver for optimization without pyomo, i.e. ``pyomo=False``.

* There was a bug in the LOPF with ``pyomo=False`` whereby if some Links
  were defined with multiple outputs (i.e. bus2, bus3, etc. were
  defined), but there remained some Links without multiple outputs
  (bus2, bus3, etc. set to ``""``), then the Links without multiple
  outputs were assigned erroneous non-zero values for p2, p3, etc. in
  the LOPF with ``pyomo=False``. Now p2, p3, etc. revert to the default
  value for Links where bus2, bus3, etc. are not defined, just like
  for the LOPF with ``pyomo=True``.

* Handle double-asterisk prefix in ``solution_fn`` when solving ``n.lopf(pyomo=False)`` using CBC.

* When solving ``n.lopf(pyomo=False, store_basis=True, solver_name="cplex")`` an error raised by trying to store a non-existing basis is caught.

* Add compatibility for Pyomo 5.7. This is also the new minimum requirement.

* Fixed bug when saving dual variables of the line volume limit. Now using dual from the second last iteration in ``pypsa.linopf``,
  because last iteration returns NaN (no optimisation of line capacities in final iteration).

* Added tracking of iterations of global constraints in the optimisation.

* When solving ``n.lopf(pyomo=False)``, PyPSA now constrains the dispatch variables for non extendable components with actual constraints, not with standard variable bounds. This allows retrieving shadow prices for all dispatch variables when running ``n.lopf(pyomo=False, keep_shadowprices=True)``.

* Can now cluster lines with different static ``s_max_pu`` values. Time-varying ``s_max_pu`` are not supported in clustering.

* Improved handling of optional dependencies for network clustering functionalities (``sklearn`` and ``community``).

Thanks to Pietro Belotti from FICO for adding the Xpress support, to Fabian Neumann (KIT) and Fabian Hofmann (FIAS) for all their
hard work on this release, and to all those who fixed bugs and reported issues.

v0.17.0 (23rd March 2020)
================================

This release contains some minor breaking changes to plotting, some
new features and bug fixes.


* For plotting geographical features ``basemap`` is not supported anymore.  Please use ``cartopy`` instead.
* Changes in the plotting functions ``n.plot()`` and ``n.iplot()`` include some **breaking changes**:

    * A set of new arguments were introduced to separate style parameters of the different branch components:  ``link_colors``, ``link_widths``, ``transformer_colors``, ``transformer_widths``, ``link_cmap``, ``transformer_cmap``
    * ``line_widths``, ``line_colors``, and ``line_cmap`` now only apply for lines and can no longer be used for other branch types (links and transformers). Passing a pandas.Series with a pandas.MultiIndex will raise an error.
    * Additionally, the function `n.iplot()` has new arguments ``line_text``, ``link_text``, ``transformer_text`` to configure the text displayed when hovering over a branch component.
    * The function ``directed_flow()`` now takes only a pandas.Series with single pandas.Index.
    * The argument ``bus_colorscale`` in ``n.iplot()`` was renamed to ``bus_cmap``.
    * The default colours changed.

* If non-standard output fields in the time-dependent ``network.components_t`` (e.g. ``network.links_t.p2`` when there are multi-links) were exported, then PyPSA will now also import them automatically without requiring the use of the ``override_component_attrs`` argument.
* Deep copies of networks can now be created with a subset of
  snapshots, e.g. ``network.copy(snapshots=network.snapshots[:2])``.
* When using the ``pyomo=False`` formulation of the LOPF (``network.lopf(pyomo=False)``):

    * It is now possible to alter the objective function.
      Terms can be added to the objective via ``extra_functionality``
      using the function `pypsa.linopt.write_objective`.
      When a pure custom objective function needs to be declared,
      one can set ``skip_objective=True``.
      In this case, only terms defined through ``extra_functionality``
      will be considered in the objective function.
    * Shadow prices of capacity bounds for non-extendable passive branches
      are parsed (similar to the ``pyomo=True`` setting)
    * Fixed `pypsa.linopf.define_kirchhoff_constraints` to handle
      exclusively radial network topologies.
    * CPLEX is now supported as an additional solver option. Enable it by installing the `cplex <https://pypi.org/project/cplex/>`_ package (e.g. via ``pip install cplex`` or ``conda install -c ibmdecisionoptimization cplex``) and setting ``solver_name='cplex'``

* When plotting, ``bus_sizes`` are now consistent when they have a ``pandas.MultiIndex``
  or a ``pandas.Index``. The default is changed to ``bus_sizes=0.01`` because the bus
  sizes now relate to the axis values.
* When plotting, ``bus_alpha`` can now be used to add an alpha channel
  which controls the opacity of the bus markers.
* The argument ``bus_colors`` can a now also be a pandas.Series.
* The ``carrier`` component has two new columns 'color' and 'nice_name'.
  The color column is used by the plotting function if ``bus_sizes`` is
  a pandas.Series with a MultiIndex and ``bus_colors`` is not explicitly defined.
* The function `pypsa.linopf.ilopf` can now track the intermediate branch capacities
  and objective values for each iteration using the ``track_iterations`` keyword.
* Fixed unit commitment:

    * when ``min_up_time`` of committable generators exceeds the length of snapshots.
    * when network does not feature any extendable generators.

* Fixed import from pandapower for transformers not based on standard types.
* The various Jupyter Notebook examples are now available on the `binder <https://mybinder.org/>`_ platform. This allows new users to interactively run and explore the examples without the need of installing anything on their computers.
* Minor adjustments for compatibility with pandas v1.0.0.
* After optimizing, the network has now an additional attribute ``objective_constant`` which reflects the capital cost of already existing infrastructure in the network referring to ``p_nom`` and ``s_nom`` values.

Thanks to Fabian Hofmann (FIAS) and Fabian Neumann (KIT) for all their
hard work on this release, and to all those who reported issues.


v0.16.1 (10th January 2020)
================================

This release contains a few minor bux fixes from the introduction of
nomopyomo in the previous release, as well as a few minor features.

* When using the ``nomopyomo`` formulation of the LOPF with
  ``network.lopf(pyomo=False)``, PyPSA was not correcting the bus
  marginal prices by dividing by the ``network.snapshot_weightings``, as is done
  in the ``pyomo`` formulation. This correction is now applied in the
  ``nomopyomo`` formulation to be consistent with the ``pyomo``
  formulation. (The reason this correction is applied is so that the
  prices have a clear currency/MWh definition regardless of the
  snapshot weightings. It also makes them stay roughly the same when
  snapshots are aggregated: e.g. if hourly simulations are sampled
  every n-hours, and the snapshot weighting is n.)
* The ``status, termination_condition`` that the ``network.lopf`` returns
  is now consistent between the ``nomopyomo`` and ``pyomo``
  formulations. The possible return values are documented in the LOPF
  docstring, see also the `LOPF documentation
  <https://pypsa.readthedocs.io/en/latest/user-guide/optimal-power-flow.html#pypsa.Network.lopf>`_.
  Furthermore in the ``nomopyomo`` formulation, the solution is still
  returned when gurobi finds a suboptimal solution, since this
  solution is usually close to optimal. In this case the LOPF returns
  a ``status`` of ``warning`` and a ``termination_condition`` of
  ``suboptimal``.
* For plotting with ``network.plot()`` you can override the bus
  coordinates by passing it a ``layouter`` function from ``networkx``. See
  the docstring for more information. This is particularly useful for
  networks with no defined coordinates.
* For plotting with ``network.iplot()`` a background from `mapbox
  <https://www.mapbox.com/>`_ can now be integrated.

Please note that we are still aware of one implementation difference
between ``nomopyomo`` and ``pyomo``, namely that ``nomopyomo`` doesn't read
out shadow prices for non-extendable branches, see the `github issue
<https://github.com/PyPSA/PyPSA/issues/119>`_.


v0.16.0 (20th December 2019)
=================================

This release contains major new features. It is also the first release
to drop support for Python 2.7. Only Python 3.6 and 3.7 are supported
going forward. Python 3.8 will be supported as soon as the gurobipy
package in conda is updated.

* A new version of the linear optimal power flow (LOPF) has been
  introduced that uses a custom optimization framework rather than
  Pyomo. The new framework, based on `nomoypomo
  <https://github.com/PyPSA/nomopyomo>`_, uses barely any memory and
  is much faster than Pyomo. As a result the total memory usage of
  PyPSA processing and gurobi is less than a third what it is with
  Pyomo for large problems with millions of variables that take
  several gigabytes of memory (see this `graphical comparison
  <https://github.com/PyPSA/PyPSA/pull/99#issuecomment-560490397>`_
  for a large network optimization). The new framework is not enabled
  by default. To enable it, use ``network.lopf(pyomo=False)``. Almost
  all features of the regular ``network.lopf`` are implemented with
  the exception of minimum down/up time and start up/shut down costs
  for unit commitment. If you use the ``extra_functionality`` argument
  for ``network.lopf`` you will need to update your code for the new
  syntax. There is `documentation
  <https://pypsa.readthedocs.io/en/latest/user-guide/optimal-power-flow.html#pyomo-is-set-to-false>`_
  for the new syntax as well as a `Jupyter notebook of examples
  <https://github.com/PyPSA/PyPSA/blob/master/examples/lopf_with_pyomo_False.ipynb>`_.

* Distributed active power slack is now implemented for the full
  non-linear power flow. If you pass ``network.pf()`` the argument
  ``distribute_slack=True``, it will distribute the slack power across
  generators proportional to generator dispatch by default, or
  according to the distribution scheme provided in the argument
  ``slack_weights``. If ``distribute_slack=False`` only the slack
  generator takes up the slack. There is further `documentation
  <https://pypsa.readthedocs.io/en/latest/user-guide/power-flow.html#full-non-linear-power-flow>`__.

* Unit testing is now performed on all of GNU/Linux, Windows and MacOS.

* NB: You may need to update your version of the package ``six``.

Special thanks for this release to Fabian Hofmann for implementing the
nomopyomo framework in PyPSA and Fabian Neumann for providing the
customizable distributed slack.


v0.15.0 (8th November 2019)
================================

This release contains new improvements and bug fixes.

* The unit commitment (UC) has been revamped to take account of
  constraints at the beginning and end of the simulated ``snapshots``
  better. This is particularly useful for rolling horizon UC. UC now
  accounts for up-time and down-time in the periods before the
  ``snapshots``. The generator attribute ``initial_status`` has been
  replaced with two attributes ``up_time_before`` and
  ``down_time_before`` to give information about the status before
  ``network.snapshots``. At the end of the simulated ``snapshots``, minimum
  up-times and down-times are also enforced. Ramping constraints also
  look before the simulation at previous results, if there are
  any. See the `unit commitment documentation
  <https://pypsa.readthedocs.io/en/latest/user-guide/optimal-power-flow.html#generator-unit-commitment-constraints>`_
  for full details. The `UC example
  <https://pypsa.readthedocs.io/en/latest/examples/unit-commitment.html>`_ has been updated
  with a rolling horizon example at the end.
* Documentation is now available on `readthedocs
  <https://pypsa.readthedocs.io/>`_, with information about functions
  pulled from the docstrings.
* The dependency on cartopy is now an optional extra.
* PyPSA now works with pandas 0.25 and above, and networkx above 2.3.
* A bug was fixed that broke the Security-Constrained Linear Optimal
  Power Flow (SCLOPF) constraints with extendable lines.
* Network plotting can now plot arrows to indicate the direction of flow by passing ``network.plot`` an ``flow`` argument.
* The objective sense (``minimize`` or ``maximize``) can now be set (default
  remains ``minimize``).
* The ``network.snapshot_weightings`` is now carried over when the network
  is clustered.
* Various other minor fixes.

We thank colleagues at TERI for assisting with testing the new unit
commitment code, Clara Büttner for finding the SCLOPF bug, and all
others who contributed issues and pull requests.


v0.14.1 (27th May 2019)
================================

This minor release contains three small bug fixes:

* Documentation parses now correctly on PyPI
* Python 2.7 and 3.6 are automatically tested using Travis
* PyPSA on Python 2.7 was fixed

This will also be the first release to be available directly from
`conda-forge <https://conda-forge.org/>`_.

v0.14.0 (15th May 2019)
============================

This release contains a new feature and bug fixes.

* Network plotting can now use the mapping library `cartopy
  <https://scitools.org.uk/cartopy/>`_ as well as `basemap
  <https://matplotlib.org/basemap/>`_, which was used in previous
  versions of PyPSA. The basemap developers will be phasing out
  basemap over the next few years in favour of cartopy (see their
  `end-of-life announcement
  <https://matplotlib.org/basemap/users/intro.html#cartopy-new-management-and-eol-announcement>`_). PyPSA
  now defaults to cartopy unless you tell it explicitly to use
  basemap. Otherwise the plotting interface is the same as in previous
  versions.
* Optimisation now works with the newest version of Pyomo 5.6.2 (there
  was a Pyomo update that affected the opt.py expression for building
  linear sums).
* A critical bug in the networkclustering sub-library has been fixed
  which was preventing the capital_cost parameter of conventional
  generators being handled correctly when networks are aggregated.
* Network.consistency_check() now only prints necessary columns when
  reporting NaN values.
* Import from `pandapower <https://www.pandapower.org/>`__ networks has
  been updated to pandapower 2.0 and to include non-standard lines and
  transformers.

We thank Fons van der Plas and Fabian Hofmann for helping with the
cartopy interface, Chloe Syranidis for pointing out the problem with
the Pyomo 5.6.2 update, Hailiang Liu for the consistency check update
and Christian Brosig for the pandapower updates.

v0.13.2 (10th January 2019)
================================

This minor release contains small new features and fixes.

* Optimisation now works with Pyomo >= 5.6 (there was a Pyomo update
  that affected the opt.py LConstraint object).
* New functional argument can be passed to Network.lopf:
  extra_postprocessing(network,snapshots,duals), which is called after
  solving and results are extracted. It can be used to get the values
  of shadow prices for constraints that are not normally extracted by
  PyPSA.
* In the lopf kirchhoff formulation, the cycle constraint is rescaled
  by a factor 1e5, which improves the numerical stability of the
  interior point algorithm (since the coefficients in the constraint
  matrix were very small).
* Updates and fixes to networkclustering, io, plot.

We thank Soner Candas of TUM for reporting the problem with the most
recent version of Pyomo and providing the fix.


v0.13.1 (27th March 2018)
==============================

This release contains bug fixes for the new features introduced in
0.13.0.

* Export network to netCDF file bug fixed (components that were all
  standard except their name were ignored).
* Import/export network to HDF5 file bug fixed and now works with more
  than 1000 columns; HDF5 format is no longer deprecated.
* When networks are copied or sliced, overridden components
  (introduced in 0.13.0) are also copied.
* Sundry other small fixes.

We thank Tim Kittel for pointing out the first and second bugs. We
thank Kostas Syranidis for not only pointing out the third issue with
copying overridden components, but also submitting a fix as a pull
request.

For this release we acknowledge funding to Tom Brown from the
`RE-INVEST project <http://www.reinvestproject.eu/>`_.



v0.13.0 (25th January 2018)
================================

This release contains new features aimed at coupling power networks to
other energy sectors, fixes for library dependencies and some minor
internal API changes.

* If you want to define your own components and override the standard
  functionality of PyPSA, you can now override the standard components
  by passing pypsa.Network() the arguments ``override_components`` and
  ``override_component_attrs``, see the section on
  :ref:`custom_components`. There are examples for defining new
  components in the git repository in ``examples/new_components/``,
  including an example of overriding ``network.lopf()`` for
  functionality for combined-heat-and-power (CHP) plants.
* The ``Link`` component can now be defined with multiple outputs in
  fixed ratio to the power in the single input by defining new columns
  ``bus2``, ``bus3``, etc. (``bus`` followed by an integer) in
  ``network.links`` along with associated columns for the efficiencies
  ``efficiency2``, ``efficiency3``, etc. The different outputs are
  then proportional to the input according to the efficiency; see
  sections :ref:`components-links-multiple-outputs` and
  :ref:`opf-links` and the `example of a CHP with a fixed power-heat
  ratio
  <https://pypsa.readthedocs.io/en/latest/examples/chp-fixed-heat-power-ratio.html>`_.
* Networks can now be exported to and imported from netCDF files with
  ``network.export_to_netcdf()`` and
  ``network.import_from_netcdf()``. This is faster than using CSV
  files and the files take up less space. Import and export with HDF5
  files, introduced in v0.12.0, is now deprecated.
* The export and import code has been refactored to be more general
  and abstract. This does not affect the API.
* The internally-used sets such as ``pypsa.components.all_components``
  and ``pypsa.components.one_port_components`` have been moved from
  ``pypsa.components`` to ``network``, i.e. ``network.all_components``
  and ``network.one_port_components``, since these sets may change
  from network to network.
* For linear power flow, PyPSA now pre-calculates the effective per
  unit reactance ``x_pu_eff`` for AC lines to take account of the
  transformer tap ratio, rather than doing it on the fly; this makes
  some code faster, particularly the kirchhoff formulation of the
  LOPF.
* PyPSA is now compatible with networkx 2.0 and 2.1.
* PyPSA now requires Pyomo version greater than 5.3.
* PyPSA now uses the `Travis CI <https://travis-ci.org/PyPSA/PyPSA>`_
  continuous integration service to test every commit in the `PyPSA
  GitHub repository <https://github.com/PyPSA/PyPSA>`_. This will
  allow us to catch library dependency issues faster.

We thank Russell Smith of Edison Energy for the pull request for the
effective reactance that sped up the LOPF code and Tom Edwards for
pointing out the Pyomo version dependency issue.

For this release we also acknowledge funding to Tom Brown from the
`RE-INVEST project <http://www.reinvestproject.eu/>`_.




v0.12.0 (30th November 2017)
=================================

This release contains new features and bug fixes.

* Support for Pyomo's persistent solver interface, so if you're making
  small changes to an optimisation model (e.g. tweaking a parameter),
  you don't have to rebuild the model every time. To enable this,
  ``network_lopf`` has been internally split into ``build_model``,
  ``prepare_solver`` and ``solve`` to allow more fine-grained control of the
  solving steps.  Currently the new Pyomo PersistentSolver interface
  is not in the main Pyomo branch, see
  `#223 <https://github.com/Pyomo/pyomo/pull/223>`_; you can obtain it with
  ``pip install git+https://github.com/Pyomo/pyomo@persistent_interfaces``
* Lines and transformers (i.e. passive branches) have a new attribute
  ``s_max_pu`` to restrict the flow in the OPF, just like ``p_max_pu``
  for generators and links. It works by restricting the absolute value
  of the flow per unit of the nominal rating ``abs(flow) <=
  s_max_pu*s_nom``. For lines this can represent an n-1 contingency
  factor or it can be time-varying to represent weather-dependent
  dynamic line rating.
* The ``marginal_cost`` attribute of generators, storage units, stores
  and links can now be time dependent.
* When initialising the Network object, i.e. ``network =
  pypsa.Network()``, the first keyword argument is now ``import_name``
  instead of ``csv_folder_name``. With ``import_name`` PyPSA
  recognises whether it is a CSV folder or an HDF5 file based on the
  file name ending and deals with it appropriately. Example usage:
  ``nw1 = pypsa.Network("my_store.h5")`` and ``nw2 =
  pypsa.Network("/my/folder")``. The keyword argument
  ``csv_folder_name`` is still there but is deprecated.
* The value ``network.objective`` is now read from the Pyomo results
  attribute ``Upper Bound`` instead of ``Lower Bound``. This is
  because for MILP problems under certain circumstances CPLEX records
  the ``Lower bound`` as the relaxed value. ``Upper bound`` is correctly
  recorded as the integer objective value.
* Bug fix due to changes in pandas 0.21.0: A bug affecting various
  places in the code, including causing ``network.lopf`` to fail with
  GLPK, is fixed. This is because in pandas 0.21.0 the sum of an empty
  Series/DataFrame returns NaN, whereas before it returned zero. This
  is a subtle bug; we hope we've fixed all instances of it, but get in
  touch if you notice NaNs creeping in where they shouldn't be. All
  our tests run fine.
* Bug fix due to changes in scipy 1.0.0: For the new version of scipy,
  ``csgraph`` has to be imported explicit.
* Bug fix: A bug whereby logging level was not always correctly being
  seen by the OPF results printout is fixed.
* Bug fix: The storage unit spillage had a bug in the LOPF, whereby it
  was not respecting ``network.snapshot_weightings`` properly.

We thank René Garcia Rosas, João Gorenstein Dedecca, Marko Kolenc,
Matteo De Felice and Florian Kühnlenz for promptly notifying us about
issues.


v0.11.0 (21st October 2017)
================================

This release contains new features but no changes to existing APIs.

* There is a new function ``network.iplot()`` which creates an
  interactive plot in Jupyter notebooks using the `plotly
  <https://plot.ly/python/>`_ library. This reveals bus and branch
  properties when the mouse hovers over them and allows users to
  easily zoom in and out on the network. See the (sparse) documentation
  :doc:`/user-guide/plotting`.
* There is a new function ``network.madd()`` for adding multiple new
  components to the network. This is significantly faster than
  repeatedly calling ``network.add()`` and uses the functions
  ``network.import_components_from_dataframe()`` and
  ``network.import_series_from_dataframe()`` internally.
* There are new functions ``network.export_to_hdf5()`` and
  ``network.import_from_hdf5()`` for exporting and importing networks
  as single files in the `Hierarchical Data Format
  <https://en.wikipedia.org/wiki/Hierarchical_Data_Format>`_.
* In the ``network.lopf()`` function the KKT shadow prices of the
  branch limit constraints are now outputted as series called
  ``mu_lower`` and ``mu_upper``.

We thank Bryn Pickering for introducing us to `plotly
<https://plot.ly/python/>`_ and helping to `hack together
<https://forum.openmod-initiative.org/t/breakout-group-on-visualising-networks-with-plotly/>`_
the first working prototype using PyPSA.


v0.10.0 (7th August 2017)
==============================

This release contains some minor new features and a few minor but
important API changes.

* There is a new component :ref:`global-constraints` for implementing
  constraints that effect many components at once (see also the
  LOPF subsection :ref:`global-constraints-opf`).  Currently only
  constraints related to primary energy (i.e. before conversion with
  losses by generators) are supported, the canonical example being CO2
  emissions for an optimisation period. Other primary-energy-related
  gas emissions also fall into this framework. Other types of global
  constraints will be added in future, e.g. "final energy" (for limits
  on the share of renewable or nuclear electricity after conversion),
  "generation capacity" (for limits on total capacity expansion of
  given carriers) and "transmission capacity" (for limits on the total
  expansion of lines and links). This replaces the ad hoc
  ``network.co2_limit`` attribute. If you were using this, instead of
  ``network.co2_limit = my_cap`` do ``network.add("GlobalConstraint",
  "co2_limit", type="primary_energy",
  carrier_attribute="co2_emissions", sense="<=",
  constant=my_cap)``. The shadow prices of the global constraints
  are automatically saved in ``network.global_constraints.mu``.
* The LOPF output ``network.buses_t.marginal_price`` is now defined
  differently if ``network.snapshot_weightings`` are not 1. Previously
  if the generator at the top of the merit order had ``marginal_cost``
  c and the snapshot weighting was w, the ``marginal_price`` was
  cw. Now it is c, which is more standard. See also
  :ref:`nodal-power-balance`.
* ``network.pf()`` now returns a dictionary of pandas DataFrames, each
  indexed by snapshots and sub-networks. ``converged`` is a table of
  booleans indicating whether the power flow has converged; ``error``
  gives the deviation of the non-linear solution; ``n_iter`` the
  number of iterations required to achieve the tolerance.
* ``network.consistency_check()`` now includes checking for
  potentially infeasible values in ``generator.p_{min,max}_pu``.
* The PyPSA version number is now saved in
  ``network.pypsa_version``. In future versions of PyPSA this
  information will be used to upgrade data to the latest version of
  PyPSA.
* ``network.sclopf()`` has an ``extra_functionality`` argument that
  behaves like that for ``network.lopf()``.
* Component attributes which are strings are now better handled on
  import and in the consistency checking.
* There is a new `generation investment screening curve example
  <https://pypsa.readthedocs.io/en/latest/examples/generation-investment-screening-curve.html>`_
  showing the long-term equilibrium of generation investment for a
  given load profile and comparing it to a screening curve
  analysis.
* There is a new `logging example
  <https://pypsa.readthedocs.io/en/latest/examples/logging-demo.html>`_ that demonstrates
  how to control the level of logging that PyPSA reports back,
  e.g. error/warning/info/debug messages.
* Sundry other bug fixes and improvements.
* All examples have been updated appropriately.


Thanks to Nis Martensen for contributing the return values of
``network.pf()`` and Konstantinos Syranidis for contributing the
improved ``network.consistency_check()``.



v0.9.0 (29th April 2017)
=============================

This release mostly contains new features with a few minor API
changes.

* Unit commitment as a MILP problem is now available for generators in
  the Linear Optimal Power Flow (LOPF). If you set ``committable ==
  True`` for the generator, an addition binary online/offline status
  is created. Minimum part loads, minimum up times, minimum down
  times, start up costs and shut down costs are implemented. See the
  documentation at :ref:`unit-commitment` and the `unit commitment
  example <https://pypsa.readthedocs.io/en/latest/examples/unit-commitment.html>`_. Note
  that a generator cannot currently have both unit commitment and
  capacity expansion optimisation.
* Generator ramping limits have also been implemented for all
  generators. See the documentation at :ref:`ramping` and the `unit
  commitment example
  <https://pypsa.readthedocs.io/en/latest/examples/unit-commitment.html>`_.
* Different mathematically-equivalent formulations for the Linear
  Optimal Power Flow (LOPF) are now documented
  and the arXiv preprint paper `Linear Optimal Power Flow Using Cycle
  Flows <https://arxiv.org/abs/1704.01881>`_. The new formulations can
  solve up to 20 times faster than the standard angle-based
  formulation.
* You can pass the ``network.lopf`` function the ``solver_io``
  argument for pyomo.
* There are some improvements to network clustering and graphing.
* API change: The attribute ``network.now`` has been removed since it
  was unnecessary. Now, if you do not pass a ``snapshots`` argument to
  network.pf() or network.lpf(), these functions will default to
  ``network.snapshots`` rather than ``network.now``.
* API change: When reading in network data from CSV files, PyPSA will
  parse snapshot dates as proper datetimes rather than text strings.


João Gorenstein Dedecca has also implemented a MILP version of the
transmission expansion, see
`<https://github.com/jdedecca/MILP_PyPSA>`_, which properly takes
account of the impedance with a disjunctive relaxation. This will be
pulled into the main PyPSA code base soon.


v0.8.0 (25th January 2017)
===============================

This is a major release which contains important new features and
changes to the internal API.

* Standard types are now available for lines and transformers so that
  you do not have to calculate the electrical parameters yourself. For
  lines you just need to specify the type and the length, see
  :ref:`line-types`. For transformers you just need to specify the
  type, see :ref:`transformer-types`. The implementation of PyPSA's
  standard types is based on `pandapower's standard types
  <https://pandapower.readthedocs.io/en/latest/std_types/basic.html>`_. The
  old interface of specifying r, x, b and g manually is still available.
* The transformer model has been substantially overhauled, see
  :ref:`transformer-model`. The equivalent model now defaults to the
  more accurate T model rather than the PI model, which you can control
  by setting the attribute ``model``. Discrete tap steps are implemented
  for transformers with types. The tap changer can be defined on the
  primary side or the secondary side. In the PF there was a sign error in the implementation of the transformer
  ``phase_shift``, which has now been fixed. In the LPF and LOPF angle formulation the ``phase_shift`` has now been
  implemented consistently. See the new `transformer example <https://pypsa.readthedocs.io/en/latest/examples/transformer_example.html>`_.
* There is now a rudimentary import function for pandapower networks,
  but it doesn't yet work with all switches and 3-winding
  transformers.
* The object interface for components has been completely
  removed. Objects for each component are no longer stored in
  e.g. ``network.lines["obj"]`` and the descriptor
  interface for components is gone. You can only access component
  attributes through the dataframes, e.g. ``network.lines``.
* Component attributes are now defined in CSV files in
  ``pypsa/data/component_attrs/``. You can access these CSVs in the code
  via the dictionary ``network.components``,
  e.g. ``network.components["Line"]["attrs"]`` will show a pandas
  DataFrame with all attributes and their types, defaults, units and
  descriptions.  These CSVs are also sourced for the documentation in
  :doc:`/user-guide/components`, so the documentation will always be up-to-date.
* All examples have been updated appropriately.




v0.7.1 (26th November 2016)
================================

This release contains bug fixes, a minor new feature and more
warnings.

* The unix-only library ``resource`` is no longer imported by default,
  which was causing errors for Windows users.
* Bugs in the setting and getting of time-varying attributes for the
  object interface have been fixed.
* The ``Link`` attribute ``efficiency`` can now be make time-varying
  so that e.g. heat pump Coefficient of Performance (COP) can change
  over time due to ambient temperature variations (see the `heat pump
  example
  <https://pypsa.readthedocs.io/en/latest/examples/power-to-heat-water-tank.html>`_).
* ``network.snapshots`` is now cast to a ``pandas.Index``.
* There are new warnings, including when you attach components to
  non-existent buses.


Thanks to Marius Vespermann for promptly pointing out the ``resource``
bug.





v0.7.0 (20th November 2016)
================================

This is a major release which contains changes to the API,
particularly regarding time-varying component attributes.

* ``network.generators_t`` are no longer pandas.Panels but
  dictionaries of pandas.DataFrames, with variable columns, so that
  you can be flexible about which components have time-varying
  attributes; please read :ref:`time-varying` carefully. Essentially
  you can either set a component attribute e.g. ``p_max_pu`` of
  ``Generator``, to be static by setting it in the DataFrame
  ``network.generators``, or you can let it be time-varying by
  defining a new column labelled by the generator name in the
  DataFrame ``network.generators_t["p_max_pu"]`` as a series, which
  causes the static value in ``network.generators`` for that generator
  to be ignored. The DataFrame ``network.generators_t["p_max_pu"]``
  now only includes columns which are specifically defined to be
  time-varying, thus saving memory.
* The following component attributes can now be time-varying:
  ``Link.p_max_pu``, ``Link.p_min_pu``, ``Store.e_max_pu`` and
  ``Store.e_min_pu``. This allows the demand-side management scheme of
  `<https://arxiv.org/abs/1401.4121>`_ to be implemented in PyPSA.
* The properties ``dispatch``, ``p_max_pu_fixed`` and
  ``p_min_pu_fixed`` of ``Generator`` and ``StorageUnit`` are now
  removed, because the ability to make ``p_max_pu`` and
  ``p_min_pu`` either static or time-varying removes the need for this
  distinction.
* All messages are sent through the standard Python library
  ``logging``, so you can control the level of messages to be
  e.g. ``debug``, ``info``, ``warning`` or ``error``. All verbose
  switches and print statements have been removed.
* There are now more warnings.
* You can call ``network.consistency_check()`` to make sure all your
  components are well defined; see :doc:`/contributing/troubleshooting`.


All `examples <https://pypsa.readthedocs.io/en/latest/examples-basic.html>`_ have been updated to
accommodate the changes listed below.


v0.6.2 (4th November 2016)
===============================

This release fixes a single library dependency issue:

* pf: A single line has been fixed so that it works with new pandas
  versions >= 0.19.0.

We thank Thorben Meiners for promptly pointing out this issue with the
new versions of pandas.


v0.6.1 (25th August 2016)
==============================

This release fixes a single critical bug:

* opf: The latest version of Pyomo (4.4.1) had a bad interaction with
  pandas when a pandas.Index was used to index variables. To fix this,
  the indices are now cast to lists; compatibility with less recent
  versions of Pyomo is also retained.

We thank Joao Gorenstein Dedecca for promptly notifying us of this
bug.



v0.6.0 (23rd August 2016)
==============================

Like the 0.5.0 release, this release contains API changes, which
complete the integration of sector coupling. You may have to update
your old code. Models for Combined Heat and Power (CHP) units, heat
pumps, resistive Power-to-Heat (P2H), Power-to-Gas (P2G), battery
electric vehicles (BEVs) and chained hydro reservoirs can now be built
(see the `sector coupling examples
<https://pypsa.readthedocs.io/en/latest/examples-index/sector-coupling.html>`_). The
refactoring of time-dependent variable handling has been postponed
until the 0.7.0 release. In 0.7.0 the object interface to attributes
may also be removed; see below.

All `examples <https://pypsa.readthedocs.io/en/latest/examples-basic.html>`_ have been updated to
accommodate the changes listed below.

**Sector coupling**

* components, opt: A new ``Store`` component has been introduced which
  stores energy, inheriting the energy carrier from the bus to which
  it is attached. The component is more fundamental than the
  ``StorageUnit``, which is equivalent to a ``Store`` and two ``Link``
  for storing and dispatching. The ``Generator`` is equivalent to a
  ``Store`` with a lossy ``Link``. There is an `example which shows
  the equivalences
  <https://pypsa.readthedocs.io/en/latest/examples/replace-generator-storage-units-with-store.html>`_.

* components, opt: The ``Source`` component and the ``Generator``
  attribute ``gen.source`` have been renamed ``Carrier`` and
  ``gen.carrier``, to be consistent with the ``bus.carrier``
  attribute. Please update your old code.

* components, opt: The ``Link`` attributes ``link.s_nom*`` have been
  renamed ``link.p_nom*`` to reflect the fact that the link can only
  dispatch active power. Please update your old code.

* components, opt: The ``TransportLink`` and ``Converter`` components,
  which were deprecated in 0.5.0, have been now completely
  removed. Please update your old code to use ``Link`` instead.

**Downgrading object interface**

The intention is to have only the pandas DataFrame interface for
accessing component attributes, to make the code simpler. The
automatic generation of objects with descriptor access to attributes
may be removed altogether.

* examples: Patterns of for loops through ``network.components.obj`` have
  been removed.

* components: The methods on ``Bus`` like ``bus.generators()`` and
  ``bus.loads()`` have been removed.

* components: ``network.add()`` no longer returns the object.

**Other**

* components, opf: Unlimited upper bounds for
  e.g. ``generator.p_nom_max`` or ``line.s_nom_max`` were previous set
  using ``np.nan``; now they are set using ``float("inf")`` which is
  more logical. You may have to update your old code accordingly.

* components: A memory leak whereby references to
  ``component.network`` were not being correctly deleted has been
  fixed.



v0.5.0 (21st July 2016)
============================

This is a relatively major release with some API changes, primarily
aimed at allowing coupling with other energy carriers (heat, gas,
etc.). The specification for a change and refactoring to the handling
of time series has also been prepared (see :ref:`time-varying`), which will
be implemented in the next major release v0.6.0 in the late
summer of 2016.

An example of the coupling between electric and heating sectors can be
found in the GitHub repository at
``pypsa/examples/coupling-with-heating/`` and at
`<https://pypsa.readthedocs.io/en/latest/examples/lopf-with-heating.html>`_.


* components: To allow other energy carriers, the attribute
  ``current_type`` fur buses and sub-neworks (sub-networks inherit the
  attribute from their buses) has been replaced by ``carrier`` which
  can take generic string values (such as "heat" or "gas"). The values
  "DC" and "AC" have a special meaning and PyPSA will treat lines and
  transformers within these sub-networks according to the load flow
  equations. Other carriers can only have single buses in sub-networks
  connected by passive branches (since they have no load flow).

* components: A new component for a controllable directed link
  ``Link`` has been introduced; ``TransportLink`` and ``Converter``
  are now *deprecated* and will be removed soon in an 0.6.x
  release. Please move your code over now. See
  :ref:`controllable-link` for more details and a description of how
  to update your code to work with the new ``Link`` component. All the
  examples in the GitHub repository in ``pypsa/examples/`` have been
  updated to us the ``Link``.

* graph: A new sub-module ``pypsa.graph`` has been introduced to
  replace most of the networkx functionality with scipy.sparse
  methods, which are more performant the pure python methods of
  networkx. The discovery of network connected components is now
  significantly faster.

* io: The function ``network.export_to_csv_folder()`` has been
  rewritten to only export non-default values of static and series
  component attributes. Static and series attributes of all components
  are not exported if they are default values.  The functionality to
  selectively export series has been removed from the export function,
  because it was clumsy and hard to use.


* plot: Plotting networks is now more performant (using matplotlib
  LineCollections) and allows generic branches to be plotted, not just
  lines.

* test: Unit testing for Security-Constrained Linear Optimal Power
  Flow (SCLOPF) has been introduced.


v0.4.2 (17th June 2016)
============================

This release improved the non-linear power flow performance and
included other small refactorings:

* pf: The non-linear power flow ``network.pf()`` now accepts a list of
  snapshots ``network.pf(snapshots)`` and has been refactored to be much
  more performant.
* pf: Neither ``network.pf()`` nor ``network.lpf()`` accept the
  ``now`` argument anymore - for the power flow on a specific
  snapshot, either set ``network.now`` or pass the snapshot as an
  argument.
* descriptors: The code has been refactored and unified for each
  simple descriptor.
* opt: Constraints now accept both an upper and lower bound with
  ``><``.
* opf: Sub-optimal solutions can also be read out of pyomo.


v0.4.1 (3rd April 2016)
============================

This was mostly a bug-fixing and unit-testing release:

* pf: A bug was fixed in the full non-linear power flow, whereby the
  reactive power output of PV generators was not being set correctly.
* io: When importing from PYPOWER ppc, the generators, lines,
  transformers and shunt impedances are given names like G1, G2, ...,
  L1, T1, S1, to help distinguish them. This change was introduced
  because the above bug was not caught by the unit-testing because the
  generators were named after the buses.
* opf: A Python 3 dict.keys() list/iterator bug was fixed for the
  spillage.
* test: Unit-testing for the pf and opf with inflow was improved to
  catch bugs better.

We thank Joao Gorenstein Dedecca for a bug fix.


v0.4.0 (21st March 2016)
================================

Additional features:

* New module ``pypsa.contingency`` for contingency analysis and
  security-constrained LOPF
* New module ``pypsa.geo`` for basic manipulation of geographic data
  (distances and areas)
* Re-formulation of LOPF to improve optimisation solving time
* New objects pypsa.opt.LExpression and pypsa.opt.LConstraint to make
  the bypassing of pyomo for linear problem construction easier to use
* Deep copying of networks with ``network.copy()`` (i.e. all
  components, time series and network attributes are copied)
* Stricter requirements for PyPI (e.g. pandas must be at least version
  0.17.1 to get all the new features)
* Updated SciGRID-based model of Germany
* Various small bug fixes

We thank Steffen Schroedter, Bjoern Laemmerzahl and Joao Gorenstein
Dedecca for comments and bug fixes.


v0.3.3 (29th February 2016)
================================

Additional features:

* ``network.lpf`` can be called on an iterable of ``snapshots``
  i.e. ``network.lpf(snapshots)``, which is more performant that
  calling ``network.lpf`` on each snapshot separately.
* Bug fix on import/export of transformers and shunt impedances (which
  were left out before).
* Refactoring of some internal code.
* Better network clustering.


v0.3.2 (17th February 2016)
================================

In this release some minor API changes were made:

* The Newton-Raphson tolerance ``network.nr_x_tol`` was moved to being
  an argument of the function ``network.pf(x_tol=1e-6)`` instead. This
  makes more sense and is then available in the docstring of
  ``network.pf``.
* Following similar reasoning ``network.opf_keep_files`` was moved to
  being an argument of the function
  ``network.lopf(keep_files=False)``.


v0.3.1 (7th February 2016)
===============================

In this release some minor API changes were made:


* Optimised capacities of generators/storage units and branches are
  now written to p_nom_opt and s_nom_opt respectively, instead of
  over-writing p_nom and s_nom
* The p_max/min limits of controllable branches are now p_max/min_pu
  per unit of s_nom, for consistency with generation and to allow
  unidirectional HVDCs / transport links for the capacity
  optimisation.
* network.remove() and io.import_series_from_dataframe() both take as
  argument class_name instead of list_name or the object - this is now
  fully consistent with network.add("Line","my line x").
* The booleans network.topology_determined and
  network.dependent_values_calculated have been totally removed - this
  was causing unexpected behaviour. Instead, to avoid repeated
  unnecessary calculations, the expert user can call functions with
  skip_pre=True.



v0.3.0 (27th January 2016)
===============================

In this release the pandas.Panel interface for time-dependent
variables was introduced. This replaced the manual attachment of
pandas.DataFrames per time-dependent variable as attributes of the
main component pandas.DataFrame.


Release process
===============

* Update ``doc/references/release-notes.rst``
* You don't need to update the version number anywhere, this is done automatically.
* ``git commit`` and put release notes in commit message
* ``git tag vx.x.x``
* ``git push`` and  ``git push --tags``
* The upload to `PyPI <https://pypi.org/>`_ is automated in the Github Action 
  ``deploy.yml``, which is triggered by pushing a tag.
  To upload manually, run ``python setup.py sdist``,
  then ``twine check dist/pypsa-0.x.0.tar.gz`` and
  ``twine upload dist/pypsa-0.x.0.tar.gz``
* The `GitHub release <https://github.com/PyPSA/PyPSA/releases>`_ is also automated in 
  the Github Action. Making a GitHub release will also trigger 
  `zenodo <https://zenodo.org/>`_ to archive the release with its own DOI.
* To update to conda-forge, check the pull request generated at the `feedstock repository
  <https://github.com/conda-forge/pypsa-feedstock>`_.
