#######################
Release Notes
#######################

Upcoming Release
================

.. warning:: The features listed below are not released yet, but will be part of the next release! To use the features already you have to install the ``master`` branch, e.g. ``pip install git+https://github.com/pypsa/pypsa#egg=pypsa``.

* The representation of networks was modified to show the number of components and snapshots.
* Accept ``pathlib.Path`` objects when importing networks with ``pypsa.Network()``.
* Add example notebook on how to model redispatch with PyPSA.

PyPSA 0.20.0 (26th July 2022)
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


PyPSA 0.19.3 (22nd April 2022)
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

PyPSA 0.19.2 (7th March 2022)
=============================

* Add standard line type for 750 kV transmission line.

PyPSA 0.19.1 (18th February 2022)
=================================

* When setting ramp limits for links and calling ``Network.lopf`` with ``pyomo=False``, an unexpected KeyError was raised. This was fixed by correctly accessing the data frame referring to the power dispatch of links.


PyPSA 0.19.0 (11th February 2022)
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

PyPSA 0.18.1 (15th October 2021)
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


PyPSA 0.18.0 (12th August 2021)
===============================

This release contains new features for pathway optimisation, improvements of the
documentation's examples section as well as compatibility and bug fixes.

**Licensing**

* With this release, we have changed the licence from the copyleft GPLv3
  to the more liberal MIT licence with the consent of all contributors
  (for the reasoning why, see the `pull request
  <https://github.com/PyPSA/PyPSA/pull/274>`_).

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
  documentation. See :doc:`examples-basic`, :doc:`examples-lopf`,
  :doc:`examples-sector_coupling` and :doc:`examples-other`.


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


PyPSA 0.17.1 (15th July 2020)
=============================

This release contains bug fixes and extensions to the features for optimization when not using Pyomo.

* N-1 security-constrained linear optimal power flow is now also supported without pyomo by running ``network.sclopf(pyomo=False)``.

* Added support for the FICO Xpress commercial solver for optimization withhout pyomo, i.e. ``pyomo=False``.

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

PyPSA 0.17.0 (23rd March 2020)
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
      using the function :func:`pypsa.linopt.write_objective`.
      When a pure custom objective function needs to be declared,
      one can set ``skip_objective=True``.
      In this case, only terms defined through ``extra_functionality``
      will be considered in the objective function.
    * Shadow prices of capacity bounds for non-extendable passive branches
      are parsed (similar to the ``pyomo=True`` setting)
    * Fixed :func:`pypsa.linopf.define_kirchhoff_constraints` to handle
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
* The function :func:`pypsa.linopf.ilopf` can now track the intermediate branch capacities
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


PyPSA 0.16.1 (10th January 2020)
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
  <https://pypsa.readthedocs.io/en/latest/optimal_power_flow.html#pypsa.Network.lopf>`_.
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


PyPSA 0.16.0 (20th December 2019)
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
  <https://pypsa.readthedocs.io/en/latest/optimal_power_flow.html#pyomo-is-set-to-false>`_
  for the new syntax as well as a `Jupyter notebook of examples
  <https://github.com/PyPSA/PyPSA/blob/master/examples/lopf_with_pyomo_False.ipynb>`_.

* Distributed active power slack is now implemented for the full
  non-linear power flow. If you pass ``network.pf()`` the argument
  ``distribute_slack=True``, it will distribute the slack power across
  generators proportional to generator dispatch by default, or
  according to the distribution scheme provided in the argument
  ``slack_weights``. If ``distribute_slack=False`` only the slack
  generator takes up the slack. There is further `documentation
  <https://pypsa.readthedocs.io/en/latest/power_flow.html#full-non-linear-power-flow>`__.

* Unit testing is now performed on all of GNU/Linux, Windows and MacOS.

* NB: You may need to update your version of the package ``six``.

Special thanks for this release to Fabian Hofmann for implementing the
nomopyomo framework in PyPSA and Fabian Neumann for providing the
customizable distributed slack.


PyPSA 0.15.0 (8th November 2019)
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
  <https://pypsa.readthedocs.io/en/latest/optimal_power_flow.html#generator-unit-commitment-constraints>`_
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


PyPSA 0.14.1 (27th May 2019)
================================

This minor release contains three small bug fixes:

* Documentation parses now correctly on PyPI
* Python 2.7 and 3.6 are automatically tested using Travis
* PyPSA on Python 2.7 was fixed

This will also be the first release to be available directly from
`conda-forge <https://conda-forge.org/>`_.

PyPSA 0.14.0 (15th May 2019)
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

PyPSA 0.13.2 (10th January 2019)
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


PyPSA 0.13.1 (27th March 2018)
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



PyPSA 0.13.0 (25th January 2018)
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
  files, introduced in PyPSA 0.12.0, is now deprecated.
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




PyPSA 0.12.0 (30th November 2017)
=================================

This release contains new features and bug fixes.

* Support for Pyomo's persistent solver interface, so if you're making
  small changes to an optimisation model (e.g. tweaking a parameter),
  you don't have to rebuild the model every time. To enable this,
  ``network_lopf`` has been internally split into ``build_model``,
  ``prepare_solver`` and ``solve`` to allow more fine-grained control of the
  solving steps.  Currently the new Pyomo PersistentSolver interface
  is not in the main Pyomo branch, see
  the `pull request <https://github.com/Pyomo/pyomo/pull/223>`_; you can obtain it with
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


PyPSA 0.11.0 (21st October 2017)
================================

This release contains new features but no changes to existing APIs.

* There is a new function ``network.iplot()`` which creates an
  interactive plot in Jupyter notebooks using the `plotly
  <https://plot.ly/python/>`_ library. This reveals bus and branch
  properties when the mouse hovers over them and allows users to
  easily zoom in and out on the network. See the (sparse) documentation
  :doc:`plotting`.
* There is a new function ``network.madd()`` for adding multiple new
  components to the network. This is significantly faster than
  repeatedly calling ``network.add()`` and uses the functions
  ``network.import_components_from_dataframe()`` and
  ``network.import_series_from_dataframe()`` internally. Documentation
  and examples can be found at :ref:`madd`.
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


PyPSA 0.10.0 (7th August 2017)
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



PyPSA 0.9.0 (29th April 2017)
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
  Optimal Power Flow (LOPF) are now documented in :ref:`formulations`
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


PyPSA 0.8.0 (25th January 2017)
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
  ``pypsa/component_attrs/``. You can access these CSVs in the code
  via the dictionary ``network.components``,
  e.g. ``network.components["Line"]["attrs"]`` will show a pandas
  DataFrame with all attributes and their types, defaults, units and
  descriptions.  These CSVs are also sourced for the documentation in
  :doc:`components`, so the documentation will always be up-to-date.
* All examples have been updated appropriately.




PyPSA 0.7.1 (26th November 2016)
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





PyPSA 0.7.0 (20th November 2016)
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
  components are well defined; see :doc:`troubleshooting`.


All `examples <https://pypsa.readthedocs.io/en/latest/examples-basic.html>`_ have been updated to
accommodate the changes listed below.


PyPSA 0.6.2 (4th November 2016)
===============================

This release fixes a single library dependency issue:

* pf: A single line has been fixed so that it works with new pandas
  versions >= 0.19.0.

We thank Thorben Meiners for promptly pointing out this issue with the
new versions of pandas.


PyPSA 0.6.1 (25th August 2016)
==============================

This release fixes a single critical bug:

* opf: The latest version of Pyomo (4.4.1) had a bad interaction with
  pandas when a pandas.Index was used to index variables. To fix this,
  the indices are now cast to lists; compatibility with less recent
  versions of Pyomo is also retained.

We thank Joao Gorenstein Dedecca for promptly notifying us of this
bug.



PyPSA 0.6.0 (23rd August 2016)
==============================

Like the 0.5.0 release, this release contains API changes, which
complete the integration of sector coupling. You may have to update
your old code. Models for Combined Heat and Power (CHP) units, heat
pumps, resistive Power-to-Heat (P2H), Power-to-Gas (P2G), battery
electric vehicles (BEVs) and chained hydro reservoirs can now be built
(see the `sector coupling examples
<https://pypsa.readthedocs.io/en/latest/examples-sector_coupling.html>`_). The
refactoring of time-dependent variable handling has been postponed
until the 0.7.0 release. In 0.7.0 the object interface to attributes
may also be removed; see below.

All `examples <https://pypsa.readthedocs.io/en/latest/examples-basic.html>`_ have been updated to
accommodate the changes listed below.

Sector coupling
---------------

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

Downgrading object interface
----------------------------

The intention is to have only the pandas DataFrame interface for
accessing component attributes, to make the code simpler. The
automatic generation of objects with descriptor access to attributes
may be removed altogether.

* examples: Patterns of for loops through ``network.components.obj`` have
  been removed.

* components: The methods on ``Bus`` like ``bus.generators()`` and
  ``bus.loads()`` have been removed.

* components: ``network.add()`` no longer returns the object.

Other
-----

* components, opf: Unlimited upper bounds for
  e.g. ``generator.p_nom_max`` or ``line.s_nom_max`` were previous set
  using ``np.nan``; now they are set using ``float("inf")`` which is
  more logical. You may have to update your old code accordingly.

* components: A memory leak whereby references to
  ``component.network`` were not being correctly deleted has been
  fixed.



PyPSA 0.5.0 (21st July 2016)
============================

This is a relatively major release with some API changes, primarily
aimed at allowing coupling with other energy carriers (heat, gas,
etc.). The specification for a change and refactoring to the handling
of time series has also been prepared (see :ref:`time-varying`), which will
be implemented in the next major release PyPSA 0.6.0 in the late
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
  methods, which are more performant the the pure python methods of
  networkx. The discovery of network connected components is now
  significantly faster.

* io: The function ``network.export_to_csv_folder()`` has been
  rewritten to only export non-default values of static and series
  component attributes. Static and series attributes of all components
  are not exported if they are default values.  The functionality to
  selectively export series has been removed from the export function,
  because it was clumsy and hard to use.  See :ref:`export-csv` for
  more details.


* plot: Plotting networks is now more performant (using matplotlib
  LineCollections) and allows generic branches to be plotted, not just
  lines.

* test: Unit testing for Security-Constrained Linear Optimal Power
  Flow (SCLOPF) has been introduced.


PyPSA 0.4.2 (17th June 2016)
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


PyPSA 0.4.1 (3rd April 2016)
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


PyPSA 0.4.0 (21st March 2016)
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


PyPSA 0.3.3 (29th February 2016)
================================

Additional features:

* ``network.lpf`` can be called on an iterable of ``snapshots``
  i.e. ``network.lpf(snapshots)``, which is more performant that
  calling ``network.lpf`` on each snapshot separately.
* Bug fix on import/export of transformers and shunt impedances (which
  were left out before).
* Refactoring of some internal code.
* Better network clustering.


PyPSA 0.3.2 (17th February 2016)
================================

In this release some minor API changes were made:

* The Newton-Raphson tolerance ``network.nr_x_tol`` was moved to being
  an argument of the function ``network.pf(x_tol=1e-6)`` instead. This
  makes more sense and is then available in the docstring of
  ``network.pf``.
* Following similar reasoning ``network.opf_keep_files`` was moved to
  being an argument of the function
  ``network.lopf(keep_files=False)``.


PyPSA 0.3.1 (7th February 2016)
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



PyPSA 0.3.0 (27th January 2016)
===============================

In this release the pandas.Panel interface for time-dependent
variables was introduced. This replaced the manual attachment of
pandas.DataFrames per time-dependent variable as attributes of the
main component pandas.DataFrame.


Release process
===============

* Update ``doc/release_notes.rst``
* Update version in ``setup.py``, ``doc/conf.py``, ``pypsa/__init__.py``
* ``git commit`` and put release notes in commit message
* ``git tag v0.x.0``
* ``git push`` and  ``git push --tags``
* The upload to `PyPI <https://pypi.org/>`_ is automated in the Github Action ``deploy.yml``.
  To upload manually, run ``python setup.py sdist``,
  then ``twine check dist/pypsa-0.x.0.tar.gz`` and
  ``twine upload dist/pypsa-0.x.0.tar.gz``
* To update to conda-forge, check the pull request generated at the `feedstock repository
  <https://github.com/conda-forge/pypsa-feedstock>`_.
* Making a `GitHub release <https://github.com/PyPSA/PyPSA/releases>`_
  will trigger `zenodo <https://zenodo.org/>`_ to archive the release
  with its own DOI.
* Inform the PyPSA mailing list.
