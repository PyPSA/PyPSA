#######################
Release Notes
#######################

PyPSA 0.7.1 (November 2016)
===========================

This release contains a bug fix, a minor new feature and more
warnings.

* The unix-only library ``resource`` is no longer imported by default,
  which was causing errors for Windows users.
* The ``Link`` attribute ``efficiency`` can now be make time-varying
  so that e.g. heat pump Coefficient of Performance (COP) can change
  over time due to ambient temperature variations (see the `heat pump
  example
  <http://www.pypsa.org/examples/power-to-heat-water-tank.html>`_).
* Bugs in the setting and getting of time-varying attributes for the
  object interface have been fixed.



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


All `examples <http://www.pypsa.org/examples/>`_ have been updated to
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
<http://www.pypsa.org/examples/#coupling-to-other-energy-sectors>`_). The
refactoring of time-dependent variable handling has been postponed
until the 0.7.0 release. In 0.7.0 the object interface to attributes
may also be removed; see below.

All `examples <http://www.pypsa.org/examples/>`_ have been updated to
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
  <http://www.pypsa.org/examples/replace-generator-storage-units-with-store.html>`_.

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
`<http://www.pypsa.org/examples/lopf-with-heating.html>`_.


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
