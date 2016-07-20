#######################
Release Notes
#######################

PyPSA 0.5.0 (21st July 2016)
============================

This is a relatively major release with some API changes, primarily
aimed at allowing coupling with other energy carriers (heat, gas,
etc.). The specification for a change and refactoring to the handling
of time series has also been prepared (see :ref:`time-varying`), which will
be implemented in the next major release PyPSA 0.6.0 in the late
summer of 2016.

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
  to update your code to work with the new ``Link`` component.

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
