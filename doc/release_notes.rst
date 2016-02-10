#######################
Release Notes
#######################

PyPSA 0.3.1
===========

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



PyPSA 0.3
=========

In this release the pandas.Panel interface for time-dependent
variables was introduced. This replaced the manual attachment of
pandas.DataFrames per time-dependent variable as attributes of the
main component pandas.DataFrame.
