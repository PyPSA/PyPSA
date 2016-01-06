###############
 Todo List
###############

.. _todo:

Improve access to time-dependent variables
----------------------

Currently time-dependent variables, such as load/generator p_set or
line p0 are accessed via pandas DataFrames attached as attributes to
the component DataFrame.

There are several problems with this:

* At the moment all time-varying series are instantiated at startup
  even if they're not used - this is memory inefficient
* The attribute access on the component DataFrame is non-standard
* The time-varying attributes don't get copied over when doing
  selective slices of the component DataFrame,
  e.g. buses[buses.sub_network == "0"]
