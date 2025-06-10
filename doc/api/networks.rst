########
Networks
########


Network
======== 

Constructor
~~~~~~~~~~~~

.. autoclass:: pypsa.Network

General methods
~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: _source/

    ~pypsa.Network.read_in_default_standard_types
    ~pypsa.Network.static
    ~pypsa.Network.df
    ~pypsa.Network.dynamic
    ~pypsa.Network.pnl
    ~pypsa.Network.to_crs
    ~pypsa.Network.set_snapshots
    ~pypsa.Network.set_investment_periods
    ~pypsa.Network.add
    ~pypsa.Network.madd
    ~pypsa.Network.remove
    ~pypsa.Network.mremove
    ~pypsa.Network.copy
    ~pypsa.Network.equals
    ~pypsa.Network.branches
    ~pypsa.Network.passive_branches
    ~pypsa.Network.controllable_branches
    ~pypsa.Network.determine_network_topology
    ~pypsa.Network.iterate_components
    ~pypsa.Network.consistency_check

Attributes
~~~~~~~~~~~
.. autosummary::
    :toctree: _source/

    ~pypsa.Network.meta
    ~pypsa.Network.crs
    ~pypsa.Network.srid
    ~pypsa.Network.snapshots
    ~pypsa.Network.snapshot_weightings
    ~pypsa.Network.investment_periods
    ~pypsa.Network.investment_period_weightings



Input and output methods
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: _source/

    ~pypsa.Network.merge
    ~pypsa.Network.import_from_csv_folder
    ~pypsa.Network.export_to_csv_folder
    ~pypsa.Network.import_from_excel
    ~pypsa.Network.export_to_excel
    ~pypsa.Network.import_from_hdf5
    ~pypsa.Network.export_to_hdf5
    ~pypsa.Network.import_from_netcdf
    ~pypsa.Network.export_to_netcdf
    ~pypsa.Network.import_from_pypower_ppc
    ~pypsa.Network.import_from_pandapower_net
    ~pypsa.Network.import_components_from_dataframe
    ~pypsa.Network.import_series_from_dataframe 

Power flow methods
~~~~~~~~~~~~~~~~~~~
Also see :doc:`power-flow` for all power flow functions.

.. autosummary::
    :toctree: _source/

    ~pypsa.Network.calculate_dependent_values
    ~pypsa.Network.lpf
    ~pypsa.Network.pf

Contingency analysis
~~~~~~~~~~~~~~~~~~~~~
See :doc:`contingency`.

Clustering methods
~~~~~~~~~~~~~~~~~~~~~
See :doc:`clustering`.

Optimization methods
~~~~~~~~~~~~~~~~~~~~~
See :doc:`optimization`.

Statistics methods
~~~~~~~~~~~~~~~~~~~~~
See :doc:`statistics`.

Plotting methods
~~~~~~~~~~~~~~~~~~~~~
See :doc:`plots`.

Graph methods
~~~~~~~~~~~~~
.. autosummary::
    :toctree: _source/

    ~pypsa.Network.graph
    ~pypsa.Network.adjacency_matrix
    ~pypsa.Network.incidence_matrix

Descriptor methods
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: _source/

    ~pypsa.Network.get_committable_i
    ~pypsa.Network.get_extendable_i
    ~pypsa.Network.get_switchable_as_dense
    ~pypsa.Network.get_non_extendable_i
    ~pypsa.Network.get_active_assets


Sub-Network
============

Constructor
~~~~~~~~~~~~

.. autoclass:: pypsa.SubNetwork

methods
~~~~~~~~

.. autosummary::
    :toctree: _source/
    
    ~pypsa.SubNetwork.adjacency_matrix
    ~pypsa.SubNetwork.branches
    ~pypsa.SubNetwork.branches_i
    ~pypsa.SubNetwork.buses
    ~pypsa.SubNetwork.buses_i
    ~pypsa.SubNetwork.calculate_BODF
    ~pypsa.SubNetwork.calculate_B_H
    ~pypsa.SubNetwork.calculate_PTDF
    ~pypsa.SubNetwork.calculate_Y
    ~pypsa.SubNetwork.find_bus_controls
    ~pypsa.SubNetwork.find_slack_bus
    ~pypsa.SubNetwork.generators
    ~pypsa.SubNetwork.generators_i
    ~pypsa.SubNetwork.graph
    ~pypsa.SubNetwork.incidence_matrix
    ~pypsa.SubNetwork.iterate_components
    ~pypsa.SubNetwork.lines_i
    ~pypsa.SubNetwork.loads
    ~pypsa.SubNetwork.loads_i
    ~pypsa.SubNetwork.lpf
    ~pypsa.SubNetwork.pf
    ~pypsa.SubNetwork.shunt_impedances
    ~pypsa.SubNetwork.shunt_impedances_i
    ~pypsa.SubNetwork.storage_units
    ~pypsa.SubNetwork.storage_units_i
    ~pypsa.SubNetwork.stores_i
    ~pypsa.SubNetwork.transformers_i
