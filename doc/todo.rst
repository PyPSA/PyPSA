###############
 Todo List
###############


Changes which definitely will be implemented
============================================

Improve access to time-dependent variables
------------------------------------------

Currently time-dependent variables, such as load/generator p_set or
line p0 are accessed via pandas DataFrames attached as attributes to
the component DataFrame.

There are several problems with this:

* At the moment all time-varying series are instantiated at startup
  even if they're not used - this is memory inefficient
* You can't slice over attributes
* The attribute access on the component DataFrame is non-standard for
  pandas
* The time-varying attributes don't get copied over when doing
  selective slices of the component DataFrame,
  e.g. buses[buses.sub_network == "0"]


Suggestion:

Create a 3d component pandas.Panel for time-varying quantities, e.g.

.. code:: python

    network.generators_t = pandas.Panel(items=["p_set","p","q_set","q"],
                                        major_axis=network.snapshots,
					minor_axis=network.generators.index)



Then sub_network.generators_t can slice the Panel, similarly for bus.generators_t.


And only instantiate the items "p", "p_set", etc. when necessary,
e.g. at start of network.pf() can instantiate "p" and "q" if they
don't already exist.

Or have switch, so that all items generated automatically for newbies,
and experts can turn it off and only generate those which they need.



Replace descriptors with __get__ and __set__ on objects
-------------------------------------------------------

Can then use obj.attr for attr which are dynamically added to DataFrame

.. code:: python

    def __set__(self,attr,val):
        attr_type = self.__class__.attributes[attr]["type"]

        try:
            val = attr_type(val)
        except:
            oops!

        df = getattr(self.network,self.__class__.list_name)

	if attr in df.columns:
            df.loc[self.name,attr] = val
        else:

            #return to normal object set
            setattr(self,attr,val)


Store attributes in:

.. code:: python

    class Branch:

        static_attributes = {{}}

        series_attributes = {{}}



Improve regression testing
---------------------------

Use classes to do multiple tests with same set-up


Ramp rate limits in OPF for generators
--------------------------------------

i.e. generator.ramp_rate_limit = x MW/h or per unit of p_nom/h


Spillage variables for storage
-------------------------------

A variable per storage unit which can spill the state of charge
without generating electricity (e.g. if the inflow overwhelms the
storage)


Regions for groups of buses
---------------------------

I.e. countries/states to allow easy grouping.

class Zone/Region


Interface for adding different constraints/objective functions
--------------------------------------------------------------

I.e. other than rewriting lopf function.

Example: Yearly import/export balances for zones


More non-linear pf examples
---------------------------

pypower import, scigrid non-linear


Improve Python 3 support
------------------------

Check and regression testing


CIM converter
-------------

cf. Richard Lincoln's PyCIM



Newton-Raphson for DC networks
------------------------------

i.e. solve P_i = \sum_j V_i G_ij V_j

where everything is real

Can set either P or V at bus

Need one slack




Generic branch impedance component
----------------------------------

Include transformer tap ratio and phase shift for trafos in linear pf
---------------------------------------------------------------------



Branch voltage angle difference limits in LOPF
----------------------------------------------



OPF DC output to v_mag not v_ang
--------------------------------
Also make v_mag per unit NOT kV



Changes which may be implemented
============================================

Constant time series series
---------------------------

i.e. have some way of setting constant time series to save memory

Rename components and attributes?
---------------------------------

SubNetwork -> ConnectedNetwork

s_nom versus p_nom for lines/branches

Take v_mag_set for PV from generators instead of bus?
-----------------------------------------------------

ike pypower

Storing component object methods in different files
---------------------------------------------------

want different files, but still have tab completion and ? and ?? magic

over-ride __dir__???

cf. pandas code


make p_set per unit?
--------------------

Database interface with sqlalchemy?
-----------------------------------

Advantages of database:

#. better scaling with size
#. easier, better querying
#. persistence
#. can swop out database for Netzbetreiber
#. Sharing data between people editing concurrently
#. Transactions (e.g. bank account transfer that fails or succeeds always at both ends)
#. For relations between tables



catch no gens in sub_network?
-----------------------------

beware nx.MultiGraph reordering of edges!
-----------------------------------------

Orders them according to collections of edges between same nodes NOT
the order in which you read them in.

Kill inheritance?
-----------------

It doesn't serve any good purpose and just serves to confuse.

e.g. storage_unit inherits generator's efficiency, which doesn't make any sense.


need to watch out for isinstance(Branch)


Do not define empty timeseries contents until called, e.g.
-----------------------------------------------------------

network.generators_df.p = pd.DataFrame(index = network.snapshots)

network.generators_df.p.loc[1,"AT"] = 45.

- this will define a new column "AT" and add NaNs in other entries.

(at least for calculated quantities - p_set etc. should be defined)

give default if name not in col????



Underscore dynamically-generated DataFrames?
--------------------------------------------
Since they are NOT linked to original data for updating, and don't contain time-dependent quantities.

Check branch.bus0 and branch.bus1 in network.buses
--------------------------------------------------

Similarly for generator.source

try:
network.buses.loc[branch.bus0]
except:
missing!
