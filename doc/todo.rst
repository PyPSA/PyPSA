###############
 Todo List
###############


Changes which definitely will be implemented
============================================



Do not define empty timeseries contents until called
----------------------------------------------------


At the moment all time-varying series are instantiated at startup in
e.g. ``network.generators_t.q_set`` even if they're not used - this
is memory inefficient.

Could instantiate them as set, then check before PF and OPF if they're
there.

Results need only to be instantiated if things are actually
calculated, e.g. instantiate ``network.buses_t.q`` only after a
non-linear power flow.

Or have switch network.memory_saving, so that all items generated
automatically for newbies, and experts can turn it off and only
generate those which they need.

This requires the descriptors to be replaced with __get__ and __set__,
since some time-dependent quantities will not exist in the _t dataframes.

Replace pandas.Panels with xarray.DataArray
-------------------------------------------

This offers better interface, forces same datatype on
e.g. network.generators_t (float).

pandas.Panel can be buggy.



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



Regions for groups of buses
---------------------------

I.e. countries/states to allow easy grouping.

class Zone/Region


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

Option to separate optimisation of max state of charge from nominal power for storage
-------------------------------------------------------------------------------------

For storage units, the maximum state of charge is currently fixed by
the "max_hours" variable multiplied by the nominal power "p_nom"
("p_nom" can be optimised). It would be nice to include the option to
de-link p_nom and state_of_charge_max and optimise them separately
with separate costs.



Include transformer tap ratio and phase shift for trafos in linear pf
---------------------------------------------------------------------



Branch voltage angle difference limits in LOPF
----------------------------------------------

Reactive-power-constrained Power Flow
-------------------------------------

If a PV bus exceeds Q-limits, convert it to PQ at the limit.

Include zero-impedance switch/breaker component
-----------------------------------------------

Connects two buses with zero impedance and can be either on or off. Would have no p0/1 or q0/1 or any time dependence (apart perhaps from the swtich on/off status?).


Introduced "active" switch/boolean for each component
-----------------------------------------------------

To allow easy deactivation of components without full removal.


Include heating sector
----------------------

Along the lines of abstraction in oemof, include heat buses, with heat
loads, gas boilers, CHP (with output to both heat and electricity
buses), P2H, heat pumps, etc.

Allow elastic demand
--------------------

I.e. allow demand bid prices for blocks of demand


Changes which may be implemented
============================================


Take v_mag_pu_set for PV from generators instead of bus?
-----------------------------------------------------

Like pypower

Would imitate set point on AVR

Thermal limits: i_nom or s_nom?
-------------------------------

At the moment PyPSA inherits the behaviour of PYPOWER and MATPOWER to
take all branch thermal limits in terms of apparent power in MVA as
branch.s_nom. This makes sense for transformers, but less so for
transmission lines, where the limit should properly be on the current
in kA as branch.i_nom. However, the only place where the limit is used
in calculation is for the linear OPF, where it is assumed anyway that
voltage is 1 p.u. and it is more convenient to have limits on the
power there. This is the logic behind using branch.s_nom.

At some point the option may be introduced to have branch.i_nom limits
on lines.



Storing component object methods in different files
---------------------------------------------------

want different files, but still have tab completion and ? and ?? magic

over-ride __dir__???

cf. pandas code

best to do in __init__.


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


Check branch.bus0 and branch.bus1 in network.buses
--------------------------------------------------

Similarly for generator.source

try:
network.buses.loc[branch.bus0]
except:
missing!
