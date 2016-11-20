###############
 Todo List
###############


Changes which definitely will be implemented
============================================


Allow standard types for lines and transformers
-----------------------------------------------

Standard types define typical electrical parameters for standard line
and transformer components. Example: For lines, the standard types
define the impedance per km, so that you only need to enter the
standard type and line length in order for all the electrical
parameters of the line to be defined.

We will probably follow the implementation in `pandapower
<https://www.uni-kassel.de/eecs/fachgebiete/e2n/software/pandapower.html>`_. The
translation into electrical parameters will take place in the function
``pf.calculate_dependent_values(network)``.



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


Allow elastic demand
--------------------

I.e. allow demand bid prices for blocks of demand.

As a work-around, dummy generators can be added to the nodes to
artificially reduce the demand beyond a certain price.


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

Similarly for generator.carrier

try:
network.buses.loc[branch.bus0]
except:
missing!
