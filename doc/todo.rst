###############
 Todo List
###############


Changes which definitely will be implemented
============================================


.. _time-varying:

Redefine interface for time-varying parameters
----------------------------------------------

These changes will be implemented in the 0.7.0 release of PyPSA in the
late summer of 2016.

The problems: At the moment component attributes are either static
(like ``generator.p_nom`` for the generator nominal power stored in
the 2d pandas.DataFrame ``network.generators``) or time-dependent
series (like ``generator.p_max_pu`` for the generator available power
in the 3d pandas.Panel ``network.generators_t``). There is no
flexibility to have some items static and some time-dependent. The
current setup also involves generating all time-dependent variables
when the components are instantiated, even if they're not used, which
often leads to a giant pandas.Panel with many unnecessary parts
(such as ``p_set, q_set, q`` for the linear optimal power flow
(LOPF)). This is not very memory efficient. The pandas.Panel is
also buggy and will soon be deprecated in ``pandas`` in favour of
``xarray``.

The solution: For each variable that can be time-varying, such as
``generator.p_max_pu``, there will be in the component
pandas.DataFrame (``network.generators``) both a static attribute
``generator.p_max_pu`` as well as a static boolean switch
``generator.p_max_pu_t``. If the switch is False, the static attribute
``generator.p_max_pu`` in ``network.generators`` will be used for all
snapshots; if the switch is True, PyPSA will look in the DataFrame
``network.generators_t.p_max_pu`` for the value for the appropriate
snapshot.

The pandas.Panel ``network.generators_t`` will be replaced by a
dictionary of pandas.DataFrame (with the small modification that
dictionary values can also be accessed as attributes with
e.g. ``d.key`` - see ``pypsa.descriptors.Dict``). The
pandas.DataFrame, e.g. ``network.generators_t.p_max_pu`` will have an
index of the ``network.snapshots`` and a column of all components with
``generator.p_max_pu_t == True``. This is very similar to the
pandas.Panel, except that the components column can be of variable
length.

When running calculations, like the power flow or optimal power flow,
any missing series inputs will be filled in with the default values
(whether it is on a sub-network or a network). All series results
(such as ``generator.p,q``) will only be generated during the
calculation and will not be instantiated before the calculation is
run.


Open: What about object interface? Don't want a gen.p_max_pu to have
to check first the boolean, then return the static or varying?


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
