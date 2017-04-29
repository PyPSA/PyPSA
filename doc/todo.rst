###############
 Todo List
###############


Changes which definitely will be implemented
============================================



Improve regression testing
---------------------------

Use classes to do multiple tests with same set-up


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
================================


Reintroduce dynamically-generated object interface
--------------------------------------------------

e.g. ``network.get_objects("Bus")`` would return a list of objects
  with attributes linked to the corresponding ``pandas.DataFrame``

.. code:: python

    def __set__(self,attr,val):
        attr_type = self.network.components[self.__class__.__name]["attrs"]["typ"]

        try:
            val = attr_type(val)
        except:
            oops!

        df = self.network.df(self.__class__.list_name)

	if attr in df.columns:
            df.loc[self.name,attr] = val
        else:
            #return to normal object set
            setattr(self,attr,val)




Take v_mag_pu_set for PV from generators instead of bus?
--------------------------------------------------------

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
