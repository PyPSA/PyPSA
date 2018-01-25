## Demonstration of Link with multiple outputs: Combined-Heat-and-Power (CHP) with fixed heat-power ratio
#
#For a CHP with a more complicated heat-power feasible operational area, see https://www.pypsa.org/examples/power-to-gas-boiler-chp.html.
#
#This example demonstrates a Link component with more than one bus output ("bus2" in this case). In general links can have many output buses.
#
#In this example a CHP must be heat-following because there is no other supply of heat to the bus "Frankfurt heat".

import pypsa, numpy as np

#First tell PyPSA that links will have a 2nd bus by
#overriding the component_attrs. This can be done for
#as many buses as you need with format busi for i = 2,3,4,5,....

override_component_attrs = pypsa.descriptors.Dict({k : v.copy() for k,v in pypsa.components.component_attrs.items()})
override_component_attrs["Link"].loc["bus2"] = ["string",np.nan,np.nan,"2nd bus","Input (optional)"]
override_component_attrs["Link"].loc["efficiency2"] = ["static or series","per unit",1.,"2nd bus efficiency","Input (optional)"]
override_component_attrs["Link"].loc["p2"] = ["series","MW",0.,"2nd bus output","Output"]


network = pypsa.Network(override_component_attrs=override_component_attrs)

network.add("Bus",
            "Frankfurt",
            carrier="AC")

network.add("Load",
            "Frankfurt",
            bus="Frankfurt",
            p_set=5)

network.add("Bus",
            "Frankfurt heat",
            carrier="heat")

network.add("Load",
            "Frankfurt heat",
            bus="Frankfurt heat",
            p_set=3)

network.add("Bus",
            "Frankfurt gas",
            carrier="gas")

network.add("Store",
            "Frankfurt gas",
            e_initial=1e6,
            e_nom=1e6,
            bus="Frankfurt gas")

network.add("Link",
            "OCGT",
            bus0="Frankfurt gas",
            bus1="Frankfurt",
            p_nom_extendable=True,
            capital_cost=600,
            efficiency=0.4)


network.add("Link",
            "CHP",
            bus0="Frankfurt gas",
            bus1="Frankfurt",
            bus2="Frankfurt heat",
            p_nom_extendable=True,
            capital_cost=1400,
            efficiency=0.3,
            efficiency2=0.3)

network.lopf()

network.loads_t.p

network.links_t.p0

network.links_t.p1

network.links_t.p2

