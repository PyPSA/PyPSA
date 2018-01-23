## Demonstration of Link with multiple outputs: Combined-Heat-and-Power (CHP) with fixed heat-power ratio
#
#For a CHP with a more complicated heat-power feasible operational area, see https://www.pypsa.org/examples/power-to-gas-boiler-chp.html.
#
#This example demonstrates a Link component with more than one bus output ("bus2" in this case). In general links can have many output buses.
#
#In this example a CHP must be heat-following because there is no other supply of heat to the bus "Frankfurt heat".

import pypsa


network = pypsa.Network()

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
            p_nom_extendable=True,
            capital_cost=1400,
            efficiency=0.3)

# Now add a second output bus for the CHP

network.links["bus2"] = ""
network.links.at["CHP","bus2"] = "Frankfurt heat"

network.links["efficiency2"] = 0.
network.links.at["CHP","efficiency2"] = 0.3

network.lopf()

network.loads_t.p

network.links_t.p0

network.links_t.p1

network.links_t.p2

