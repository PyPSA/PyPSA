## Example of linear optimal power flow with coupling to heating sector
#
#In this example three locations are optimised, each with an electric bus and a heating bus and corresponding loads. At each location the electric and heating buses are connected with heat pumps; heat can also be supplied to the heat bus with a boiler. The electric buses are connected with transmission lines and there are electrical generators at two of the nodes.
#
#Available as a Jupyter notebook at <http://www.pypsa.org/examples/lopf-with-heating.ipynb>.

import pypsa

network = pypsa.Network()

#add three buses of AC and heat carrier each
for i in range(3):
    network.add("Bus","electric bus {}".format(i),v_nom=20.)
    network.add("Bus","heat bus {}".format(i),carrier="heat")

print(network.buses)

network.buses["carrier"].value_counts()

#add three lines in a ring
for i in range(3):
    network.add("Line","line {}".format(i),
                bus0="electric bus {}".format(i),
                bus1="electric bus {}".format((i+1)%3),
                x=0.1,
                s_nom=1000)

print(network.lines)

#connect the electric to the heat buses with heat pumps with COP 3.
for i in range(3):
    network.add("Link",
                "heat pump {}".format(i),
                bus0="electric bus {}".format(i),
                bus1="heat bus {}".format(i),
                p_nom=100,
                efficiency=3.)

print(network.links)

#add carriers
network.add("Carrier","gas",
           co2_emissions=0.27)
network.add("Carrier","biomass",
           co2_emissions=0.)

print(network.carriers)


#add a gas generator at bus 0
network.add("Generator","gas generator",
            bus="electric bus 0",
            p_nom=100,
            marginal_cost=50,
            carrier="gas",
            efficiency=0.3)

#add a biomass generator at bus 1
network.add("Generator","biomass generator",
            bus="electric bus 1",
            p_nom=100,
            marginal_cost=100,
            efficiency=0.3,
            carrier="biomass")

#add a boiler at all heat buses
for i in range(3):
    network.add("Generator","boiler {}".format(i),
            bus="heat bus {}".format(i),
            p_nom=1000,
            efficiency=0.9,
            marginal_cost=20.,
            carrier="gas")

print(network.generators)

#add electric loads
for i in range(3):
    network.add("Load","electric load {}".format(i),
                bus="electric bus {}".format(i),
                p_set=i*10)

#add heat loads
for i in range(3):
    network.add("Load","heat load {}".format(i),
                bus="heat bus {}".format(i),
                p_set=(3-i)*10)

print(network.loads)

print(network.loads.p_set)

#function for the LOPF

def run_lopf():
    network.lopf(keep_files=True)
    print("Objective:",network.objective)
    print(network.generators_t.p)
    print(network.links_t.p0)
    print(network.loads_t.p)

run_lopf()

#rerun with marginal costs for the heat pump operation

network.links.marginal_cost = 10
run_lopf()

#rerun with no CO2 emissions

network.add("GlobalConstraint",
            "co2_limit",
            sense="<=",
            constant=0.)

run_lopf()

