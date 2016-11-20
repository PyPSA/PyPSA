## Power to Gas Example with Optional Coupling to Heat Sector (via Boiler OR Combined-Heat-and-Power (CHP))
#
#A location has an electric, gas and heat bus. The primary source is wind power, which can be converted to gas. The gas can be stored to convert into electricity or heat (with either a boiler or a CHP).

import pypsa
import pandas as pd

from pyomo.environ import Constraint

heat = True
chp = True


network = pypsa.Network()

network.set_snapshots(pd.date_range("2016-01-01 00:00","2016-01-01 03:00",freq="H"))

network.add("Bus",
            "0",
            carrier="AC")

network.add("Bus",
            "0 gas",
            carrier="gas")

network.add("Carrier",
            "wind")

network.add("Carrier",
            "gas",
            co2_emissions=0.9)


network.add("Generator",
            "wind turbine",
            bus="0",
            carrier="wind",
            p_nom_extendable=True,
            p_max_pu=[0.,0.2,0.7,0.4],
            capital_cost=500)

network.add("Load",
            "load",
            bus="0",
            p_set=5.)



network.add("Link",
            "P2G",
            bus0="0",
            bus1="0 gas",
            efficiency=0.3,
            capital_cost=1000,
            p_nom_extendable=True)

network.add("Link",
            "generator",
            bus0="0 gas",
            bus1="0",
            efficiency=0.4,
            capital_cost=200,
            p_nom_extendable=True)


network.add("Store",
            "gas depot",
            bus="0 gas",
            e_cyclic=True,
            e_nom_extendable=True)


if heat:
    
    network.add("Bus",
            "0 heat",
            carrier="heat")
    
    network.add("Carrier",
               "heat")

    network.add("Load",
            "heat load",
            bus="0 heat",
            p_set=10.)

    network.add("Link",
            "boiler",
            bus0="0 gas",
            bus1="0 heat",
            efficiency=0.9,
            capital_cost=100,
            p_nom_extendable=True)
    
    network.add("Store",
            "water tank",
            bus="0 heat",
            e_cyclic=True,
            e_nom_extendable=True)    


if heat and chp:
    
    #follows http://www.ea-energianalyse.dk/reports/student-reports/integration_of_50_percent_wind%20power.pdf pages 35-6
    #ratio between max heat output and max electric output
    nom_r = 1.
        
    #backpressure limit
    c_m = 0.75
        
    #marginal loss for each additional generation of heat
    c_v = 0.15
            
    network.links.at["boiler","efficiency"] = network.links.at["generator","efficiency"]/c_v
    
    def extra_functionality(network,snapshots):

        network.model.chp_nom = Constraint(rule=lambda model : nom_r*model.link_p_nom["generator"] == model.link_p_nom["boiler"])

        def backpressure(model,snapshot):
            return c_m*model.link_p["boiler",snapshot] <= model.link_p["generator",snapshot] 
        
        network.model.backpressure = Constraint(list(snapshots),rule=backpressure)
        
else:
    extra_functionality = None

network.co2_limit=0.
network.lopf(network.snapshots, extra_functionality=extra_functionality)
print("Objective:",network.objective)

print(pd.DataFrame({attr: network.stores_t[attr]["gas depot"] for attr in ["p","e"]}))

if heat:
    print(pd.DataFrame({attr: network.stores_t[attr]["water tank"] for attr in ["p","e"]}))
    print(pd.DataFrame({attr: network.links_t[attr]["boiler"] for attr in ["p0","p1"]}))

print(network.stores.loc["gas depot"])

print(network.generators.loc["wind turbine"])

print(network.links.p_nom_opt)

#Calculate the overall efficiency of the CHP

eta_elec = 0.4
c_v = 0.15
c_m = 0.75

r = 1/c_m

#P_h = r*P_e

print((1+r)/((1/eta_elec)*(1+c_v*r)))

