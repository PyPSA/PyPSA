## Simple electricity market examples
#
#This Jupyter notebook is meant for teaching purposes. To use it, you need to install a Python environment with Jupyter notebooks, and the Python for Power System Analysis (PyPSA) library. See
#
#https://pypsa.org/doc/installation.html
#
#for tips on installation.
#
#It gradually builds up more and more complicated energy-only electricity markets in PyPSA, starting from a single bidding zone, going up to multiple bidding zones connected with transmission (NTCs) along with variable renewables and storage.
#
#Available as a Jupyter notebook at http://www.pypsa.org/examples/simple-electricity-market-examples.ipynb.

### Preliminaries
#
#Here libraries are imported and data is defined.

import pypsa, numpy as np

#marginal costs in EUR/MWh
marginal_costs = {"Wind" : 0,
                  "Hydro" : 0,
                  "Coal" : 30,
                  "Gas" : 60,
                  "Oil" : 80}

#power plant capacities (nominal powers in MW) in each country (not necessarily realistic)
power_plant_p_nom = {"South Africa" : {"Coal" : 35000,
                                       "Wind" : 3000,
                                       "Gas" : 8000,
                                       "Oil" : 2000
                                      },
                     "Mozambique" : {"Hydro" : 1200,
                                    },
                     "Swaziland" : {"Hydro" : 600,
                                    },
                    }

#transmission capacities in MW (not necessarily realistic)
transmission = {"South Africa" : {"Mozambique" : 500,
                                  "Swaziland" : 250},
                "Mozambique" : {"Swaziland" : 100}}

#country electrical loads in MW (not necessarily realistic)
loads = {"South Africa" : 42000,
         "Mozambique" : 650,
         "Swaziland" : 250}

### Single bidding zone with fixed load, one period
#
#In this example we consider a single market bidding zone, South Africa.
#
#The inelastic load has essentially infinite marginal utility (or higher than the marginal cost of any generator).

country = "South Africa"

network = pypsa.Network()

network.add("Bus",country)

for tech in power_plant_p_nom[country]:
    network.add("Generator",
                "{} {}".format(country,tech),
                bus=country,
                p_nom=power_plant_p_nom[country][tech],
                marginal_cost=marginal_costs[tech])


network.add("Load",
            "{} load".format(country),
            bus=country,
            p_set=loads[country])

#Run optimisation to determine market dispatch
network.lopf()

#print the load active power (P) consumption
network.loads_t.p

#print the generator active power (P) dispatch
network.generators_t.p

#print the clearing price (corresponding to gas)
network.buses_t.marginal_price

### Two bidding zones connected by transmission, one period
#
#In this example we have bidirectional transmission capacity between two bidding zones. The power transfer is treated as controllable (like an A/NTC (Available/Net Transfer Capacity) or HVDC line). Note that in the physical grid, power flows passively according to the network impedances.

network = pypsa.Network()

countries = ["Mozambique", "South Africa"]

for country in countries:
    
    network.add("Bus",country)

    for tech in power_plant_p_nom[country]:
        network.add("Generator",
                    "{} {}".format(country,tech),
                    bus=country,
                    p_nom=power_plant_p_nom[country][tech],
                    marginal_cost=marginal_costs[tech])


    network.add("Load",
                "{} load".format(country),
                bus=country,
                p_set=loads[country])
    
    #add transmission as controllable Link
    if country not in transmission:
        continue
    
    for other_country in countries:
        if other_country not in transmission[country]:
            continue
        
        #NB: Link is by default unidirectional, so have to set p_min_pu = -1
        #to allow bidirectional (i.e. also negative) flow
        network.add("Link",
                   "{} - {} link".format(country, other_country),
                   bus0=country,
                   bus1=other_country,
                   p_nom=transmission[country][other_country],
                   p_min_pu=-1)

network.lopf()

network.loads_t.p

network.generators_t.p

network.links_t.p0

#print the clearing price (corresponding to water in Mozambique and gas in SA)
network.buses_t.marginal_price

#link shadow prices
network.links_t.mu_lower

### Three bidding zones connected by transmission, one period
#
#In this example we have bidirectional transmission capacity between three bidding zones. The power transfer is treated as controllable (like an A/NTC (Available/Net Transfer Capacity) or HVDC line). Note that in the physical grid, power flows passively according to the network impedances.

network = pypsa.Network()

countries = ["Swaziland", "Mozambique", "South Africa"]

for country in countries:
    
    network.add("Bus",country)

    for tech in power_plant_p_nom[country]:
        network.add("Generator",
                    "{} {}".format(country,tech),
                    bus=country,
                    p_nom=power_plant_p_nom[country][tech],
                    marginal_cost=marginal_costs[tech])


    network.add("Load",
                "{} load".format(country),
                bus=country,
                p_set=loads[country])
    
    #add transmission as controllable Link
    if country not in transmission:
        continue
    
    for other_country in countries:
        if other_country not in transmission[country]:
            continue
        
        #NB: Link is by default unidirectional, so have to set p_min_pu = -1
        #to allow bidirectional (i.e. also negative) flow
        network.add("Link",
                   "{} - {} link".format(country, other_country),
                   bus0=country,
                   bus1=other_country,
                   p_nom=transmission[country][other_country],
                   p_min_pu=-1)

network.lopf()

network.loads_t.p

network.generators_t.p

network.links_t.p0

#print the clearing price (corresponding to hydro in S and M, and gas in SA)
network.buses_t.marginal_price

#link shadow prices
network.links_t.mu_lower

### Single bidding zone with price-sensitive industrial load, one period
#
#In this example we consider a single market bidding zone, South Africa.
#
#Now there is a large industrial load with a marginal utility which is low enough to interact with the generation marginal cost.

country = "South Africa"

network = pypsa.Network()

network.add("Bus",country)

for tech in power_plant_p_nom[country]:
    network.add("Generator",
                "{} {}".format(country,tech),
                bus=country,
                p_nom=power_plant_p_nom[country][tech],
                marginal_cost=marginal_costs[tech])

#standard high marginal utility consumers
network.add("Load",
            "{} load".format(country),
            bus=country,
            p_set=loads[country])

#add an industrial load as a dummy negative-dispatch generator with marginal utility of 70 EUR/MWh for 8000 MW
network.add("Generator",
            "{} industrial load".format(country),
            bus=country,
            p_max_pu=0,
            p_min_pu=-1,
            p_nom=8000,
            marginal_cost=70)

network.lopf()

network.loads_t.p

#NB only half of industrial load is served, because this maxes out 
#Gas. Oil is too expensive with a marginal cost of 80 EUR/MWh
network.generators_t.p

network.buses_t.marginal_price

### Single bidding zone with fixed load, several periods
#
#In this example we consider a single market bidding zone, South Africa.
#
#We consider multiple time periods (labelled [0,1,2,3]) to represent variable wind generation.

country = "South Africa"

network = pypsa.Network()

#snapshots labelled by [0,1,2,3]
network.set_snapshots(range(4))

network.add("Bus",country)

#p_max_pu is variable for wind
for tech in power_plant_p_nom[country]:
    network.add("Generator",
                "{} {}".format(country,tech),
                bus=country,
                p_nom=power_plant_p_nom[country][tech],
                marginal_cost=marginal_costs[tech],
                p_max_pu=([0.3,0.6,0.4,0.5] if tech == "Wind" else 1),
                )

#load which varies over the snapshots
network.add("Load",
            "{} load".format(country),
            bus=country,
            p_set=loads[country] + np.array([0,1000,3000,4000]))

#specify that we consider all snapshots
network.lopf(network.snapshots)

network.loads_t.p

network.generators_t.p

network.buses_t.marginal_price

### Single bidding zone with fixed load and storage, several periods
#
#In this example we consider a single market bidding zone, South Africa.
#
#We consider multiple time periods (labelled [0,1,2,3]) to represent variable wind generation. Storage is allowed to do price arbitrage to reduce oil consumption.

country = "South Africa"

network = pypsa.Network()

#snapshots labelled by [0,1,2,3]
network.set_snapshots(range(4))

network.add("Bus",country)

#p_max_pu is variable for wind
for tech in power_plant_p_nom[country]:
    network.add("Generator",
                "{} {}".format(country,tech),
                bus=country,
                p_nom=power_plant_p_nom[country][tech],
                marginal_cost=marginal_costs[tech],
                p_max_pu=([0.3,0.6,0.4,0.5] if tech == "Wind" else 1),
                )

#load which varies over the snapshots
network.add("Load",
            "{} load".format(country),
            bus=country,
            p_set=loads[country] + np.array([0,1000,3000,4000]))

#storage unit to do price arbitrage
network.add("StorageUnit",
            "{} pumped hydro".format(country),
            bus=country,
            p_nom=1000,
            max_hours=6, #energy storage in terms of hours at full power
           )

network.lopf(network.snapshots)

network.loads_t.p

network.generators_t.p

network.storage_units_t.p

network.storage_units_t.state_of_charge

network.buses_t.marginal_price

