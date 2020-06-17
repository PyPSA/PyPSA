## Demonstrate PyPSA unit commitment with a one-bus two-generator example
#
#
#This tutorial runs through examples of unit commitment for generators at a single bus. Examples of minimum part-load, minimum up time, minimum down time, start up costs, shut down costs and ramp rate restrictions are shown.
#
#To enable unit commitment on a generator, set its attribute committable = True.
#
#
#Available as a Jupyter notebook at http://www.pypsa.org/examples/unit-commitment.ipynb.

import pypsa

### Minimum part load demonstration
#
#In final hour load goes below part-load limit of coal gen (30%), forcing gas to commit.

nu = pypsa.Network()

nu.set_snapshots(range(4))

nu.add("Bus","bus")


nu.add("Generator","coal",bus="bus",
       committable=True,
       p_min_pu=0.3,
       marginal_cost=20,
       p_nom=10000)

nu.add("Generator","gas",bus="bus",
       committable=True,
       marginal_cost=70,
       p_min_pu=0.1,
       p_nom=1000)

nu.add("Load","load",bus="bus",p_set=[4000,6000,5000,800])

nu.lopf(nu.snapshots)

nu.generators_t.status

nu.generators_t.p

### Minimum up time demonstration
#
#Gas has minimum up time, forcing it to be online longer

nu = pypsa.Network()

nu.set_snapshots(range(4))

nu.add("Bus","bus")


nu.add("Generator","coal",bus="bus",
       committable=True,
       p_min_pu=0.3,
       marginal_cost=20,
       p_nom=10000)

nu.add("Generator","gas",bus="bus",
       committable=True,
       marginal_cost=70,
       p_min_pu=0.1,
       up_time_before=0,
       min_up_time=3,
       p_nom=1000)

nu.add("Load","load",bus="bus",p_set=[4000,800,5000,3000])

nu.lopf(nu.snapshots)

nu.generators_t.status

nu.generators_t.p

### Minimum down time demonstration
#
#Coal has a minimum down time, forcing it to go off longer.

nu = pypsa.Network()

nu.set_snapshots(range(4))

nu.add("Bus","bus")


nu.add("Generator","coal",bus="bus",
       committable=True,
       p_min_pu=0.3,
       marginal_cost=20,
       min_down_time=2,
       down_time_before=1,
       p_nom=10000)

nu.add("Generator","gas",bus="bus",
       committable=True,
       marginal_cost=70,
       p_min_pu=0.1,
       p_nom=4000)

nu.add("Load","load",bus="bus",p_set=[3000,800,3000,8000])

nu.lopf(nu.snapshots)

nu.objective

nu.generators_t.status

nu.generators_t.p

### Start up and shut down costs
#
#Now there are associated costs for shutting down, etc



nu = pypsa.Network()

nu.set_snapshots(range(4))

nu.add("Bus","bus")


nu.add("Generator","coal",bus="bus",
       committable=True,
       p_min_pu=0.3,
       marginal_cost=20,
       min_down_time=2,
       start_up_cost=5000,
       p_nom=10000)

nu.add("Generator","gas",bus="bus",
       committable=True,
       marginal_cost=70,
       p_min_pu=0.1,
       shut_down_cost=25,
       p_nom=4000)

nu.add("Load","load",bus="bus",p_set=[3000,800,3000,8000])

nu.lopf(nu.snapshots)

nu.objective

nu.generators_t.status

nu.generators_t.p

## Ramp rate limits

nu = pypsa.Network()

nu.set_snapshots(range(6))

nu.add("Bus","bus")


nu.add("Generator","coal",bus="bus",
       marginal_cost=20,
       ramp_limit_up=0.1,
       ramp_limit_down=0.2,
       p_nom=10000)

nu.add("Generator","gas",bus="bus",
       marginal_cost=70,
       p_nom=4000)

nu.add("Load","load",bus="bus",p_set=[4000,7000,7000,7000,7000,3000])

nu.lopf(nu.snapshots)

nu.generators_t.p

nu = pypsa.Network()

nu.set_snapshots(range(6))

nu.add("Bus","bus")


nu.add("Generator","coal",bus="bus",
       marginal_cost=20,
       ramp_limit_up=0.1,
       ramp_limit_down=0.2,
       p_nom_extendable=True,
       capital_cost=1e2)

nu.add("Generator","gas",bus="bus",
       marginal_cost=70,
       p_nom=4000)

nu.add("Load","load",bus="bus",p_set=[4000,7000,7000,7000,7000,3000])

nu.lopf(nu.snapshots)

nu.generators.p_nom_opt

nu.generators_t.p


nu = pypsa.Network()

nu.set_snapshots(range(7))

nu.add("Bus","bus")


#Can get bad interactions if SU > RU and p_min_pu; similarly if SD > RD


nu.add("Generator","coal",bus="bus",
       marginal_cost=20,
       committable=True,
       p_min_pu=0.05,
       initial_status=0,
       ramp_limit_start_up=0.1,
       ramp_limit_up=0.2,
       ramp_limit_down=0.25,
       ramp_limit_shut_down=0.15,
       p_nom=10000.)

nu.add("Generator","gas",bus="bus",
       marginal_cost=70,
       p_nom=10000)

nu.add("Load","load",bus="bus",p_set=[0.,200.,7000,7000,7000,2000,0])

nu.lopf(nu.snapshots)

nu.generators_t.p

nu.generators_t.status

nu.generators.loc["coal"]

## Rolling horizon example
#
#This example solves sequentially in batches

sets_of_snapshots = 6
p_set = [4000,5000,700,800,4000]

nu = pypsa.Network()

nu.set_snapshots(range(len(p_set)*sets_of_snapshots))

nu.add("Bus","bus")


nu.add("Generator","coal",bus="bus",
       committable=True,
       p_min_pu=0.3,
       marginal_cost=20,
       min_down_time=2,
       min_up_time=3,
       up_time_before=1,
       ramp_limit_up=1,
       ramp_limit_down=1,
       ramp_limit_start_up=1,
       ramp_limit_shut_down=1,
       shut_down_cost=150,
       start_up_cost=200,
       p_nom=10000)

nu.add("Generator","gas",bus="bus",
       committable=True,
       marginal_cost=70,
       p_min_pu=0.1,
       up_time_before=2,
       min_up_time=3,
       shut_down_cost=20,
       start_up_cost=50,
       p_nom=1000)

nu.add("Load","load",bus="bus",p_set=p_set*sets_of_snapshots)

overlap = 2
for i in range(sets_of_snapshots):
    nu.lopf(nu.snapshots[i*len(p_set):(i+1)*len(p_set)+overlap])

nu.generators_t.status

nu.generators_t.p


