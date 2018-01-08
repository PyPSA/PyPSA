## Demonstrate PyPSA unit commitment with a one-bus two-generator example
#
#
#To enable unit commitment on a generator, set its attribute committable = True.
#
#
#Available as a Jupyter notebook at http://www.pypsa.org/examples/unit-commitment.ipynb.

import pypsa

from pypsa.opf import network_lopf_build_model as build_model

import argparse
import logging

parser = argparse.ArgumentParser()

parser.add_argument(
    '-v', '--verbose',
    help="Be verbose",
    action="store_const", dest="loglevel", const=logging.INFO,
)
parser.add_argument(
    '-d', '--debug',
    help="Print lots of debugging statements",
    action="store_const", dest="loglevel", const=logging.DEBUG,
    default=logging.WARNING,
)
args = parser.parse_args()
logging.basicConfig( level=args.loglevel)

### Minimum part load demonstration
#
#In final hour load goes below part-load limit of coal gen (30%), forcing gas to commit.

for test_i in range(1):

    snapshots = range( 1, 1001)
    p_set = [ p*20 for p in snapshots]

    nu = pypsa.Network()

    nu.set_snapshots(snapshots)

    nu.add("Bus","bus")

    generator_p_nom = [ i for i in range( 100)]
    generator_marginal_cost = [ 1000 - i for i in range(100) ]
    p_min_pu = .3

    for gen_i, gen in enumerate( generator_p_nom):
        nu.add( "Generator",
                "gas_" + str(gen_i),
                bus = "bus",
                committable = True,
                p_min_pu = p_min_pu,
                marginal_cost = generator_marginal_cost[gen_i],
                p_nom = generator_p_nom[gen_i],)

    nu.add("Load","load",bus="bus",p_set= p_set )#[4000,6000,5000,800])

    build_model( nu, nu.snapshots, formulation = "kirchoff")

    nu.generators_t.status

    nu.generators_t.p

    ### Minimum up time demonstration
    #
    #Gas has minimum up time, forcing it to be online longer

    nu = pypsa.Network()

    nu.set_snapshots( snapshots)
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
           initial_status=0,
           min_up_time=3,
           p_nom=1000)

    nu.add("Load","load",bus="bus",p_set= p_set )#[4000,800,5000,3000])

    build_model( nu, nu.snapshots, formulation = "kirchoff")
