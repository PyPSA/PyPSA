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
import random
import copy

random.seed( 55)

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

    snapshots = range( 1, 101)
    p_set = [ p*20 for p in snapshots]

    nu = pypsa.Network()

    nu.set_snapshots(snapshots)


    n_gen = 100

    generator_p_nom = [ i for i in range( n_gen)]
    generator_marginal_cost = [ 1000 - i for i in range( n_gen)]
    p_min_pu = .3

    for gen_i, gen in enumerate( generator_p_nom):
        nu.add("Bus","bus" + str(gen_i))
        nu.add("Load","load" + str(gen_i),bus= "bus" + str(gen_i),
               p_set= generator_p_nom[gen_i] / 2. )#[4000,6000,5000,800])

        nu.add( "Generator",
                "gas_" + str(gen_i),
                bus = "bus" + str(gen_i),
                committable = True,
                p_min_pu = p_min_pu,
                marginal_cost = generator_marginal_cost[gen_i],
                p_nom = generator_p_nom[gen_i],)
        if gen_i > 0:
            nu.add("Line",
                   "{} - {} line".format( gen_i - 1, gen_i),
                   bus0= "bus" + str(gen_i - 1),
                   bus1= "bus" + str(gen_i),
                   x=0.1,
                   r=0.01)
            random_gen_to_connect = random.randint( 0, n_gen - 1)
            nu.add("Line",
                   "{} - {} line".format( gen_i, random_gen_to_connect),
                   bus0= "bus" + str(random_gen_to_connect),
                   bus1= "bus" + str(gen_i),
                   x = .2,
                   r = .08)

    formulations = ["kirchhoff", "angles", "cycles"] #"ptdf"
    for formulation in formulations:
        nu_copy = copy.deepcopy(nu)
        build_model( nu, nu.snapshots, formulation = formulation)
#    build_model( nu, nu.snapshots, formulation = "angles")
#    nu.lopf( nu.snapshots, formulation = "kirchhoff")
