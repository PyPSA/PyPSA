# -*- coding: utf-8 -*-
import pypsa

n = pypsa.Network(
    "/home/philipp/Documents/01_Python/01_pypsa_repos/02_pypsa-eur/results/networks/elec_s_100_ec_lcopt_Co2L-24H.nc"
)
n.statistics.calculate_capex("test")

print("end")
