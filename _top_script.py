import numpy as np
import os
import pyomo
import sys

import pypsa

network1 = pypsa.Network(name="network1")
print(network1.name)

solvername='glpk'
solverpath_folder='C:\\glpk\\w64' #does not need to be directly on c drive
solverpath_exe='C:\\glpk\\w64\\glpsol' #does not need to be directly on c drive

glpk_dir = r'C:\Program Files (x86)\Microsoft Visual Studio\Shared\glpk-4.65\w64'
solverpath_exe = os.path.join(glpk_dir, "glpsol.exe")

opt = pyomo.environ.SolverFactory('glpk')

# sys.path.append(solverpath_folder)

# solver=SolverFactory(solvername,executable=solverpath_exe)

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

network.lopf()

# network.lopf(snapshots=None,
        # pyomo=True,
        # solver_name='glpk',
        # solver_options={'executable': solverpath_exe},
        # solver_logfile=None,
        # formulation='kirchhoff',
        # keep_files=True
        # extra_functionality=None,
        # multi_investment_periods=False)