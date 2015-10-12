

#generateeurope does what regions/generate does

import generateeurope



network = generateeurope.build_entsoe(spatial_zoom=hundred_nodes)



results_folder = time_stamp + scenario_name


network.lopf(solver="Gurobi",options=["IBM"],index_set=representative_hours)

network.save_results()


network.perform_standard_analysis()
