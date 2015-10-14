

#library with non-Europe-specific data generation functionality
import generate

#library with Europe-specific data generation
import europe.generate


network = europe.generate.build_entsoe(spatial_zoom=hundred_nodes)



results_folder = time_stamp + scenario_name


model = network.lopf(index_set=representative_hours)

#at this point can add custom constraints

model.solve(solver="Gurobi",options=["IBM"])

network.save_results(result_folder)


network.perform_standard_analysis()
