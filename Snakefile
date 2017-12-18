configfile: "config.yaml"

localrules: all, prepare_links_p_nom, base_network, add_electricity, add_sectors, extract_summaries, plot_network, scenario_comparions

wildcard_constraints:
    resarea="[a-zA-Z0-9]+",
    cost="[-a-zA-Z0-9]+",
    sectors="[+a-zA-Z0-9]+",
    opts="[-+a-zA-Z0-9]+"

rule all:
    input: "results/version-{version}/summaries/costs2-summary.csv".format(version=config['version'])

rule prepare_links_p_nom:
    output: 'data/links_p_nom.csv'
    threads: 1
    resources: mem_mb=500
    script: 'scripts/prepare_links_p_nom.py'

rule base_network:
    input:
        eg_buses='data/entsoegridkit/buses.csv',
        eg_lines='data/entsoegridkit/lines.csv',
        eg_links='data/entsoegridkit/links.csv',
        eg_converters='data/entsoegridkit/converters.csv',
        eg_transformers='data/entsoegridkit/transformers.csv',
        parameter_corrections='data/parameter_corrections.yaml',
        links_p_nom='data/links_p_nom.csv'
    output: "networks/base_{opts}.h5"
    benchmark: "benchmarks/base_network_{opts}"
    threads: 1
    resources: mem_mb=500
    script: "scripts/base_network.py"

rule landuse_remove_protected_and_conservation_areas:
    input:
        landuse = "data/Original_UTM35north/sa_lcov_2013-14_gti_utm35n_vs22b.tif",
        protected_areas = "data/SAPAD_OR_2017_Q2/",
        conservation_areas = "data/SACAD_OR_2017_Q2/"
    output: "resources/landuse_without_protected_conservation.tiff"
    benchmark: "benchmarks/landuse_remove_protected_and_conservation_areas"
    threads: 1
    resources: mem_mb=10000
    script: "scripts/landuse_remove_protected_and_conservation_areas.py"

rule landuse_map_to_tech_and_supply_region:
    input:
        landuse = "resources/landuse_without_protected_conservation.tiff",
        supply_regions = "data/supply_regions/supply_regions.shp",
        resarea = lambda w: config['data']['resarea'][w.resarea]
    output:
        raster = "resources/raster_{tech}_percent_{resarea}.tiff",
        area = "resources/area_{tech}_{resarea}.csv"
    benchmark: "benchmarks/landuse_map_to_tech_and_supply_region/{tech}_{resarea}"
    threads: 1
    resources: mem_mb=17000
    script: "scripts/landuse_map_to_tech_and_supply_region.py"

rule inflow_per_country:
    input: EIA_hydro_gen="data/EIA_hydro_generation_2011_2014.csv"
    output: "resources/hydro_inflow.nc"
    benchmark: "benchmarks/inflow_per_country"
    threads: 1
    resources: mem_mb=1000
    script: "scripts/inflow_per_country.py"

rule add_electricity:
    input:
        base_network='networks/base_{opts}.h5',
        supply_regions='data/supply_regions/supply_regions.shp',
        load='data/SystemEnergy2009_13.csv',
        wind_pv_profiles='data/Wind_PV_Normalised_Profiles.xlsx',
        wind_area='resources/area_wind_{resarea}.csv',
        solar_area='resources/area_solar_{resarea}.csv',
        existing_generators="data/Existing Power Stations SA.xlsx",
        hydro_inflow="resources/hydro_inflow.csv",
        tech_costs="data/technology_costs.xlsx"
    output: "networks/elec_{cost}_{resarea}_{opts}.h5"
    benchmark: "benchmarks/add_electricity/elec_{resarea}_{opts}"
    threads: 1
    resources: mem_mb=1000
    script: "scripts/add_electricity.py"

rule add_sectors:
    input:
        network="networks/elec_{cost}_{resarea}_{opts}.h5",
        emobility="data/emobility"
    output: "networks/sector_{cost}_{resarea}_{sectors}_{opts}.h5"
    benchmark: "benchmarks/add_sectors/sector_{resarea}_{sectors}_{opts}"
    threads: 1
    resources: mem_mb=1000
    script: "scripts/add_sectors.py"

rule solve_network:
    input: network="networks/sector_{cost}_{resarea}_{sectors}_{opts}.h5"
    output: "results/version-{version}/networks/{{cost}}_{{resarea}}_{{sectors}}_{{opts}}.h5".format(version=config['version'])
    shadow: "shallow"
    log:
        gurobi="logs/{cost}_{resarea}_{sectors}_{opts}_gurobi.log",
        python="logs/{cost}_{resarea}_{sectors}_{opts}_python.log"
    benchmark: "benchmarks/solve_network/{cost}_{resarea}_{sectors}_{opts}"
    threads: 4
    resources: mem_mb=19000 # for electricity only
    script: "scripts/solve_network.py"

rule plot_network:
    input:
        network='results/version-{version}/networks/{{cost}}_{{resarea}}_{{sectors}}_{{opts}}.h5'.format(version=config['version']),
        supply_regions='data/supply_regions/supply_regions.shp',
        resarea=lambda w: config['data']['resarea'][w.resarea]
    output:
        only_map=touch('results/version-{version}/plots/network_{{cost}}_{{resarea}}_{{sectors}}_{{opts}}_{{attr}}'.format(version=config['version'])),
        ext=touch('results/version-{version}/plots/network_{{cost}}_{{resarea}}_{{sectors}}_{{opts}}_{{attr}}_ext'.format(version=config['version']))
    params: ext=['png', 'pdf']
    script: "scripts/plot_network.py"

# rule plot_costs:
#     input: 'results/summaries/costs2-summary.csv'
#     output:
#         expand('results/plots/costs_{cost}_{resarea}_{sectors}_{opt}',
#                **dict(chain(config['scenario'].items(), (('{param}')))
#         touch('results/plots/scenario_plots')
#     params:
#         tmpl="results/plots/costs_[cost]_[resarea]_[sectors]_[opt]"
#         exts=["pdf", "png"]
#     scripts: "scripts/plot_costs.py"

rule scenario_comparison:
    input:
        expand('results/version-{version}/plots/network_{cost}_{sectors}_{opts}_{attr}_ext',
               version=config['version'],
               attr=['p_nom'],
               **config['scenario'])
    output:
       html='results/version-{version}/plots/scenario_{{param}}.html'.format(version=config['version'])
    params:
       tmpl="network_[cost]_[resarea]_[sectors]_[opts]_[attr]_ext",
       plot_dir='results/version-{}/plots'.format(config['version'])
    script: "scripts/scenario_comparison.py"

rule extract_summaries:
    input:
        expand("results/version-{version}/networks/{cost}_{sectors}_{opts}.h5",
               version=config['version'],
               **config['scenario'])
    output:
        **{n: "results/version-{version}/summaries/{}-summary.csv".format(n, version=config['version'])
           for n in ['costs', 'costs2', 'e_curtailed', 'e_nom_opt', 'e', 'p_nom_opt']}
    params:
        scenario_tmpl="[cost]_[resarea]_[sectors]_[opts]",
        scenarios=config['scenario']
    script: "scripts/extract_summaries.py"


# Local Variables:
# mode: python
# End:
