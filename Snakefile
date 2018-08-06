configfile: "config.yaml"

localrules: all, prepare_links_p_nom, base_network, build_renewable_potentials, build_powerplants, add_electricity, add_sectors, prepare_network, extract_summaries, plot_network, scenario_comparions

wildcard_constraints:
    lv="[0-9\.]+",
    simpl="[a-zA-Z0-9]*",
    clusters="[0-9]+m?",
    sectors="[+a-zA-Z0-9]+",
    opts="[-+a-zA-Z0-9]*"

# rule all:
#     input: "results/summaries/costs2-summary.csv"

rule solve_all_elec_networks:
    input:
        expand("results/networks/elec_s{simpl}_{clusters}_lv{lv}_{opts}.nc",
               **config['scenario'])

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
        links_p_nom='data/links_p_nom.csv',
        country_shapes='resources/country_shapes.geojson',
        offshore_shapes='resources/offshore_shapes.geojson',
        europe_shape='resources/europe_shape.geojson'
    output: "networks/base.nc"
    benchmark: "benchmarks/base_network"
    threads: 1
    resources: mem_mb=500
    script: "scripts/base_network.py"

rule build_shapes:
    input:
        naturalearth='data/bundle/naturalearth/ne_10m_admin_0_countries.shp',
        eez='data/bundle/eez/World_EEZ_v8_2014.shp',
        nuts3='data/bundle/NUTS_2013_60M_SH/data/NUTS_RG_60M_2013.shp',
        nuts3pop='data/bundle/nama_10r_3popgdp.tsv.gz',
        nuts3gdp='data/bundle/nama_10r_3gdp.tsv.gz',
        ch_cantons='data/bundle/ch_cantons.csv',
        ch_popgdp='data/bundle/je-e-21.03.02.xls'
    output:
        country_shapes='resources/country_shapes.geojson',
        offshore_shapes='resources/offshore_shapes.geojson',
        europe_shape='resources/europe_shape.geojson',
        nuts3_shapes='resources/nuts3_shapes.geojson'
    threads: 1
    resources: mem_mb=500
    script: "scripts/build_shapes.py"

rule build_powerplants:
    input: base_network="networks/base.nc"
    output: "resources/powerplants.csv"
    threads: 1
    resources: mem_mb=500
    script: "scripts/build_powerplants.py"

rule build_bus_regions:
    input:
        country_shapes='resources/country_shapes.geojson',
        offshore_shapes='resources/offshore_shapes.geojson',
        base_network="networks/base.nc"
    output:
        regions_onshore="resources/regions_onshore.geojson",
        regions_offshore="resources/regions_offshore.geojson"
    resources: mem_mb=1000
    script: "scripts/build_bus_regions.py"

rule build_cutout:
    output: "cutouts/{cutout}"
    resources: mem_mb=5000
    threads: config['atlite'].get('nprocesses', 4)
    benchmark: "benchmarks/build_cutout_{cutout}"
    script: "scripts/build_cutout.py"

rule build_renewable_potentials:
    input:
        cutout=lambda wildcards: "cutouts/" + config["renewable"][wildcards.technology]['cutout'],
        corine="data/bundle/corine/g250_clc06_V18_5.tif",
        natura="data/bundle/natura/Natura2000_end2015.shp"
    output: "resources/potentials_{technology}.nc"
    resources: mem_mb=10000
    benchmark: "benchmarks/build_renewable_potentials_{technology}"
    script: "scripts/build_renewable_potentials.py"

rule build_renewable_profiles:
    input:
        base_network="networks/base.nc",
        potentials="resources/potentials_{technology}.nc",
        regions=lambda wildcards: ("resources/regions_onshore.geojson"
                                   if wildcards.technology in ('onwind', 'solar')
                                   else "resources/regions_offshore.geojson"),
        cutout=lambda wildcards: "cutouts/" + config["renewable"][wildcards.technology]['cutout']
    output:
        profile="resources/profile_{technology}.nc",
    resources: mem_mb=5000
    benchmark: "benchmarks/build_renewable_profiles_{technology}"
    script: "scripts/build_renewable_profiles.py"

rule build_hydro_profile:
    input:
        country_shapes='resources/country_shapes.geojson',
        eia_hydro_generation='data/bundle/EIA_hydro_generation_2000_2014.csv',
        cutout="cutouts/" + config["renewable"]['hydro']['cutout']
    output: 'resources/profile_hydro.nc'
    resources: mem_mb=5000
    script: 'scripts/build_hydro_profile.py'

rule add_electricity:
    input:
        base_network='networks/base.nc',
        tech_costs='data/costs.csv',
        regions="resources/regions_onshore.geojson",
        powerplants='resources/powerplants.csv',
        hydro_capacities='data/bundle/hydro_capacities.csv',
        opsd_load='data/bundle/time_series_60min_singleindex_filtered.csv',
        nuts3_shapes='resources/nuts3_shapes.geojson',
        **{'profile_' + t: "resources/profile_" + t + ".nc"
           for t in config['renewable']}
    output: "networks/elec.nc"
    benchmark: "benchmarks/add_electricity"
    threads: 1
    resources: mem_mb=3000
    script: "scripts/add_electricity.py"

rule simplify_network:
    input:
        network='networks/{network}.nc',
        regions_onshore="resources/regions_onshore.geojson",
        regions_offshore="resources/regions_offshore.geojson"
    output:
        network='networks/{network}_s{simpl}.nc',
        regions_onshore="resources/regions_onshore_{network}_s{simpl}.geojson",
        regions_offshore="resources/regions_offshore_{network}_s{simpl}.geojson",
        clustermaps='resources/clustermaps_{network}_s{simpl}.h5'
    benchmark: "benchmarks/simplify_network/{network}_s{simpl}"
    threads: 1
    resources: mem_mb=4000
    script: "scripts/simplify_network.py"

rule cluster_network:
    input:
        network='networks/{network}_s{simpl}.nc',
        regions_onshore="resources/regions_onshore_{network}_s{simpl}.geojson",
        regions_offshore="resources/regions_offshore_{network}_s{simpl}.geojson",
        clustermaps='resources/clustermaps_{network}_s{simpl}.h5'
    output:
        network='networks/{network}_s{simpl}_{clusters}.nc',
        regions_onshore="resources/regions_onshore_{network}_s{simpl}_{clusters}.geojson",
        regions_offshore="resources/regions_offshore_{network}_s{simpl}_{clusters}.geojson",
        clustermaps='resources/clustermaps_{network}_s{simpl}_{clusters}.h5'
    benchmark: "benchmarks/cluster_network/{network}_s{simpl}_{clusters}"
    threads: 1
    resources: mem_mb=3000
    script: "scripts/cluster_network.py"

# rule add_sectors:
#     input:
#         network="networks/elec_{cost}_{resarea}_{opts}.nc",
#         emobility="data/emobility"
#     output: "networks/sector_{cost}_{resarea}_{sectors}_{opts}.nc"
#     benchmark: "benchmarks/add_sectors/sector_{resarea}_{sectors}_{opts}"
#     threads: 1
#     resources: mem_mb=1000
#     script: "scripts/add_sectors.py"

rule prepare_network:
    input: 'networks/{network}_s{simpl}_{clusters}.nc'
    output: 'networks/{network}_s{simpl}_{clusters}_lv{lv}_{opts}.nc'
    threads: 1
    resources: mem_mb=1000
    benchmark: "benchmarks/prepare_network/{network}_s{simpl}_{clusters}_lv{lv}_{opts}"
    script: "scripts/prepare_network.py"

def partition(w):
    return 'vres' if memory(w) >= 60000 else 'x-men'

def memory(w):
    if w.clusters.endswith('m'):
        return 18000 + 180 * int(w.clusters[:-1])
    else:
        return 10000 + 190 * int(w.clusters)
        # return 4890+310 * int(w.clusters)

rule solve_network:
    input: "networks/{network}_s{simpl}_{clusters}_lv{lv}_{opts}.nc"
    output: "results/networks/{network}_s{simpl}_{clusters}_lv{lv}_{opts}.nc"
    shadow: "shallow"
    params: partition=partition
    log:
        gurobi="logs/{network}_s{simpl}_{clusters}_lv{lv}_{opts}_gurobi.log",
        python="logs/{network}_s{simpl}_{clusters}_lv{lv}_{opts}_python.log",
        memory="logs/{network}_s{simpl}_{clusters}_lv{lv}_{opts}_memory.log"
    benchmark: "benchmarks/solve_network/{network}_s{simpl}_{clusters}_lv{lv}_{opts}"
    threads: 4
    resources:
        mem_mb=memory,
        x_men=lambda w: 1 if partition(w) == 'x-men' else 0,
        vres=lambda w: 1 if partition(w) == 'vres' else 0
    script: "scripts/solve_network.py"

def partition_op(w):
    return 'vres' if memory_op(w) >= 60000 else 'x-men'

def memory_op(w):
    return 5000 + 372 * int(w.clusters)

rule solve_operations_network:
    input:
        unprepared="networks/{network}_s{simpl}_{clusters}.nc",
        optimized="results/networks/{network}_s{simpl}_{clusters}_lv{lv}_{opts}.nc"
    output: "results/networks/{network}_s{simpl}_{clusters}_lv{lv}_{opts}_op.nc"
    shadow: "shallow"
    params: partition=partition_op
    log:
        gurobi="logs/solve_operations_network/{network}_s{simpl}_{clusters}_lv{lv}_{opts}_op_gurobi.log",
        python="logs/solve_operations_network/{network}_s{simpl}_{clusters}_lv{lv}_{opts}_op_python.log",
        memory="logs/solve_operations_network/{network}_s{simpl}_{clusters}_lv{lv}_{opts}_op_memory.log"
    benchmark: "benchmarks/solve_operations_network/{network}_s{simpl}_{clusters}_lv{lv}_{opts}"
    threads: 4
    resources:
        mem_mb=memory_op,
        x_men=lambda w: 1 if partition_op(w) == 'x-men' else 0,
        vres=lambda w: 1 if partition_op(w) == 'vres' else 0
    script: "scripts/solve_operations_network.py"

rule plot_network:
    input:
        network='results/networks/{cost}_{resarea}_{sectors}_{opts}.nc',
        supply_regions='data/supply_regions/supply_regions.shp',
        resarea=lambda w: config['data']['resarea'][w.resarea]
    output:
        'results/plots/network_{cost}_{resarea}_{sectors}_{opts}_{attr}.pdf'
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

# rule scenario_comparison:
#     input:
#         expand('results/plots/network_{cost}_{sectors}_{opts}_{attr}.pdf',
#                version=config['version'],
#                attr=['p_nom'],
#                **config['scenario'])
#     output:
#        html='results/plots/scenario_{param}.html'
#     params:
#        tmpl="network_[cost]_[resarea]_[sectors]_[opts]_[attr]",
#        plot_dir='results/plots'
#     script: "scripts/scenario_comparison.py"

# rule extract_summaries:
#     input:
#         expand("results/networks/{cost}_{sectors}_{opts}.nc",
#                **config['scenario'])
#     output:
#         **{n: "results/summaries/{}-summary.csv".format(n)
#            for n in ['costs', 'costs2', 'e_curtailed', 'e_nom_opt', 'e', 'p_nom_opt']}
#     params:
#         scenario_tmpl="[cost]_[resarea]_[sectors]_[opts]",
#         scenarios=config['scenario']
#     script: "scripts/extract_summaries.py"


# Local Variables:
# mode: python
# End:
