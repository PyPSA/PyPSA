configfile: "config.yaml"

localrules: all, prepare_links_p_nom, base_network, add_electricity, add_sectors, extract_summaries, plot_network, scenario_comparions

wildcard_constraints:
    lv="[0-9\.]+",
    simpl="[a-zA-Z0-9]*",
    clusters="[0-9]+",
    sectors="[+a-zA-Z0-9]+",
    opts="[-+a-zA-Z0-9]+"

rule all:
    input: "results/summaries/costs2-summary.csv"

rule solve_all_elec_networks:
    input:
        expand("results/networks/elec_s{simpl}_{clusters}_lv{lv}_{opts}.nc",
               simpl='',
               clusters=config['scenario']['clusters'],
               lv='1.5',
               opts=config['scenario']['opts'])

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
    output: "networks/base.nc"
    benchmark: "benchmarks/base_network"
    threads: 1
    resources: mem_mb=500
    script: "scripts/base_network.py"

rule build_bus_regions:
    input:
        base_network="networks/base.nc"
    output:
        regions_onshore="resources/regions_onshore.geojson",
        regions_offshore="resources/regions_offshore.geojson"
    script: "scripts/build_bus_regions.py"

rule build_renewable_potentials:
    output: "resources/potentials_{technology}.nc"
    script: "scripts/build_renewable_potentials.py"

rule build_renewable_profiles:
    input:
        base_network="networks/base.nc",
        potentials="resources/potentials_{technology}.nc",
        regions=lambda wildcards: ("resources/regions_onshore.geojson"
                                   if wildcards.technology in ('onwind', 'solar')
                                   else "resources/regions_offshore.geojson")
    output:
        profile="resources/profile_{technology}.nc",
    script: "scripts/build_renewable_profiles.py"

rule build_hydro_profile:
    output: 'resources/profile_hydro.nc'
    script: 'scripts/build_hydro_profile.py'

rule add_electricity:
    input:
        base_network='networks/base.nc',
        tech_costs='data/costs/costs.csv',
        regions="resources/regions_onshore.geojson",
        **{'profile_' + t: "resources/profile_" + t + ".nc"
           for t in config['renewable']}
    output: "networks/elec.nc"
    benchmark: "benchmarks/add_electricity"
    threads: 1
    resources: mem_mb=1000
    script: "scripts/add_electricity.py"

rule simplify_network:
    input:
        network='networks/{network}.nc',
        regions_onshore="resources/regions_onshore.geojson",
        regions_offshore="resources/regions_offshore.geojson"
    output:
        network='networks/{network}_s{simpl}.nc',
        regions_onshore="resources/regions_onshore_{network}_s{simpl}.geojson",
        regions_offshore="resources/regions_offshore_{network}_s{simpl}.geojson"
    benchmark: "benchmarks/simplify_network/{network}_s{simpl}"
    threads: 1
    resources: mem_mb=1000
    script: "scripts/simplify_network.py"

rule cluster_network:
    input:
        network='networks/{network}_s{simpl}.nc',
        regions_onshore="resources/regions_onshore_{network}_s{simpl}.geojson",
        regions_offshore="resources/regions_offshore_{network}_s{simpl}.geojson"
    output:
        network='networks/{network}_s{simpl}_{clusters}.nc',
        regions_onshore="resources/regions_onshore_{network}_s{simpl}_{clusters}.geojson",
        regions_offshore="resources/regions_offshore_{network}_s{simpl}_{clusters}.geojson"
    benchmark: "benchmarks/cluster_network/{network}_s{simpl}_{clusters}"
    threads: 1
    resources: mem_mb=1000
    script: "scripts/cluster_network.py"

rule add_sectors:
    input:
        network="networks/elec_{cost}_{resarea}_{opts}.nc",
        emobility="data/emobility"
    output: "networks/sector_{cost}_{resarea}_{sectors}_{opts}.nc"
    benchmark: "benchmarks/add_sectors/sector_{resarea}_{sectors}_{opts}"
    threads: 1
    resources: mem_mb=1000
    script: "scripts/add_sectors.py"

rule prepare_network:
    input: 'networks/{network}_s{simpl}_{clusters}.nc'
    output: 'networks/{network}_s{simpl}_{clusters}_lv{lv}_{opts}.nc'
    threads: 1
    resources: mem_mb=1000
    script: "scripts/prepare_network.py"

rule solve_network:
    input: "networks/{network}_s{simpl}_{clusters}_lv{lv}_{opts}.nc"
    output: "results/networks/{network}_s{simpl}_{clusters}_lv{lv}_{opts}.nc"
    shadow: "shallow"
    log:
        gurobi="logs/{network}_s{simpl}_{clusters}_lv{lv}_{opts}_gurobi.log",
        python="logs/{network}_s{simpl}_{clusters}_lv{lv}_{opts}_python.log"
    benchmark: "benchmarks/solve_network/{network}_s{simpl}_{clusters}_lv{lv}_{opts}"
    threads: 4
    resources: mem_mb=lambda w: 100000 * int(w.clusters) // 362
    script: "scripts/solve_network.py"

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
