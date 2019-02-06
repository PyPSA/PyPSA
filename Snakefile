configfile: "config.yaml"

COSTS="data/costs.csv"

wildcard_constraints:
    ll="(v|c)([0-9\.]+|opt|all)", # line limit, can be volume or cost
    simpl="[a-zA-Z0-9]*|all",
    clusters="[0-9]+m?|all",
    sectors="[+a-zA-Z0-9]+",
    opts="[-+a-zA-Z0-9]*"

rule cluster_all_elec_networks:
    input:
        expand("networks/elec_s{simpl}_{clusters}.nc",
               **config['scenario'])

rule prepare_all_elec_networks:
    input:
        expand("networks/elec_s{simpl}_{clusters}_l{ll}_{opts}.nc",
               **config['scenario'])

rule solve_all_elec_networks:
    input:
        expand("results/networks/elec_s{simpl}_{clusters}_l{ll}_{opts}.nc",
               **config['scenario'])

if config['enable']['prepare_links_p_nom']:
    rule prepare_links_p_nom:
        output: 'data/links_p_nom.csv'
        threads: 1
        resources: mem=500
        # group: 'nonfeedin_preparation'
        script: 'scripts/prepare_links_p_nom.py'

if config['enable']['powerplantmatching']:
    rule build_powerplants:
        input: base_network="networks/base.nc"
        output: "resources/powerplants.csv"
        threads: 1
        resources: mem=500
        # group: 'nonfeedin_preparation'
        script: "scripts/build_powerplants.py"

rule base_network:
    input:
        eg_buses='data/entsoegridkit/buses.csv',
        eg_lines='data/entsoegridkit/lines.csv',
        eg_links='data/entsoegridkit/links.csv',
        eg_converters='data/entsoegridkit/converters.csv',
        eg_transformers='data/entsoegridkit/transformers.csv',
        parameter_corrections='data/parameter_corrections.yaml',
        links_p_nom='data/links_p_nom.csv',
        links_tyndp='data/links_tyndp.csv',
        country_shapes='resources/country_shapes.geojson',
        offshore_shapes='resources/offshore_shapes.geojson',
        europe_shape='resources/europe_shape.geojson'
    output: "networks/base.nc"
    benchmark: "benchmarks/base_network"
    threads: 1
    resources: mem=500
    # group: 'nonfeedin_preparation'
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
    resources: mem=500
    # group: 'nonfeedin_preparation'
    script: "scripts/build_shapes.py"

rule build_bus_regions:
    input:
        country_shapes='resources/country_shapes.geojson',
        offshore_shapes='resources/offshore_shapes.geojson',
        base_network="networks/base.nc"
    output:
        regions_onshore="resources/regions_onshore.geojson",
        regions_offshore="resources/regions_offshore.geojson"
    resources: mem=1000
    # group: 'nonfeedin_preparation'
    script: "scripts/build_bus_regions.py"

rule build_cutout:
    output: "cutouts/{cutout}"
    resources: mem=config['atlite'].get('nprocesses', 4) * 1000
    threads: config['atlite'].get('nprocesses', 4)
    benchmark: "benchmarks/build_cutout_{cutout}"
    # group: 'feedin_preparation'
    script: "scripts/build_cutout.py"

rule build_natura_raster:
    input: "data/bundle/natura/Natura2000_end2015.shp"
    output: "resources/natura.tiff"
    script: "scripts/build_natura_raster.py"

rule build_renewable_profiles:
    input:
        base_network="networks/base.nc",
        corine="data/bundle/corine/g250_clc06_V18_5.tif",
        natura="resources/natura.tiff",
        gebco="data/bundle/GEBCO_2014_2D.nc",
        country_shapes='resources/country_shapes.geojson',
        offshore_shapes='resources/offshore_shapes.geojson',
        regions=lambda wildcards: ("resources/regions_onshore.geojson"
                                   if wildcards.technology in ('onwind', 'solar')
                                   else "resources/regions_offshore.geojson"),
        cutout=lambda wildcards: "cutouts/" + config["renewable"][wildcards.technology]['cutout']
    output: profile="resources/profile_{technology}.nc",
    resources: mem=config['atlite'].get('nprocesses', 2) * 5000
    threads: config['atlite'].get('nprocesses', 2)
    benchmark: "benchmarks/build_renewable_profiles_{technology}"
    # group: 'feedin_preparation'
    script: "scripts/build_renewable_profiles.py"

rule build_hydro_profile:
    input:
        country_shapes='resources/country_shapes.geojson',
        eia_hydro_generation='data/bundle/EIA_hydro_generation_2000_2014.csv',
        cutout="cutouts/" + config["renewable"]['hydro']['cutout']
    output: 'resources/profile_hydro.nc'
    resources: mem=5000
    # group: 'feedin_preparation'
    script: 'scripts/build_hydro_profile.py'

rule add_electricity:
    input:
        base_network='networks/base.nc',
        tech_costs=COSTS,
        regions="resources/regions_onshore.geojson",
        powerplants='resources/powerplants.csv',
        hydro_capacities='data/bundle/hydro_capacities.csv',
        geth_hydro_capacities='data/geth2015_hydro_capacities.csv',
        opsd_load='data/bundle/time_series_60min_singleindex_filtered.csv',
        nuts3_shapes='resources/nuts3_shapes.geojson',
        **{'profile_' + t: "resources/profile_" + t + ".nc"
           for t in config['renewable']}
    output: "networks/elec.nc"
    benchmark: "benchmarks/add_electricity"
    threads: 1
    resources: mem=3000
    # group: 'build_pypsa_networks'
    script: "scripts/add_electricity.py"

rule simplify_network:
    input:
        network='networks/{network}.nc',
        tech_costs=COSTS,
        regions_onshore="resources/regions_onshore.geojson",
        regions_offshore="resources/regions_offshore.geojson"
    output:
        network='networks/{network}_s{simpl}.nc',
        regions_onshore="resources/regions_onshore_{network}_s{simpl}.geojson",
        regions_offshore="resources/regions_offshore_{network}_s{simpl}.geojson",
        clustermaps='resources/clustermaps_{network}_s{simpl}.h5'
    benchmark: "benchmarks/simplify_network/{network}_s{simpl}"
    threads: 1
    resources: mem=4000
    # group: 'build_pypsa_networks'
    script: "scripts/simplify_network.py"

rule cluster_network:
    input:
        network='networks/{network}_s{simpl}.nc',
        regions_onshore="resources/regions_onshore_{network}_s{simpl}.geojson",
        regions_offshore="resources/regions_offshore_{network}_s{simpl}.geojson",
        clustermaps=ancient('resources/clustermaps_{network}_s{simpl}.h5')
    output:
        network='networks/{network}_s{simpl}_{clusters}.nc',
        regions_onshore="resources/regions_onshore_{network}_s{simpl}_{clusters}.geojson",
        regions_offshore="resources/regions_offshore_{network}_s{simpl}_{clusters}.geojson",
        clustermaps='resources/clustermaps_{network}_s{simpl}_{clusters}.h5'
    benchmark: "benchmarks/cluster_network/{network}_s{simpl}_{clusters}"
    threads: 1
    resources: mem=3000
    # group: 'build_pypsa_networks'
    script: "scripts/cluster_network.py"

# rule add_sectors:
#     input:
#         network="networks/elec_{cost}_{resarea}_{opts}.nc",
#         emobility="data/emobility"
#     output: "networks/sector_{cost}_{resarea}_{sectors}_{opts}.nc"
#     benchmark: "benchmarks/add_sectors/sector_{resarea}_{sectors}_{opts}"
#     threads: 1
#     resources: mem=1000
#     script: "scripts/add_sectors.py"

rule prepare_network:
    input: 'networks/{network}_s{simpl}_{clusters}.nc', tech_costs=COSTS
    output: 'networks/{network}_s{simpl}_{clusters}_l{ll}_{opts}.nc'
    threads: 1
    resources: mem=1000
    # benchmark: "benchmarks/prepare_network/{network}_s{simpl}_{clusters}_l{ll}_{opts}"
    script: "scripts/prepare_network.py"

def memory(w):
    factor = 3.
    for o in w.opts.split('-'):
        m = re.match(r'^(\d+)h$', o, re.IGNORECASE)
        if m is not None:
            factor /= int(m.group(1))
            break
    if w.clusters.endswith('m'):
        return int(factor * (18000 + 180 * int(w.clusters[:-1])))
    else:
        return int(factor * (10000 + 195 * int(w.clusters)))
        # return 4890+310 * int(w.clusters)

rule solve_network:
    input: "networks/{network}_s{simpl}_{clusters}_l{ll}_{opts}.nc"
    output: "results/networks/{network}_s{simpl}_{clusters}_l{ll}_{opts}.nc"
    shadow: "shallow"
    log:
        solver="logs/{network}_s{simpl}_{clusters}_l{ll}_{opts}_solver.log",
        python="logs/{network}_s{simpl}_{clusters}_l{ll}_{opts}_python.log",
        memory="logs/{network}_s{simpl}_{clusters}_l{ll}_{opts}_memory.log"
    benchmark: "benchmarks/solve_network/{network}_s{simpl}_{clusters}_l{ll}_{opts}"
    threads: 4
    resources: mem=memory
    # group: "solve" # with group, threads is ignored https://bitbucket.org/snakemake/snakemake/issues/971/group-job-description-does-not-contain
    script: "scripts/solve_network.py"

rule trace_solve_network:
    input: "networks/{network}_s{simpl}_{clusters}_l{ll}_{opts}.nc"
    output: "results/networks/{network}_s{simpl}_{clusters}_l{ll}_{opts}_trace.nc"
    shadow: "shallow"
    log: python="logs/{network}_s{simpl}_{clusters}_l{ll}_{opts}_python_trace.log",
    threads: 4
    resources: mem=memory
    script: "scripts/trace_solve_network.py"

rule solve_operations_network:
    input:
        unprepared="networks/{network}_s{simpl}_{clusters}.nc",
        optimized="results/networks/{network}_s{simpl}_{clusters}_l{ll}_{opts}.nc"
    output: "results/networks/{network}_s{simpl}_{clusters}_l{ll}_{opts}_op.nc"
    shadow: "shallow"
    log:
        solver="logs/solve_operations_network/{network}_s{simpl}_{clusters}_l{ll}_{opts}_op_solver.log",
        python="logs/solve_operations_network/{network}_s{simpl}_{clusters}_l{ll}_{opts}_op_python.log",
        memory="logs/solve_operations_network/{network}_s{simpl}_{clusters}_l{ll}_{opts}_op_memory.log"
    benchmark: "benchmarks/solve_operations_network/{network}_s{simpl}_{clusters}_l{ll}_{opts}"
    threads: 4
    resources: mem=(lambda w: 5000 + 372 * int(w.clusters))
    # group: "solve_operations"
    script: "scripts/solve_operations_network.py"

rule plot_network:
    input:
        network="results/networks/{network}_s{simpl}_{clusters}_l{ll}_{opts}.nc",
        tech_costs=COSTS
    output:
        only_map="results/plots/{network}_s{simpl}_{clusters}_l{ll}_{opts}_{attr}.{ext}",
        ext="results/plots/{network}_s{simpl}_{clusters}_l{ll}_{opts}_{attr}_ext.{ext}"
    script: "scripts/plot_network.py"

def input_make_summary(w):
    # It's mildly hacky to include the separate costs input as first entry
    return ([COSTS] +
            expand("results/networks/{network}_s{simpl}_{clusters}_l{ll}_{opts}.nc",
                   network=w.network,
                   **{k: config["scenario"][k] if getattr(w, k) == "all" else getattr(w, k)
                      for k in ["simpl", "clusters", "l", "opts"]}))

rule make_summary:
    input: input_make_summary
    output: directory("results/summaries/{network}_s{simpl}_{clusters}_l{ll}_{opts}_{country}")
    script: "scripts/make_summary.py"

rule plot_summary:
    input: "results/summaries/{network}_s{simpl}_{clusters}_l{ll}_{opts}_{country}"
    output: "results/plots/summary_{summary}_{network}_s{simpl}_{clusters}_l{ll}_{opts}_{country}.{ext}"
    script: "scripts/plot_summary.py"

def input_plot_p_nom_max(wildcards):
    return [('networks/{network}_s{simpl}{maybe_cluster}.nc'
             .format(maybe_cluster=('' if c == 'full' else ('_' + c)), **wildcards))
            for c in wildcards.clusters.split(",")]
rule plot_p_nom_max:
    input: input_plot_p_nom_max
    output: "results/plots/{network}_s{simpl}_cum_p_nom_max_{clusters}_{technology}_{country}.{ext}"
    script: "scripts/plot_p_nom_max.py"

# Local Variables:
# mode: python
# End:
