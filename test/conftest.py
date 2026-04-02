# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

import pypsa
from pypsa.constants import DEFAULT_EPSG


@pytest.fixture(autouse=True)
def _set_test_options():
    """Ensure test-specific options are set before each test."""
    pypsa.options.debug.runtime_verification = True
    pypsa.options.params.optimize.include_objective_constant = True
    return


@pytest.fixture(autouse=True)
def close_matplotlib_figures():
    """Close all matplotlib figures before and after each test."""
    plt.close("all")
    yield
    plt.close("all")


def pytest_addoption(parser):
    """Add custom pytest command line options."""
    parser.addoption(
        "--new-components-api",
        action="store_true",
        default=False,
        help="Activate the new components API (options.api.new_components_api)",
    )
    parser.addoption(
        "--test-docs",
        action="store_true",
        default=False,
        help="Run documentation tests (doctest tests)",
    )
    parser.addoption(
        "--fix-notebooks",
        action="store_true",
        default=False,
        help="Auto-fix notebook issues found during validation (self-healing mode)",
    )


def pytest_configure(config):
    """Configure pytest session with custom options."""
    if config.getoption("--new-components-api"):
        pypsa.options.api.new_components_api = True


COMPONENT_NAMES = [
    "sub_networks",
    "buses",
    "carriers",
    "global_constraints",
    "lines",
    "line_types",
    "transformers",
    "transformer_types",
    "links",
    "loads",
    "generators",
    "storage_units",
    "stores",
    "shunt_impedances",
    "shapes",
]


@pytest.fixture(params=COMPONENT_NAMES)
def component_name(request):
    return request.param


# Example Networks


@pytest.fixture
def ac_dc_network():
    return pypsa.examples.ac_dc_meshed()


@pytest.fixture
def storage_hvdc_network():
    return pypsa.examples.storage_hvdc()


@pytest.fixture
def scigrid_de_network():
    return pypsa.examples.scigrid_de()


@pytest.fixture
def model_energy_network():
    return pypsa.examples.model_energy()


@pytest.fixture
def stochastic_network():
    return pypsa.examples.stochastic_network()


# AC-DC-Meshed types


@pytest.fixture
def ac_dc_solved():
    n = pypsa.examples.ac_dc_meshed()
    n.optimize()
    del n.model.solver_model
    return n


@pytest.fixture
def ac_dc_periods(ac_dc_network):
    n = ac_dc_network
    n.snapshots = pd.MultiIndex.from_product([[2013], n.snapshots])
    n.investment_periods = [2013]
    gens_i = n.c.generators.static.index
    rng = np.random.default_rng()  # Create a random number generator
    n.c.generators.dynamic.p[gens_i] = rng.random(size=(len(n.snapshots), len(gens_i)))
    return n


@pytest.fixture
def ac_dc_stochastic():
    n = pypsa.examples.ac_dc_meshed()
    n.set_scenarios({"low": 0.3, "high": 0.7})
    return n


AC_DC_MESHED_TYPES = [
    "ac_dc_network",
    "ac_dc_solved",
    "ac_dc_periods",
    "ac_dc_stochastic",
]


@pytest.fixture(params=AC_DC_MESHED_TYPES)
def ac_dc_types(request):
    return request.getfixturevalue(request.param)


# AC-DC-Meshed results


@pytest.fixture
def ac_dc_network_r():
    csv_folder = Path(__file__).parent / "data" / "ac-dc-meshed" / "results-lopf"
    return pypsa.Network(csv_folder)


@pytest.fixture
def ac_dc_stochastic_r(ac_dc_network_r):
    n = ac_dc_network_r.copy()
    n.set_scenarios({"low": 0.3, "high": 0.7})
    return n


@pytest.fixture
def ac_dc_shapes(ac_dc_network):
    n = ac_dc_network

    # Create bounding boxes around points
    def create_bbox(x, y, delta=0.1):
        return Polygon(
            [
                (x - delta, y - delta),
                (x - delta, y + delta),
                (x + delta, y + delta),
                (x + delta, y - delta),
            ]
        )

    bboxes = n.c.buses.static.apply(lambda row: create_bbox(row["x"], row["y"]), axis=1)

    # Convert to GeoSeries
    geo_series = gpd.GeoSeries(bboxes, crs=DEFAULT_EPSG)

    n.add(
        "Shape",
        name=geo_series.index,
        geometry=geo_series,
        idx=geo_series.index,
        component="Bus",
    )

    return n


# Other network fixtures


@pytest.fixture
def scipy_network():
    n = pypsa.examples.scigrid_de()
    n.c.generators.static.control = "PV"
    g = n.c.generators.static[n.c.generators.static.bus == "492"]
    n.c.generators.static.loc[g.index, "control"] = "PQ"
    n.calculate_dependent_values()
    n.determine_network_topology()
    return n


@pytest.fixture
def network_only_component_names():
    n = pypsa.Network()
    n.add("Bus", "bus1", x=0, y=0)
    n.add("Bus", "bus2", x=1, y=0)
    n.add("Bus", "bus3", x=0, y=1)
    # Add components with no extra data
    n.add("Carrier", "carrier1")
    n.add("Carrier", "carrier2")
    return n


# Other fixture collections
UNSOLVED_NETWORKS = [
    "ac_dc_network",
    "scigrid_de_network",
    "storage_hvdc_network",
    "model_energy_network",
    "stochastic_network",
    "network_only_component_names",
]

SOLVED_NETWORKS = [
    "ac_dc_solved",
]


@pytest.fixture(params=UNSOLVED_NETWORKS)
def networks(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(params=UNSOLVED_NETWORKS + SOLVED_NETWORKS)
def networks_including_solved(request):
    return request.getfixturevalue(request.param)


# Network collections


@pytest.fixture(scope="module")
def multiport_process_network():
    n = pypsa.Network()
    n.set_snapshots([0, 1, 2])
    for c in ["AC", "H2", "heat", "electrolyser"]:
        n.add("Carrier", c)
    n.add("Bus", "elec", carrier="AC")
    n.add("Bus", "h2", carrier="H2")
    n.add("Bus", "heat", carrier="heat")
    n.add("Generator", "gen", bus="elec", carrier="AC", p_nom=200, marginal_cost=10)
    n.add(
        "Process",
        "electrolyser",
        bus0="elec",
        bus1="h2",
        bus2="heat",
        carrier="electrolyser",
        rate0=-1,
        rate1=0.7,
        rate2=0.2,
        p_nom=100,
    )
    n.add("Load", "h2_load", bus="h2", carrier="H2", p_set=50)

    def _df(data):
        df = pd.DataFrame(data, index=n.snapshots)
        df.columns.name = "name"
        return df

    p = _df({"electrolyser": [80.0, 60.0, 70.0]})
    n.c.processes.dynamic["p"] = p
    n.c.processes.dynamic["p0"] = p.copy()
    n.c.processes.dynamic["p1"] = -p * 0.7
    n.c.processes.dynamic["p2"] = -p * 0.2
    n.c.generators.dynamic["p"] = _df({"gen": [130.0, 110.0, 120.0]})
    n.c.loads.dynamic["p"] = _df({"h2_load": [50.0, 50.0, 50.0]})
    n.c.buses.dynamic["marginal_price"] = _df(
        {
            "elec": [50.0, 40.0, 45.0],
            "h2": [100.0, 90.0, 95.0],
            "heat": [30.0, 25.0, 28.0],
        }
    )
    return n


@pytest.fixture
def network_collection(ac_dc_network_r):
    return pypsa.NetworkCollection(
        [ac_dc_network_r],
        index=pd.MultiIndex.from_tuples([("a", 2030)], names=["scenario", "year"]),
    )


# Pandapower networks
@pytest.fixture(scope="module")
def pandapower_custom_network():
    pp = pytest.importorskip("pandapower", reason="pandapower not installed")
    net = pp.create_empty_network()
    bus1 = pp.create_bus(net, vn_kv=20.0, name="Bus 1")
    bus2 = pp.create_bus(net, vn_kv=0.4, name="Bus 2")
    bus3 = pp.create_bus(net, vn_kv=0.4, name="Bus 3")
    # create bus elements
    pp.create_ext_grid(net, bus=bus1, vm_pu=1.02, name="Grid Connection")
    pp.create_load(net, bus=bus3, p_mw=0.100, q_mvar=0.05, name="Load")
    pp.create_shunt(net, bus=bus3, p_mw=0.0, q_mvar=0.0, name="Shunt")
    # create branch elements
    pp.create_transformer(
        net, hv_bus=bus1, lv_bus=bus2, std_type="0.4 MVA 20/0.4 kV", name="Trafo"
    )
    pp.create_line(
        net,
        from_bus=bus2,
        to_bus=bus3,
        length_km=0.1,
        std_type="NAYY 4x50 SE",
        name="Line",
    )
    return net


@pytest.fixture(scope="module")
def pandapower_cigre_network():
    try:
        import pandapower.networks as pn
    except ImportError:
        pytest.skip("pandapower not installed")
    return pn.create_cigre_network_mv(with_der="all")


# Complex stochastic network


@pytest.fixture
def stochastic_benchmark_network():
    """
    Create a network for benchmarking stochastic problems.
    This optimization problem is also uploaded to the pypsa examples repository
    with stochastic problem solved in two ways: out-of-the-box using PyPSA
    functionality and hardcoded using linopy.
    """
    # Configuration
    GAS_PRICE = 40  # Default scenario
    FREQ = "3h"
    LOAD_MW = 1
    TS_URL = "https://tubcloud.tu-berlin.de/s/pKttFadrbTKSJKF/download/time-series-lecture-2.csv"

    # Technology specs
    TECH = {
        "solar": {"profile": "solar", "inv": 1e6, "m_cost": 0.01},
        "wind": {"profile": "onwind", "inv": 2e6, "m_cost": 0.02},
        "gas": {"inv": 7e5, "eff": 0.6},
        "lignite": {"inv": 1.3e6, "eff": 0.4, "m_cost": 130},
    }
    FOM, DR, LIFE = 3.0, 0.03, 25

    for cfg in TECH.values():
        cfg["fixed_cost"] = (pypsa.costs.annuity(DR, LIFE) + FOM / 100) * cfg["inv"]

    # Load time series data from URL - same as in the original script
    ts = pd.read_csv(TS_URL, index_col=0, parse_dates=True).resample(FREQ).asfreq()

    n = pypsa.Network()
    n.set_snapshots(ts.index)
    n.snapshot_weightings = pd.Series(int(FREQ[:-1]), index=ts.index)

    n.add("Bus", "DE")
    n.add("Load", "DE_load", bus="DE", p_set=LOAD_MW)

    for tech in ["solar", "wind"]:
        cfg = TECH[tech]
        n.add(
            "Generator",
            tech,
            bus="DE",
            p_nom_extendable=True,
            p_max_pu=ts[cfg["profile"]],
            capital_cost=cfg["fixed_cost"],
            marginal_cost=cfg["m_cost"],
        )

    for tech in ["gas", "lignite"]:
        cfg = TECH[tech]
        mc = (GAS_PRICE / cfg["eff"]) if tech == "gas" else cfg["m_cost"]
        n.add(
            "Generator",
            tech,
            bus="DE",
            p_nom_extendable=True,
            efficiency=cfg["eff"],
            capital_cost=cfg["fixed_cost"],
            marginal_cost=mc,
        )
    # Set up scenarios
    n.set_scenarios({"low": 0.4, "medium": 0.3, "high": 0.3})

    return n
