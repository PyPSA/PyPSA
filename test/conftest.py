from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

import pypsa
from pypsa.constants import DEFAULT_EPSG


def pytest_addoption(parser):
    parser.addoption(
        "--test-docs-build",
        action="store_true",
        default=False,
        help="run sphinx build test",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "test_sphinx_build: mark test as sphinx build")


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


@pytest.fixture
def ac_dc_network():
    return pypsa.examples.ac_dc_meshed()


@pytest.fixture
def storage_hvdc_network():
    return pypsa.examples.storage_hvdc()


@pytest.fixture
def model_energy_network():
    return pypsa.examples.model_energy()


@pytest.fixture
def n():
    return pypsa.examples.ac_dc_meshed()


@pytest.fixture
def ac_dc_meshed_stoch():
    n = pypsa.examples.ac_dc_meshed()
    n.set_scenarios({"low": 0.5, "high": 0.5})
    return n


@pytest.fixture
def ac_dc_meshed_stoch_r(ac_dc_network_r):
    n = ac_dc_network_r.copy()
    n.set_scenarios({"low": 0.5, "high": 0.5})
    return n


@pytest.fixture
def scigrid_de_network():
    return pypsa.examples.scigrid_de()


@pytest.fixture
def ac_dc_network_r():
    csv_folder = Path(__file__).parent / "data" / "ac-dc-meshed" / "results-lopf"
    return pypsa.Network(csv_folder)


@pytest.fixture
def ac_dc_network_mi(ac_dc_network):
    n = ac_dc_network
    n.snapshots = pd.MultiIndex.from_product([[2013], n.snapshots])
    n.investment_periods = [2013]
    gens_i = n.generators.index
    rng = np.random.default_rng()  # Create a random number generator
    n.generators_t.p[gens_i] = rng.random(size=(len(n.snapshots), len(gens_i)))
    return n


@pytest.fixture
def ac_dc_network_shapes(ac_dc_network):
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

    bboxes = n.buses.apply(lambda row: create_bbox(row["x"], row["y"]), axis=1)

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


@pytest.fixture
def scipy_network():
    n = pypsa.examples.scigrid_de()
    n.generators.control = "PV"
    g = n.generators[n.generators.bus == "492"]
    n.generators.loc[g.index, "control"] = "PQ"
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


@pytest.fixture(
    params=[
        "ac_dc_network",
        "scigrid_de_network",
        "storage_hvdc_network",
        "model_energy_network",
        # "ac_dc_network_r",
        # "ac_dc_network_mi",
        # "ac_dc_network_shapes",
        "network_only_component_names",
    ]
)
def network_all(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def network_collection(ac_dc_network_r):
    return pypsa.NetworkCollection(
        [ac_dc_network_r],
        index=pd.MultiIndex.from_tuples([("a", 2030)], names=["scenario", "year"]),
    )


@pytest.fixture(scope="module")
def pandapower_custom_network():
    try:
        import pandapower as pp
    except ImportError:
        pytest.skip("pandapower not installed")
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

    def annuity(life, rate):
        return rate / (1 - (1 + rate) ** -life) if rate else 1 / life

    for cfg in TECH.values():
        cfg["fixed_cost"] = (annuity(LIFE, DR) + FOM / 100) * cfg["inv"]

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
