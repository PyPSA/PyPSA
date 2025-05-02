import numpy as np
import pytest

rng = np.random.default_rng()


@pytest.fixture(params=["ac_dc_network", "ac_dc_network_mi"], scope="function")
def n(request):
    # Here we return the appropriate fixture based on the parameters.
    return request.getfixturevalue(request.param)


# @pytest.mark.parametrize("add_scenarios", [True, False])
# def test_xarray_cast(n, add_scenarios):
#     c = n.components.generators
#     # Check coords
#     coords_check = {
#         "snapshot": c.snapshots,
#         "component": c.component_names,
#     }
#     if add_scenarios:
#         n.scenarios = ["scenario1", "scenario2"]
#         assert c.has_scenarios
#         assert coords_check["scenario"] == c.scenarios
#     if c.has_investment_periods:
#         coords_check["period"] = c.snapshots.get_level_values("period")
#         coords_check["timestep"] = c.timesteps
#     assert set(c.ds.coords.keys()) == set(coords_check.keys())
#     for coord in coords_check:
#         assert Counter(c.ds.coords[coord].values) == Counter(coords_check[coord].values)

#     # Static and dynamic data
#     assert (c.static.p_max_pu.to_xarray().isin(c.ds.p_max_pu)).all()
#     assert (c.dynamic.p_max_pu.to_xarray().isin(c.ds.p_max_pu)).all()
