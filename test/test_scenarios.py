from collections import Counter

import numpy as np
import pandas as pd
import pytest

rng = np.random.default_rng()


@pytest.fixture(
    params=["ac_dc_network", "ac_dc_network_multiindexed"], scope="function"
)
def n(request):
    # Here we return the appropriate fixture based on the parameters.
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize("add_scenarios", [True, False])
def test_ds_cast(n, add_scenarios):
    # ---
    # TODO: Only temporary until
    # - scenarios are implemented
    # - index names are uniform

    def cast_to_multiindex(df, level_name, level_values, axis):
        current_names = getattr(df, axis).names
        new_columns = pd.MultiIndex.from_product([level_values, getattr(df, axis)])
        new_columns.names = [level_name] + current_names
        new_df = pd.concat([df] * 3, axis=axis, keys=level_values)
        setattr(new_df, axis, new_columns)
        return new_df

    n = n.copy()
    c = n.components.generators

    if add_scenarios:
        scenarios = ["A", "B", "C"]
        n.scenarios = pd.Index(scenarios)
        c.static.columns.name = "attribute"
        c.static = cast_to_multiindex(c.static, "scenario", c.scenarios, axis="index")
        for k, v in c.dynamic.items():
            # Add Uniform index names
            assert "timestep" in v.index.names
            v = v.reindex(n.snapshots)
            v = cast_to_multiindex(v, "scenario", c.scenarios, axis="columns")

    # --- end of temporary code

    # Check coords
    coords_check = {
        c.ct.name: c.component_names,
        "timestep": c.timesteps,
    }
    if add_scenarios:
        coords_check["scenario"] = c.scenarios
    if c.has_investment_periods:
        coords_check["period"] = c.investment_periods
    assert list(c.ds.coords.keys()) == list(coords_check.keys())
    for coord in coords_check:
        assert Counter(c.ds.coords[coord].values) == Counter(coords_check[coord].values)

    # Number of variables
    assert len(c.ds.data_vars) == len(c.ct.defaults) - 1  # -1 for the name column)

    # Static and dynamic data
    assert (c.static.p_max_pu.to_xarray().isin(c.ds.p_max_pu)).all()
    assert (c.dynamic.p_max_pu.to_xarray().isin(c.ds.p_max_pu)).all()
