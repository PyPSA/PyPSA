# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import pandas as pd

from pypsa.components.types import component_types_df, default_components, get


def test_custom_components():
    df = pd.read_csv(
        Path(__file__).parent.parent / "pypsa" / "data" / "components.csv", index_col=0
    )

    assert component_types_df.equals(df)

    for component in df.index:
        get(component)

    assert default_components == df.index.to_list()
