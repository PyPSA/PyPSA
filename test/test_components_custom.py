from pathlib import Path

import pandas as pd

import pypsa
from pypsa.components.types import component_types_df, default_components, get


def test_custom_components():
    df = pd.read_csv(
        Path(__file__).parent.parent / "pypsa" / "data" / "components.csv", index_col=0
    )

    assert component_types_df.equals(df)

    for component in df.index:
        get(component)

    assert default_components == df.index.to_list()


def test_custom_component_registration():
    defaults_data = {
        "attribute": ["name", "attribute_a"],
        "type": ["string", "float"],
        "unit": ["n/a", "n/a"],
        "default": ["n/a", 1],
        "description": ["Unique name", "Some custom attribute"],
        "status": ["Input (required)", "Input (optional)"],
    }

    defaults_df = pd.DataFrame(defaults_data)
    pypsa.components.types.add_component_type(
        name="CustomComponent",
        list_name="custom_components",
        description="A custom component example",
        category="custom",
        defaults_df=defaults_df,
    )

    custom_component = get("custom_components")
    assert custom_component.name == "CustomComponent"
    assert custom_component.list_name == "custom_components"
    assert custom_component.description == "A custom component example"
    assert custom_component.category == "custom"
    assert custom_component.defaults.equals(defaults_df)


# def test_unregistered_custom_components():
#     import pypsa

#     with pytest.raises(ValueError, match="Component type 'MyComponent' not found."):
#         pypsa.Network(custom_components=["MyComponent"])
