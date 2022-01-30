import pypsa, numpy as np

def test_344():
    "Overridden multi-links but empty n.links."

    override = pypsa.descriptors.Dict({k : v.copy() for k,v in pypsa.components.component_attrs.items()})
    override["Link"].loc["bus2"] = ["string",np.nan,np.nan,"2nd bus","Input (optional)"]
    override["Link"].loc["efficiency2"] = ["static or series","per unit",1.,"2nd bus efficiency","Input (optional)"]
    override["Link"].loc["p2"] = ["series","MW",0.,"2nd bus output","Output"]

    network = pypsa.Network(override_component_attrs=override)

    network.add("Bus", "a")
    network.add("Load", "a", bus="a", p_set=5)
    network.add("Generator", "a", bus="a", p_nom=5)

    network.lopf(pyomo=False)
