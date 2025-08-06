import pypsa


def test_ac_dc_meshed():
    n = pypsa.examples.ac_dc_meshed()
    assert not n.buses.empty


def test_storage_hvdc():
    n = pypsa.examples.storage_hvdc()
    assert not n.buses.empty


def test_scigrid_de():
    n = pypsa.examples.scigrid_de()
    assert not n.buses.empty


def test_model_energy():
    n = pypsa.examples.model_energy()
    assert not n.buses.empty


def test_carbon_management():
    n = pypsa.examples.carbon_management()
    assert not n.buses.empty
