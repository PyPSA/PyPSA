# Sub-Network

Sub-networks are determined by PyPSA and are not be entered by the user.

Sub-networks are subsets of buses and passive branches (i.e. lines and transformers) that are connected.

They have a uniform energy `carrier` inherited from the buses, such as "DC", "AC", "heat" or "gas". In the case of "AC" sub-networks, these correspond to synchronous areas. Only "AC" and "DC" sub-networks can contain passive branches; all other sub-networks must contain a single isolated bus.

The power flow in sub-networks is determined by the passive flow through passive branches due to the impedances of the passive branches following Kirchhoff's voltage law.

Sub-Network are determined by calling [`n.determine_network_topology()`][pypsa.Network.determine_network_topology].

{{ read_csv('../../../pypsa/data/component_attrs/sub_networks.csv') }}
