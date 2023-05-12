#%%
import pypsa
import pandas as pd
n = pypsa.Network()
n.add('Bus', '1')
n.add('Bus', '2')
n.add('Bus', '3')
n.add('Line', 'Line12', bus0='1', bus1='2', x=0.1, s_nom=10)
n.add('Line', 'Line23', bus0='2', bus1='3', x=0.2, s_nom=10)
n.add('Line', 'Line31', bus0='3', bus1='1', x=0.3, s_nom=10)

n.determine_network_topology()
s = n.sub_networks.iloc[0,2]
s.calculate_PTDF()
print(s.PTDF)

n.buses['control'] = 'PQ'
n.buses.loc['2', 'control'] = 'Slack'

n.determine_network_topology()
s = n.sub_networks.iloc[0,2]
s.calculate_PTDF()
print(s.PTDF)

n.buses['control'] = 'PQ'
n.buses.loc['3', 'control'] = 'Slack'

n.determine_network_topology()
s = n.sub_networks.iloc[0,2]
s.calculate_PTDF()
print(s.PTDF)
s.calculate_BODF()
print(s.BODF)

#%%
import pypsa
n = pypsa.Network()
n.add('Bus', '1')
n.add('Bus', '2')
n.add('Line', 'Line12', bus0='1', bus1='2', x=0.1, s_nom=10)

n.determine_network_topology()
s = n.sub_networks.iloc[0,2]
s.calculate_PTDF()
print(s.PTDF)
s.calculate_BODF()
print(s.BODF)

#%%
import pypsa
n = pypsa.Network()
n.add('Bus', '1')

n.determine_network_topology()
s = n.sub_networks.iloc[0,2]
s.calculate_PTDF()
print(s.PTDF)
s.calculate_BODF()
print(s.BODF)

#%%

import pypsa
import pandas as pd
n = pypsa.Network()
n.add('Bus', '1')
n.add('Bus', '2')
n.add('Bus', '3')
n.add('Line', 'Line12', bus0='1', bus1='2', x=0.1, s_nom=10)
n.add('Line', 'Line23', bus0='2', bus1='3', x=0.2, s_nom=10)
n.add('Line', 'Line31', bus0='3', bus1='1', x=0.3, s_nom=10)

n.add('Bus', '4')
n.add('Bus', '5')
n.add('Line', 'Line45', bus0='4', bus1='5', x=0.1, s_nom=10)

n.add('Bus', '6')
print(n.PTDF())