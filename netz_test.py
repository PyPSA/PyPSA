import pypsa
import pandas

# Neues leeres Netz
netz = pypsa.Network()

# Busse
netz.add("Bus", "Bus 1", v_nom=110)
netz.add("Bus", "Bus 2", v_nom=110)

# Leitung
netz.add("Line", "1-2", bus0="Bus 1", bus1="Bus 2", x=0.05, r=0.01)

# Generator (Slack)
netz.add("Generator", "Gen 1", bus="Bus 1", p_set=100, control="Slack")

# Last
netz.add("Load", "Load 2", bus="Bus 2", p_set=100)

# Power Flow
netz.pf()

# Ergebnisse anzeigen
print(netz.buses_t.v_mag_pu)
