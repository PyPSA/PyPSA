import pypsa
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --- Network & Snapshots ------------------------------------------------------
n = pypsa.Network()
n.snapshots = list(range(7))
snap = n.snapshots

# --- Buses --------------------------------------------------------------------
n.add("Bus", "A",  carrier="electricity")
n.add("Bus", "B",  carrier="electricity")
n.add("Bus", "A1", carrier="electricity")
n.add("Bus", "B1", carrier="electricity")
n.add("Bus", "C",  carrier="electricity")

# --- Cost scenarios (7 steps) ------------------------------------------------
gen_a_cost   = pd.Series([1.00, 3.00, 1.00, 1.00, 1.00, 2.00, 1.00], index=snap)
gen_b_cost   = pd.Series([1.0001, 1.00, 1.00, 1.00, 1.00, 1.00, 1.0001], index=snap)  # epsilon tie-break
line_a_cost  = pd.Series([0.0, 0.0, 5.0, 0.0, 2.0, 3.0, 0.0], index=snap)
trafo_b_cost = pd.Series([0.0, 0.0, 0.0, 5.0, 4.0, 2.0, 0.0], index=snap)

# --- Generators ---------------------------------------------------------------
n.add("Generator", "Gen_A", bus="A", p_nom=100,
      marginal_cost=gen_a_cost, carrier="gas")
n.add("Generator", "Gen_B", bus="B", p_nom=100,
      marginal_cost=gen_b_cost, carrier="gas")

# --- Load ---------------------------------------------------------------------
n.add("Load", "Load_C", bus="C", p_set=[100]*7, carrier="load")

# --- Transformers & Lines -----------------------------------------------------
n.add("Transformer", "Trafo_A_A1", bus0="A", bus1="A1",
      r=0.1, x=0.2, s_nom=150, carrier="ac")
n.add("Transformer", "Trafo_B_B1", bus0="B", bus1="B1",
      r=0.1, x=0.2, s_nom=150, carrier="ac", marginal_cost=trafo_b_cost)

n.add("Line", "A1_C", bus0="A1", bus1="C",
      r=0.1, x=0.1, s_nom=150, carrier="ac", marginal_cost=line_a_cost)
n.add("Line", "B1_C", bus0="B1", bus1="C",
      r=0.1, x=0.1, s_nom=150, carrier="ac")

# --- Carriers (decorative only) ----------------------------------------------
n.add("Carrier", "gas",         co2_emissions=10, color="orange")
n.add("Carrier", "ac",          co2_emissions=0,  color="purple")
n.add("Carrier", "load",        co2_emissions=0,  color="blue")
n.add("Carrier", "electricity", co2_emissions=0,  color="blue")

# --- Optimization -------------------------------------------------------------
n.optimize()
print(n.generators_t.p)

# --- Prepare costs for plotting ----------------------------------------------
total_cost_a = gen_a_cost.reset_index(drop=True) + line_a_cost.reset_index(drop=True)
total_cost_b = gen_b_cost.reset_index(drop=True) + trafo_b_cost.reset_index(drop=True)

# --- Regimes ------------------------------------------------------------------
regimes = [
    ("ε-Tie-break → A",            "#d0f0c0"),
    ("Gen_A > Gen_B → B", "#ffe4e1"),
    ("Line_A > 0 → B",        "#ffa07a"),
    ("Trafo_B > 0 → A", "#add8e6"),
    ("Trafo_B > Line A -> A",   "#e6e6fa"),
    ("Gen_A + Line_A > Gen_B + Trafo_B → B",     "#fffacd"),
    ("ε-Tie-break → A",            "#d0f0c0"),
]

# --- Plot ---------------------------------------------------------------------
fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Generator dispatch
n.generators_t.p.plot(ax=axs[0])
axs[0].set_title("Generator Dispatch")
axs[0].set_ylabel("Power [MW]")

# Apply background colors to both subplots
for ax in axs:
    for i, (_, color) in enumerate(regimes):
        ax.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.5)

# Costs subplot
x = range(7)
axs[1].plot(x, total_cost_a, marker="o", label="Path A: Gen_A + Line A")
axs[1].plot(x, total_cost_b, marker="o", label="Path B: Gen_B + Transformer B")
axs[1].plot(x, gen_a_cost.values, marker="x", linestyle="--", label="Gen_A cost")
axs[1].plot(x, gen_b_cost.values, marker="x", linestyle="--", label="Gen_B cost")
axs[1].plot(x, line_a_cost.values, marker=".", linestyle=":", label="Line A cost")
axs[1].plot(x, trafo_b_cost.values, marker=".", linestyle=":", label="Transformer B cost")
axs[1].set_title("Marginal Costs per Path and Component")
axs[1].set_ylabel("Cost")
axs[1].set_xlabel("Snapshot")

# --- Build unified legends for each subplot ----------------------------------
# Unique regime patches
unique_regimes = dict(regimes)
patch_handles = [Patch(color=color, alpha=0.5, label=label)
                 for label, color in unique_regimes.items()]

# Subplot 1: dispatch + regimes
line_handles, line_labels = axs[0].get_legend_handles_labels()
axs[0].legend(line_handles + patch_handles,
              line_labels + [p.get_label() for p in patch_handles],
              loc="center left", bbox_to_anchor=(1.02, 0.5))

# Subplot 2: costs + regimes
line_handles2, line_labels2 = axs[1].get_legend_handles_labels()
axs[1].legend(line_handles2 + patch_handles,
              line_labels2 + [p.get_label() for p in patch_handles],
              loc="center left", bbox_to_anchor=(1.02, 0.5))

plt.tight_layout()
plt.show()
