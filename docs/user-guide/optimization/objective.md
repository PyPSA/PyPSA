<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Objective

The objective is to **minimise total system costs** (i.e. the sum of all costs) subject to
meeting all constraints of the optimisation problem (e.g. energy balance, dispatch limits, etc.) for the selected snapshots.

The objective function is composed of investment costs and multiple forms of operational costs, such as marginal costs, quadratic marginal costs, marginal storage costs, spillage costs, start-up, shut-down and stand-by costs.

Which cost components are included in the objective function depends on the data provided for the network components. For instance, if no extendable components are present (`{p,s,e}_nom_extendable=False`), the investment cost terms are skipped. Or, if no components are marked as committable (`committable=False`), the start-up, shut-down and stand-by costs are skipped. All cost coefficients default to zero, so that the objective function is empty if no costs are defined.

The total cost sums the investment costs of all components and the operational costs, weighted over the scenarios:

<!-- md:spec total_cost -->

## Investment Costs

For extendable components (`{p,s,e}_nom_extendable=True`), the investment costs of expanding their capacity are:

=== "Generator"

    <!-- md:spec Generator_capex Generator_build_cost -->

=== "StorageUnit"

    <!-- md:spec StorageUnit_capex StorageUnit_build_cost -->

=== "Store"

    <!-- md:spec Store_capex Store_build_cost -->

=== "Link"

    <!-- md:spec Link_capex Link_build_cost -->

=== "Process"

    <!-- md:spec Process_capex Process_build_cost -->

=== "Line"

    <!-- md:spec Line_capex Line_build_cost -->

=== "Transformer"

    <!-- md:spec Transformer_capex Transformer_build_cost -->

Here $c$ is the capital cost per unit of nominal capacity of each component. It multiplies the power capacities $G_i$ (generators), $H_v$ (storage units), $F_k$ (links), $R_m$ (processes) and $P_l$ (lines and transformers), and the energy capacity $E_y$ of stores. The capital costs are given in currency/MW for generators, storage units, links, processes, lines and transformers, and in currency/MWh for stores. With a piecewise capital cost curve, the build cost follows the curve instead (see [piecewise costs](piecewise.md)).

To minimise **long-run annual system costs** (currency/a), capital costs for components should be set to annualised investment costs (i.e. currency/MW/a and currency/MWh/a ), marginal costs for dispatch in currency/MWh, and the weightings (h/a) are chosen such that $\sum_t w^{\mathrm{o}}_t = 8760$ hours per annum.

If no extendable components are present, only the dispatch of the components is optimised as in a **short-run market model**.

### Specifying Capital Costs

PyPSA supports two approaches for specifying investment costs:

=== "Direct periodized capital cost (default)"

    Provide `capital_cost` directly as annualized cost per unit capacity (currency/MW or currency/MWh) per model period:

    ``` py
    n.add("Generator", "wind",
          bus="bus",
          p_nom_extendable=True,
          capital_cost=50000)  # Already periodized: €/MW
    ```

=== "Overnight cost with automatic periodization"

    Provide `overnight_cost` (upfront investment cost), `discount_rate`, and `lifetime`. PyPSA automatically calculates the periodized cost:

    ``` py
    n.add("Generator", "wind",
          bus="bus",
          p_nom_extendable=True,
          overnight_cost=1200000,  # Upfront cost: €/MW
          discount_rate=0.07,      # 7% discount rate
          lifetime=25)             # 25 years
    ```

    !!! note "Background"

        The effective periodized cost per MW installation used in optimization is calculated as:

        $$c = \mathit{overnight\_cost} \cdot \mathrm{annuity}(\mathit{discount\_rate}, \mathit{lifetime}) \cdot N_{\mathrm{years}}$$

        where $N_{\mathrm{years}}$ refers to the model time span in units of years, derived from `n.snapshot_weightings["objective"]`. The annuity factor converts overnight cost to annual payments:

        $$\mathrm{annuity}(r, L) = \begin{cases} \dfrac{r}{1 - (1 + r)^{-L}} & \text{if } r > 0 \\ \dfrac{1}{L} & \text{if } r = 0 \end{cases}$$

        with discount rate $r$ and lifetime $L$ in years.

### Fixed Operation & Maintenance Costs

Fixed O&M costs (`fom_cost`) represent another type of periodized costs that are incurred regardless of dispatch, such as maintenance, insurance, and land lease:

``` py
# Wind turbine with 2% of overnight cost as annual FOM
n.generators.loc["wind", "fom_cost"] = 0.02 * n.generators.loc["wind", "overnight_cost"]
```

??? note "Symbols"

    === "Generator"

        <!-- md:spec-legend total_cost Generator_capex Generator_build_cost -->

    === "StorageUnit"

        <!-- md:spec-legend total_cost StorageUnit_capex StorageUnit_build_cost -->

    === "Store"

        <!-- md:spec-legend total_cost Store_capex Store_build_cost -->

    === "Link"

        <!-- md:spec-legend total_cost Link_capex Link_build_cost -->

    === "Process"

        <!-- md:spec-legend total_cost Process_capex Process_build_cost -->

    === "Line"

        <!-- md:spec-legend total_cost Line_capex Line_build_cost -->

    === "Transformer"

        <!-- md:spec-legend total_cost Transformer_capex Transformer_build_cost -->

## Operational Costs

The operational costs of each scenario $s$ sum the costs of all components. The total cost weights them with the scenario probability $\pi_s$ and, for risk-averse optimisation, the conditional value at risk (see [stochastic optimisation](stochastic.md)):

<!-- md:spec scenario_opex risk_weighted_opex -->

All operational costs are weighted with the objective snapshot weighting $w^{\mathrm{o}}_t$ and, with multiple investment periods, with the period weighting $W^{\mathrm{o}}$ (see [pathway planning](pathway-planning.md)).

### Marginal Costs

The marginal costs $o$ of dispatch apply to the generator dispatch $g_{i,s,t}$, the storage unit discharge $h^{-}_{s,t,v}$, the store dispatch $h_{s,t,y}$, the link flow $f_{k,s,t}$ and the process dispatch $r_{m,s,t}$. Storage units only incur costs for discharging. For stores, `marginal_cost` applies to the net dispatch, so it is a cost for discharging and a revenue for charging, while the store attributes `marginal_cost_dispatch` and `marginal_cost_store` apply to one direction only (see [storage](storage.md#stores)).

**Quadratic marginal costs** (`marginal_cost_quadratic`) multiply the square of the same dispatch variables and turn the problem into a quadratic program (QP). With a piecewise marginal cost curve, the running cost follows the curve instead (see [piecewise costs](piecewise.md)).

=== "Generator"

    <!-- md:spec Generator_opex Generator_running_cost -->

=== "StorageUnit"

    <!-- md:spec StorageUnit_opex StorageUnit_running_cost -->

=== "Store"

    <!-- md:spec Store_opex Store_running_cost Store_charge_opex -->

=== "Link"

    <!-- md:spec Link_opex Link_running_cost -->

=== "Process"

    <!-- md:spec Process_opex Process_running_cost -->

### Marginal Storage and Spillage Costs

Marginal storage costs (`marginal_cost_storage`) represent the cost of holding energy in storage units ($\mathit{soc}_{s,t,v}$) and stores ($e_{s,t,y}$). Spillage costs (`spill_cost`) apply to spillage, e.g. sending water in a hydro reservoir over a spillway without generating electricity:

=== "StorageUnit"

    <!-- md:spec StorageUnit_storage_opex -->

=== "Store"

    <!-- md:spec Store_storage_opex -->

### Start-Up, Shut-Down, Stand-By Costs

For generators, links and processes with unit commitment (`committable=True`), the stand-by costs (`stand_by_cost`), start-up costs (`start_up_cost`) and shut-down costs (`shut_down_cost`) are linked to the status $u$, start-up $su$ and shut-down $sd$ variables:

=== "Generator"

    <!-- md:spec Generator_commitment_opex -->

=== "Link"

    <!-- md:spec Link_commitment_opex -->

=== "Process"

    <!-- md:spec Process_commitment_opex -->

Only the stand-by costs are weighted by the objective snapshot weightings $w^{\mathrm{o}}_t$. Components with unit commitment constraints turn the problem into a mixed-integer linear program (MILP).

??? note "Symbols"

    === "Generator"

        <!-- md:spec-legend scenario_opex risk_weighted_opex Generator_opex Generator_running_cost Generator_commitment_opex -->

    === "StorageUnit"

        <!-- md:spec-legend scenario_opex risk_weighted_opex StorageUnit_opex StorageUnit_running_cost StorageUnit_storage_opex -->

    === "Store"

        <!-- md:spec-legend scenario_opex risk_weighted_opex Store_opex Store_running_cost Store_storage_opex Store_charge_opex -->

    === "Link"

        <!-- md:spec-legend scenario_opex risk_weighted_opex Link_opex Link_running_cost Link_commitment_opex -->

    === "Process"

        <!-- md:spec-legend scenario_opex risk_weighted_opex Process_opex Process_running_cost Process_commitment_opex -->

Some decision variables do not show up in the objective function, such as the power flow on lines and transformers ($p_{l,s,t}$) and the storage unit charging ($h^{+}_{s,t,v}$). They are only used to enforce constraints, e.g. the power flow on lines and transformers.
