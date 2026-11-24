<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Dispatch Limits

Each [`Generator`][pypsa.components.Generators] $i$ has a dispatch variable $g_{i,s,t}$ where $s$ labels the scenario and $t$ labels the snapshot. Each [`Link`][pypsa.components.Links] $k$ has a dispatch variable $f_{k,s,t}$. Each [`Process`][pypsa.components.Processes] $m$ has an internal dispatch variable $r_{m,s,t}$. Each [`Line`][pypsa.components.Lines] $l$ has a flow variable $p_{l,s,t}$ and each [`Transformer`][pypsa.components.Transformers] $q$ a flow variable $p_{s,t,q}$. The traits [`dispatch`](formulation/traits/dispatch.md) and [`rating`](formulation/traits/rating.md) state the limits below.

!!! note "Dispatch limits of [`Store`][pypsa.components.Stores] and [`StorageUnit`][pypsa.components.StorageUnits]"

    Dispatch limits of stores and storage units are described together with their
    storage consistency equations in the [Storage](storage.md) section.

## Non-extendable Components

For non-extendable components (`{p,s}_nom_extendable=False`), the dispatch is limited by:

<!-- md:spec-table Generator_fix_p_lower Generator_fix_p_upper Link_fix_p_lower Link_fix_p_upper Process_fix_p_lower Process_fix_p_upper Line_fix_s_lower Line_fix_s_upper Transformer_fix_s_lower Transformer_fix_s_upper -->

where the rated capacity (e.g. `Generator_p_nom_finite`) is the nominal capacity $\hat{g}_{i,s}$, $\hat{f}_{k,s}$, $\hat{r}_{m,s}$ or $\hat{p}_{l,s}$, reduced by planned maintenance where it applies; $\underline{g}_{i,s,t}$, $\underline{f}_{k,s,t}$, $\underline{r}_{m,s,t}$, $\bar{g}_{i,s,t}$, $\bar{f}_{k,s,t}$, $\bar{r}_{m,s,t}$ and $\bar{p}_{l,s,t}$ are time-dependent restrictions on the dispatch given per unit of nominal capacity (e.g. due to wind availability for generators or dynamic line rating and security margins for lines). With [lossy transmission](../optimization/power-flow.md#linear-loss-approximation), the line limits include the loss $\psi_{l,s,t}$.

The dual variables, scaled by the snapshot weighting $w^{\mathrm{o}}_t$, are stored in `mu_lower` and `mu_upper` of the dynamic component data (e.g. `n.generators_t.mu_upper`).

## Extendable Components

For extendable components  (`{p,s}_nom_extendable=True`), the dispatch is limited by:

<!-- md:spec-table Generator_ext_p_lower Generator_ext_p_upper Link_ext_p_lower Link_ext_p_upper Process_ext_p_lower Process_ext_p_upper Line_ext_s_lower Line_ext_s_upper Transformer_ext_s_lower Transformer_ext_s_upper -->

where $G_{i}$, $F_{k}$, $R_{m}$, and $P_{l}$ are the nominal capacities to be optimised. Their dual variables are stored as for non-extendable components.

## Fixed Dispatch

Additionally, the dispatch can be fixed to a certain value $\tilde{g}_{i,s,t}$ for generators, $\tilde{f}_{k,s,t}$ for links and $\tilde{r}_{m,s,t}$ for processes. In this case, the dispatch is given by:

<!-- md:spec-table Generator_p_set Link_p_set Process_p_set -->

The dual variables, scaled by $w^{\mathrm{o}}_t$, are stored in `mu_p_set` (e.g. `n.generators_t.mu_p_set`).

## Volume Limits

Generators, links, processes and stores can also have volume limits, i.e. the weighted sum of their power over all snapshots must be above a minimum (`e_sum_min`) or below a maximum (`e_sum_max`). The limits apply once over all snapshots. With multiple investment periods, they apply to the total over all periods, and the energy of each period is multiplied by its investment period weighting `years`, as for [`operational_limit`](./global-constraints.md) global constraints.

<!-- md:spec-table Generator_e_sum_min Generator_e_sum_max Link_e_sum_min Link_e_sum_max Process_e_sum_min Process_e_sum_max Store_e_sum_min Store_e_sum_max -->

The limited power has the sign convention of `p`. For links it is the withdrawal at `bus0`, and for processes the internal power `p`. For stores it is the net power, i.e. discharge minus charge (see [Storage](storage.md) for the symbol $h$). All volume limits use the generator snapshot weightings $w^{\mathrm{g}}_t$, as [`n.statistics.energy_balance()`][pypsa.statistics.StatisticsAccessor.energy_balance]. Their dual variables are only stored in `n.model`.

!!! note "Rolling horizon"

    In [`n.optimize.optimize_with_rolling_horizon()`][pypsa.optimization.OptimizationAccessor.optimize_with_rolling_horizon] the volume limits apply to all snapshots of the rolling horizon. Each window gets the part of the limit left open by previous windows and by what later windows can at most (`e_sum_max`) or must at least (`e_sum_min`) produce given their capacity, `p_min_pu` and `p_max_pu`. Committable and maintainable components may produce nothing in later windows, inactive components produce nothing. Without foresight, a maximum is depleted by the earliest windows and a minimum is filled by the latest windows, which may turn the last window infeasible.

## Voltage Angle Limits

AC lines and transformers can limit the magnitude of their linearised voltage angle difference to $\bar{\theta}$ (attribute `v_ang_max` in degrees). The limit applies symmetrically to both flow directions; the attribute `v_ang_min` is deprecated and ignored. The limit is imposed as a bound on the flow by dividing through the strictly positive effective reactance $x$ (`x_pu_eff`), and is only added for branches with a finite `v_ang_max`. Unlike a manual conversion into `s_max_pu`, this formulation contains no branch capacity, neither $\hat{p}$ nor $P$, and therefore stays linear for extendable branches, although the reactance is a fixed parameter in the optimisation (see [`n.optimize.optimize_transmission_expansion_iteratively()`][pypsa.optimization.OptimizationAccessor.optimize_transmission_expansion_iteratively]).

For a line $l$ the angle difference is $x_{l,s}\, p_{l,s,t}$. For a transformer $q$ it additionally includes the phase shift: either the fixed `phase_shift` or, when `phase_shift_min < phase_shift_max`, the optimised phase-shift variable $\phi_{s,t,q}$, so the angle difference is $x_{s,q}\, p_{s,t,q} + \phi_{s,t,q}$.

<!-- md:spec-table Line_v_ang_lower Line_v_ang_upper Transformer_v_ang_lower Transformer_v_ang_upper Transformer_v_ang_var_lower Transformer_v_ang_var_upper -->

For background on voltage angle difference limits, see the [MATPOWER manual](https://matpower.org/docs/MATPOWER-manual.pdf), pages 76-77.

??? note "Symbols"

    === "Generator"

        <!-- md:spec-legend Generator_fix_p_lower Generator_fix_p_upper Generator_ext_p_lower Generator_ext_p_upper Generator_p_set Generator_e_sum_min Generator_e_sum_max -->

    === "Link"

        <!-- md:spec-legend Link_fix_p_lower Link_fix_p_upper Link_ext_p_lower Link_ext_p_upper Link_p_set Link_e_sum_min Link_e_sum_max -->

    === "Process"

        <!-- md:spec-legend Process_fix_p_lower Process_fix_p_upper Process_ext_p_lower Process_ext_p_upper Process_p_set Process_e_sum_min Process_e_sum_max -->

    === "Store"

        <!-- md:spec-legend Store_e_sum_min Store_e_sum_max -->

    === "Line"

        <!-- md:spec-legend Line_fix_s_lower Line_fix_s_upper Line_ext_s_lower Line_ext_s_upper Line_v_ang_lower Line_v_ang_upper -->

    === "Transformer"

        <!-- md:spec-legend Transformer_fix_s_lower Transformer_fix_s_upper Transformer_ext_s_lower Transformer_ext_s_upper Transformer_v_ang_lower Transformer_v_ang_upper Transformer_v_ang_var_lower Transformer_v_ang_var_upper -->
