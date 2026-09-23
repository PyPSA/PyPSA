<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Store

The [`Store`][pypsa.components.Stores] component connects to a single bus and provides
inter-temporal storage of the carrier of the [`Bus`][pypsa.components.Buses] it attaches to.
Its nominal quantity is the energy capacity `e_nom` in MWh. The dispatch `p` is
positive when the store supplies the bus. Where losses or direction-specific costs
require it, a charging variable `p_store` complements it, so that `p + p_store` is
the discharge. The outputs `p_dispatch` and `p_store` report both directions.

By default, charging and discharging power are not limited. A finite `max_hours`
limits them to `e_nom / max_hours` (scaled by `p_max_pu` and `p_min_pu`), i.e. it
couples the power capacity to the energy capacity as for a battery with a fixed
energy-to-power ratio. Losses on the way into and out of the store are given by
`efficiency_store` and `efficiency_dispatch`, standing losses by `standing_loss`.
An exogenous `inflow` (e.g. river inflow to a hydro reservoir) may be spilled at
`spill_cost`.

For independent charging and discharging power capacities, connect separate
[`Link`][pypsa.components.Links] components to an auxiliary
[`Bus`][pypsa.components.Buses] to which the [`Store`][pypsa.components.Stores] attaches.
This decoupled approach also enables asymmetric power ratings for charging and discharging.

!!! example "Hydrogen storage system with a [`Store`][pypsa.components.Stores] and two [`Link`][pypsa.components.Links] components"

    ```mermaid
    graph LR
        ElectricityBus["Electricity Bus"]:::bus
        HydrogenBus["Hydrogen Bus"]:::bus
        SteelTank["Steel Tank Store"]:::store

        ElectricityBus -->|Electrolyser Link| HydrogenBus
        HydrogenBus -->|Hydrogen Turbine Link| ElectricityBus
        HydrogenBus --> SteelTank

        classDef bus fill:#f9f,stroke:#333,stroke-width:2,shape:circle;
        classDef store fill:#bbf,stroke:#333,stroke-width:2,shape:rect;
    ```

## Marginal costs

The `marginal_cost` attribute applies to the net dispatch `p`: discharging incurs
a cost and charging is credited. It represents the value of the stored carrier,
e.g. for trading it at a fixed price or for assigning a water value. The
`marginal_cost_dispatch` and `marginal_cost_store` attributes apply to
discharging and charging separately, e.g. for variable operating costs of a
battery inverter or for wear costs per MWh moved.

## Relation to storage units

[`Store`][pypsa.components.Stores] covers all functionality of
[`StorageUnit`][pypsa.components.StorageUnits] components and, with infinite
`max_hours`, leaves the power unconstrained.
[`n.storage_units_to_stores()`][pypsa.Network.storage_units_to_stores] converts
storage units into equivalent stores. In custom constraints, the net dispatch of
a store is the variable `n.model["Store-p"]` and the charging power `n.model["Store-p_store"]`.

{{ read_csv('../../../pypsa/data/component_attrs/stores.csv', disable_numparse=True) }}
