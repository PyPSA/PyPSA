<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Store

The [`Store`][pypsa.components.Stores] component connects to a single bus and provides fundamental
inter-temporal storage functionality of the carrier of the [`Bus`][pypsa.components.Buses] it attaches to.
It is not limited in charging or discharging power.

To control charging and discharging power, separate [`Link`][pypsa.components.Links] components must be
connected to the [`Bus`][pypsa.components.Buses] to which the [`Store`][pypsa.components.Stores] attaches. This decoupled approach
enables independent optimization of power and energy capacities as well as
asymmetric power ratings for charging and discharging.

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

The `marginal_cost` attribute applies equally to both charging and discharging
operations, representing the cost per unit of energy stored or released. This
differs from [`StorageUnit`][pypsa.components.StorageUnits] components where marginal costs apply only to the
discharging power. For instance, the `marginal_cost` of the [`Store`][pypsa.components.Stores] component
can represent trading in external energy markets where the stored carrier can be
bought or sold at fixed market prices.

!!! note "When to use [`StorageUnit`][pypsa.components.StorageUnits] instead?"

    Use [`StorageUnit`][pypsa.components.StorageUnits] when power and energy capacities have a fixed relationship and you need integrated dispatch control within a single component. For example, this is recommended for a storage device where the power-to-energy ratio is predetermined by the manufacturer.
    The [`StorageUnit`][pypsa.components.StorageUnits] also has attributes for hydro-electric `inflow` and `spillage`.
    See [this example](../../examples/replace-generator-storage-units-with-store.ipynb) for implementation differences.

{{ read_csv('../../../pypsa/data/component_attrs/stores.csv', disable_numparse=True) }} 
