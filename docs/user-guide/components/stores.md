# Store

The [`Store`](/api/components/types/stores) component connects to a single bus and provides fundamental
inter-temporal storage functionality of the carrier of the [`Bus`](/api/components/types/buses) it attaches to.
It is not limited in charging or discharging power.

To control charging and discharging power, separate [`Link`](/api/components/types/links) components must be
connected to the [`Bus`](/api/components/types/buses) to which the [`Store`](/api/components/types/stores) attaches. This decoupled approach
enables independent optimization of power and energy capacities as well as
asymmetric power ratings for charging and discharging.

!!! example "Hydrogen storage system with a [`Store`](/api/components/types/stores) and two [`Link`](/api/components/types/links) components"

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
differs from [`StorageUnit`](/api/components/types/storage_units) components where marginal costs apply only to the
discharging power. For instance, the `marginal_cost` of the [`Store`](/api/components/types/stores) component
can represent trading in external energy markets where the stored carrier can be
bought or sold at fixed market prices.

!!! note "When to use [`StorageUnit`](/api/components/types/storage_units) instead?"

    Use [`StorageUnit`](/api/components/types/storage_units) when power and energy capacities have a fixed relationship and you need integrated dispatch control within a single component. For example, this is recommended for a storage device where the power-to-energy ratio is predetermined by the manufacturer.
    The [`StorageUnit`](/api/components/types/storage_units) also has attributes for hydro-electric `inflow` and `spillage`. 
    See [this example](https://pypsa.readthedocs.io/en/latest/examples/replace-generator-storage-units-with-store.html) for implementation differences.
