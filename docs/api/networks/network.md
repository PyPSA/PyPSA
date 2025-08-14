::: pypsa.Network
    options:
        members_order: source 
        filters:
          - "!^_[^_]"
          - "!logger"
          - "!iteration"
          # Components
       #   - "!^(buses|carrier|generators|global_constraints|line_types|lines|links|loads|shapes|shunt_impedances|sub_networks|stores|storage_units|transformers|transformer_types)"
       #   - "!(_components)$"
      #    - "!^(components|c|df|static|pnl|dynamic)"
          # Descriptors
     #     - "!^get_(extendable|non_extendable|committable)_i$|^get_(active_assets|switchable_as_(dense|iter))$|^bus_carrier_unit$"
