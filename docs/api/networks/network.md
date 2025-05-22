::: pypsa.Network
    options:
        inherited_members: true
        filters:
          - "!^_[^_]"
          - "!logger"
          - "!^(buses|carrier|generators|line_types|lines|links|loads|shapes|shunt_impedances|sub_networks|stores|storage_units|transformers|transformer_types)"
          - "!(_components|components|c|df|static|pnl|dynamic)"
    