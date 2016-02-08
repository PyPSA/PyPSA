################
Pitfalls/Gotchas
################


Some attributes are generated dynamically and are therefore only
copies. If you change data in them, this will NOT update the original
data. They are all defined as functions to make this clear.

For example:

network.branches() returns a DataFrame which is a union of
network.lines and network.transformers

bus.generators() returns a DataFrame consisting of generators attached
to bus
