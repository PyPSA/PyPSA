# Unofficial ENTSO-E dataset processed by GridKit

This dataset was generated based on a map extract from May 11, 2016.
This is an _unofficial_ extract of the
[ENTSO-E interactive map](https://www.entsoe.eu/map/Pages/default.aspx)
of the European power system (including to a limited extent North
Africa and the Middle East). The dataset has been processed by GridKit
to form complete topological connections.  This dataset is neither
approved nor endorsed by ENTSO-E.

This dataset may be inaccurate in several ways, notably:

+ Geographical coordinates are transfered from the ENTSO-E map, which
  is known to choose topological clarity over geographical
  accuracy. Hence coordinates will not correspond exactly to reality.
+ Voltage levels are typically provided as ranges by ENTSO-E, of which
  the lower bound has been reported in this dataset. Not all lines -
  especially DC lines - contain voltage information.
+ Line structure conflicts are resolved by picking the first structure
  in the set
+ Transformers are _not present_ in the original ENTSO-E dataset,
  there presence has been derived from the different voltages from
  connected lines.
+ The connection between generators and busses is derived as the
  geographically nearest station at the lowest voltage level. This
  information is again not present in the ENTSO-E dataset.

All users are advised to exercise caution in the use of this
dataset. No liability is taken for inaccuracies.



## Contents of dataset

This dataset is provided as set of CSV files that describe the ENTSO-E
network. These files use the comma (`,`) as field separator, single
newlines (`\n`) as record separator, and single quotes (`'`) as string
quote characters. The CSV files have headers.

Example code for reading the files:

    # R
    buses <- read.csv("buses.csv", header=TRUE, quote="'")
    # python
    import io, csv
    class dialect(csv.excel):
        quotechar = "'"
    with io.open('buses.csv', 'rb') as handle:
        buses = list(csv.DictReader(handle, dialect))

### buses.csv:

Describes terminals, vertices, or 'nodes' of the system

+ `bus_id`: the unique identifier for the bus
+ `station_id`: the substation or plant of this; a station may have
  multiple buses, which are typically connected by transformers
+ `voltage`: the operating voltage of this bus
+ `dc`: boolean ('t' or 'f'), describes whether the bus is a HVDC
  terminal (t) or a regular AC terminal (f)
+ `symbol`: type of station of this bus.
+ `tags`: _hstore_ encoded dictionary of 'extra' properties for this bus
+ `under_construction`: boolean ('t' if station is currently under
  construction, 'f' otherwise)
+ `geometry`: location of the station in well-known-text format (WGS84)

**NOTA BENE**: During the processing of the network, so called
'synthetic' stations may be inserted on locations where lines are
apparantly connected. Such synthetic stations can be recognised
because their symbol is always `joint`.

### links.csv:

Connections between buses:

+ `link_id`: unique identifier for the link
+ `src_bus_id`: first of the two connected buses
+ `dst_bus_id`: second of two connected buses
+ `voltage`: operating voltage of the link (_must_ be identical to
  operating voltage of the buses)
+ `circuits`: number of (independent) circuits in this link, each of
  which typically has 3 cables (for AC lines).
+ `dc`: boolean, `t` if this is a HVDC line
+ `underground`: boolean, `t` if this is an underground cable, `f` for
  an overhead line
+ `under_construction`: boolean, `t` for lines that are currently
  under construction
+ `length_m`: length of line in meters
+ `tags`: _hstore_ encoded dictionary of extra properties for this link
+ `geometry`: extent of this line in well-known-text format (WGS84)

### generators.csv

Generators attached to the network.

+ `generator_id`: unique identifier for the generator
+ `bus_id`: the bus to which this generator is connected
+ `symbol`: type of generator
+ `capacity`: capacity of this generator (in megawatt)
+ `tags`: _hstore_ encoded dictionary of extra attributes
+ `geometry`: location of generator in well-known text format (WGS84)

### transformers.csv

A transformer forms a link between buses which operate at distinct
voltages.  **NOTA BENE**: Transformers _never_ originate from the
original dataset, and transformers are _only_ infered in 'real'
stations, never in synthetic ('joint') stations.

+ `transformer_id`: unique identifier
+ `symbol`: either `transformer` for AC-to-AC voltage transformers, or
  `ac/dc` for AC-to-DC converters.
+ `src_bus_id`: first of the connected buses
+ `dst_bus_id`: second of connected buses
+ `src_voltage`: voltage of first bus
+ `dst_voltage`: voltage of second bus
+ `src_dc`: boolean, `t` if first bus is a DC terminal
+ `dst_dc`: boolean, `f` if second bus is a DC terminal
+ `geometry`: location of station of this transformer in well-known
  text format (WGS84)
