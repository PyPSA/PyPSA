# Unofficial ENTSO-E dataset processed by GridKit

This dataset was generated based on a map extract from May 25, 2018.
This is an _unofficial_ extract of the
[ENTSO-E interactive map](https://www.entsoe.eu/data/map/)
of the European power system (including to a limited extent North
Africa and the Middle East). The dataset has been processed by GridKit
to form complete topological connections.  This dataset is neither
approved nor endorsed by ENTSO-E.

This dataset may be inaccurate in several ways, notably:

+ Geographical coordinates are transfered from the ENTSO-E map, which
  is known to choose topological clarity over geographical
  accuracy. Hence coordinates will not correspond exactly to reality.
+ Voltage levels are typically provided as ranges by ENTSO-E, of which
  the lower bound has been reported in this dataset.
+ Line structure conflicts are resolved by picking the first structure
  in the set
+ Transformers are _not present_ in the original ENTSO-E dataset,
  their presence has been derived from the different voltages from
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
+ `station_id`: unique identifier of its substation; a station may have multiple buses, which are typically connected by transformers
+ `voltage`: the operating voltage of this bus
+ `dc`: boolean ('t' or 'f'), describes whether the bus is a HVDC
  terminal (t) or a regular AC terminal (f)
+ `symbol`: type of station of this bus.
+ `under_construction`: boolean ('t' if station is currently under construction,
  'f' otherwise)
+ `tags`: _hstore_ encoded dictionary of 'extra' properties for this bus
+ `x`: longitude of its location
+ `y`: latitude of its location

**NOTA BENE**: During the processing of the network, so called
'synthetic' stations may be inserted on locations where lines are
apparantly connected. Such synthetic stations can be recognised
because their symbol is always `joint`.

### lines.csv:

Buses are connected by AC-lines:

+ `line_id`: unique identifier for the line
+ `bus0`: first of the two connected buses
+ `bus1`: second of two connected buses
+ `voltage`: operating voltage of the line (identical to operating voltage of
  the bus)
+ `circuits`: number of (independent) circuits in this link, each of which
  typically has 3 cables.
+ `length`: length of line in km
+ `underground`: boolean, `t` if this is an underground cable, `f` for
  an overhead line
+ `under_construction`: boolean, `t` for lines that are currently
  under construction
+ `tags`: _hstore_ encoded dictionary of extra properties for this link
+ `geometry`: extent of this line in well-known-text format (WGS84)

### links.csv:

Connections between buses:

+ `link_id`: unique identifier for the link
+ `bus0`: first of the two connected buses
+ `bus1`: second of two connected buses
+ `length`: length of line in km
+ `under_construction`: boolean, `t` for lines that are currently
  under construction
+ `tags`: _hstore_ encoded dictionary of extra properties for this link
+ `geometry`: extent of this line in well-known-text format (WGS84)

### generators.csv

Generators attached to the network.

+ `generator_id`: unique identifier for the generator
+ `bus_id`: the bus to which this generator is connected
+ `technology`: type of generator
+ `capacity`: capacity of this generator in MW
+ `tags`: _hstore_ encoded dictionary of extra attributes
+ `geometry`: location of generator in well-known text format (WGS84)

### transformers.csv

A transformer connects buses which operate at distinct voltages. **NOTA BENE**:
Transformers are _not_ represented in the original dataset, but instead have
been added at substations to connect AC transmission lines of distinct voltage
levels.

+ `transformer_id`: unique identifier
+ `bus0`: Bus at lower voltage level
  `bus1`: Bus at higher voltage level

### converters.csv

Back-to-back converters connecting non-synchronized buses.

+ `converter_id`: unique identifier
+ `bus0`: First bus
  `bus1`: Second bus
