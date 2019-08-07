

##########################################
Simplifying Networks
##########################################

Simplify Network
================

The rule simplify_network does up to four things:

1. Create an equivalent transmission network in which all voltage levels are mapped to the 380 kV level by the function ``simplify_network(...)``.

2. DC only sub-networks that are connected at only two buses to the AC network are reduced to a single representative link by the function ``simplify_links(...)``. The components attached to buses in between are moved to the nearest endpoint. The grid connection cost of offshore wind generators are added to the captial costs of the generator.

3. Stub lines and links, i.e. dead-ends of the network, are sequentially removed from the network by the function ``remove_stubs(...)``. Components are moved along.

4. If a number was provided after the s (as in elec_s500_...), the network is clustered to this number of clusters with the routines from the cluster_network rule by the function cluster. This step is usually skipped.


Cluster Network
===============

The rule cluster_network instead clusters the network to a given number of buses.

    -Why is this cluster function used?
    -Why the user can set a number behind the elec_sXXX for simplification?

As you found out for yourself, elec_s100_50.nc for example is a network in which simplify_network clusters the network to 100 buses and in a second step cluster_network reduces it down to 50 buses.

Well, let me provide a use-case where this makes sense:

In preliminary tests, it turns out, that the principal effect of changing spatial resolution is actually only partially due to the transmission network. It is more important to differentiate between wind generators with higher capacity factors from those with lower capacity factors, ie to have a higher spatial resolution in the renewable generation than in the number of buses.

This two-step clustering can take advantage of that fact (and allows to study it)
by looking at networks like networks/elec_s100_50m.nc (note the additional m in the cluster wildcard). For this example simplify_network clusters to 100 buses and then cluster_network clusters to 50m buses, which means 50 buses for the network topology but only moving instead of aggregating the generators to the clustered buses. So in this network you still have up to 100 different wind generators, 2 at each bus on average.

In combination these two features allow you to study the spatial resolution of the transmission network separately from the spatial resolution of renewable generators. Beware: There is no clear evidence telling you what is a good representation of the full model. These options are under active study.

    Why we have a cluster function inside of the simplification method?

Why are you asking three times the same question?

    Is it possible to run the model without the simplification method / rule?
    I tryed to run the snakemake without the s for simplification.

No, the network clustering methods in PyPSA's networkclustering module don't work reliably with multiple voltage levels and transformers. If it is somehow necessary for you we could include switches to make Step 2 and 3 optional as well. But that's about it.


Prepare Network
===============
