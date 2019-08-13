#####################
A PyPSA-EUR Example
#####################

As PyPSA-EUR has mostly the same features as PyPSA in general, this provided example will guide the reader through running a simplified model:

#. how to configure a (PyPSA-EUR) network model

#. how to run snakemake (step by step) from network creation to the fully solved network

#. short analysis of the solved network (provided jupyter notebook) 


All of the data can be found in ``/pypsa-eur/example/``.

.. note::
   If you were to re-run the example, make sure to take care of the following two arrangement-changes:

   #. exchange the ``pypsa-eur/cutouts/europe-2013-era5`` folder by the provided one in ``pypsa-eur/example/europe-2013-era5``

   #. exchange the ``pypsa-eur/config.yaml`` file by the provided ``pypsa-eur/example/config.yaml``

   You can also rename the original files before pasting the examplary cutout and configuration which will make it easier to switch back after the example.

Configuration
=============
All references in this section relate to the ``config.yaml`` provided for this example unless mentioned otherwise. We discuss only major changes to the provided default configuration or point out important settings, as the full discussion of configuration would be too extensive.

Our model example considers only Germany (from all the european countries)

.. code-block:: yaml
   :emphasize-lines: 0

   countries: ['DE']
   
for the duration of only a single month (03.2013):

.. code-block:: yaml
   :emphasize-lines: 0

   snapshots:
      start: "2013-03-01"
      end: "2013-04-01"
      closed: 'left' #end is not inclusive

As we consider a huge cutout (in comparison to whole europe), we should also adapt the CO2-Limit:

.. code-block:: yaml
   :emphasize-lines: 0

   electricity:
      [...]
      co2limit: 110.6e+6 #(1/12 DE)

We consider all the available conventional generators

.. code-block:: yaml
   :emphasize-lines: 0

   conventional_carriers: [nuclear, oil, CCGT, coal, lignite, geothermal, biomass]

with fixed capacity (data provided by pypsa-eur in ``/pypsa-eur/resources/powerplants.csv``), while an additional expandable carrier steps in whenever there is unmet demand that cannot be covered by included Storage Units:

.. code-block:: yaml
   :emphasize-lines: 0

   electricity:
      [...]
      extendable_carriers:
         Generator: [OCGT]
         StorageUnit: [battery, H2]
      max_hours:
         battery: 6
         H2: 168

To include profiles (availability curves) for the variable renewable we restrict to the ERA5 dataset in ``/pypsa-eur/cutouts/europe-2013-era5/`` for the same month (03.2013) as well as the approximate latitude and longitude of Germany:

.. code-block:: yaml
   :emphasize-lines: 0

   atlite:
      cutouts:
         europe-2013-era5:
            xs: [4.,15.]
            ys: [56., 46.]
            months: [3, 3]
            years: [2013, 2013]

.. code-block:: yaml
   :emphasize-lines: 0

   renewable:
      [...]
      solar:
         cutout: europe-2013-era5


This last step is insignificant (should not be carried out when running any snakerule) as this example provides the necessary cutout.

Finally, the solver is adapted to ``cbc``, as gurobi might not be free of charge:

.. code-block:: yaml
   :emphasize-lines: 0

   solving:
      [...]
      solver:
         name: cbc

.. note::
    For this configuration it might be necessary to install the package ``ipopt``:

	``conda install -c conda-forge ipopt``
		 
All the other configuration parameters remain default.

Snakemake
=========
In order to obtain the resulting network (which also is provided, but try to recunstruct it yourself), open a ``terminal`` (or use the shell in EMACS), activate the pypsa-eur environment (``activate pypsa-eur``) and run:

    ``snakemake results/networks/elec_s_10_lvopt_Co2L-3H.nc``

where the final solved network can be now found in ``pypsa-eur/results/networks`` with the name ``elec_s_10_lvopt_Co2L-3H.nc``. Each of the intermediate stages is also saved (which we will not discuss here in detail - for more information, see [reference Section] Rules Overview) in ``pypsa-eur/networks``:

    #. ``base.nc``:

        Contains the basic network with buses, links and lines.
	   
    #. ``elec.nc``:

        Adds electricity to the network, such as generators (individual carriers with their respective capacities), loads and availability profiles for each individual carrier (all of the previous located at the corresponding buses)
	   
    #. ``elec_s.nc``:

        Simplifies the previous network (e.g. aggregates generators of same type within one bus)
	   
    #. ``elec_s_10.nc``:

        Clusters the previous network to a smaller one with only 10 buses. The previous contained 333 buses.
	   
    #. ``elec_s_10_lvopt_Co2L-3H.nc``:

        Reduces the time resulution of the network from hourly to every 3 hours and includes the option to optimive line volumes (lvopt) for the solver.

Short Analysis of the German Network (Jupyter)
==============================================

Open the jupyter notebook ``pypsa-eur_example.ipynb`` and consider buses as well as the timeseries for dispatch.
