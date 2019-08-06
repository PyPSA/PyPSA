##########################################
Installation
##########################################

TODO: install via bash installation script

.. note:: 
    The steps are demonstrated as shell commands, where the path before the ``%`` sign denotes the
    directory in which the commands following the ``%`` should be entered.

Clone the Repository
====================

Clone the PyPSA-Eur repository using ``git``

.. code:: bash

    /some/other/path % cd /some/path/without/spaces

    /some/path/without/spaces % git clone https://github.com/PyPSA/pypsa-eur.git

.. note::
    The path to the directory into which the ``git repository`` is cloned, 
    must not have any spaces.


Install Python Dependencies
===============================

TODO: conda link

The python package requirements are curated in the conda `environment.yaml <https://github.com/PyPSA/pypsa-eur/blob/master/environment.yaml>`_ file.
The environment can be installed and activated using

.. code:: bash

    .../pypsa-eur % conda env create -f environment.yaml

    .../pypsa-eur % conda activate pypsa-eur

.. note::
    Note that activation is local to the currently open shell!
    After opening a new terminal window,
    one needs to reissue the second command! 


Download Data Dependencies
==============================

Not all data dependencies are shipped with the git repository, since git is not suited for handling large changing files. Instead we provide separate data bundles:

1. **Data Bundle:** `pypsa-eur-data-bundle.tar.xz <https://vfs.fias.science/d/0a0ca1e2fb/files/?p=/pypsa-eur-data-bundle.tar.xz>`_ contains common GIS datasets like NUTS3 shapes, EEZ shapes, CORINE Landcover, Natura 2000 and also electricity specific summary statistics like historic per country yearly totals of hydro generation, GDP and POP on NUTS3 levels and per-country load time-series. It should be extracted in the ``data`` sub-directory, such that all files of the bundle are stored in the ``data/bundle`` subdirectory)

.. code:: bash

    .../pypsa-eur/data % curl -OL "https://vfs.fias.science/d/0a0ca1e2fb/files/?dl=1&p=/pypsa-eur-data-bundle.tar.xz"

    .../pypsa-eur/data % tar xJf pypsa-eur-data-bundle.tar.xz


2. **Cutouts:** `pypsa-eur-cutouts.tar.xz <https://vfs.fias.science/d/0a0ca1e2fb/files/?p=/pypsa-eur-cutouts.tar.xz>`_ are spatiotemporal subsets of the European weather data from the `ECMWF ERA5 <https://software.ecmwf.int/wiki/display/CKB/ERA5+data+documentation>`_ reanalysis dataset and the `CMSAF SARAH-2 <https://wui.cmsaf.eu/safira/action/viewDoiDetails?acronym=SARAH_V002>`_ solar surface radiation dataset for the year 2013. They have been prepared by and are for use with the `atlite <https://github.com/PyPSA/atlite>`_ tool. You can either generate them yourself using the ``build_cutouts`` rule or extract them directly into the ``pypsa-eur`` directory. Extracting the bundle is recommended, since procuring the source weather data files for ``atlite`` is not properly documented at the moment:

.. code:: bash

    .../pypsa-eur % curl -OL "https://vfs.fias.science/d/0a0ca1e2fb/files/?dl=1&p=/pypsa-eur-cutouts.tar.xz"

    .../pypsa-eur % tar xJf pypsa-eur-cutouts.tar.xz

3. **Natura:** Optionally, you can download a rasterized version of the NATURA dataset `natura.tiff <https://vfs.fias.science/d/0a0ca1e2fb/files/?p=/natura.tiff&dl=1>`_ and put it into the ``resources`` sub-directory. If you don't, it will be generated automatically, which is a time-consuming process.

.. code:: bash

    .../pypsa-eur % curl -L "https://vfs.fias.science/d/0a0ca1e2fb/files/?p=/natura.tiff&dl=1" -o "resources/natura.tiff"


4. **Remove Archives:** Optionally, if you want to save disk space, you can delete ``data/pypsa-eur-data-bundle.tar.xz`` and ``pypsa-eur-cutouts.tar.xz`` once extracting the bundles is complete. E.g.

.. code:: bash

    .../pypsa-eur % rm -rf data/pypsa-eur-data-bundle.tar.xz pypsa-eur-cutouts.tar.xz
