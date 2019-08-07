##########################################
Configuration
##########################################

by changing the default configuration in ``config.yaml``.

Top-level configuration
=======================

.. literalinclude:: ../config.yaml
   :language: yaml
   :lines: 1-5,15

.. csv-table::
   :header-rows: 1
   :widths: 25,7,22,30
   :file: configtables/toplevel.csv


Wildcards and ``scenario``
==========================

.. literalinclude:: ../config.yaml
   :language: yaml
   :lines: 6-13

.. csv-table::
   :header-rows: 1
   :widths: 25,7,22,30
   :file: configtables/scenario.csv
   


``snapshots``
=============

Arguments to `pandas.date_range <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.date_range.html>`_

.. literalinclude:: ../config.yaml
   :language: yaml
   :lines: 17-21

.. csv-table::
   :header-rows: 1
   :widths: 25,7,22,30
   :file: configtables/snapshots.csv


``electricity``
===============

.. literalinclude:: ../config.yaml
   :language: yaml
   :lines: 27-44

.. csv-table::
   :header-rows: 1
   :widths: 25,7,22,30
   :file: configtables/electricity.csv

.. warning::
    Carriers in ``conventional_carriers`` must not also be in ``extendable_carriers``.

``atlite``
=============

.. literalinclude:: ../config.yaml
   :language: yaml
   :lines: 46-59

.. csv-table::
   :header-rows: 1
   :widths: 25,7,22,30
   :file: configtables/atlite.csv

``renewable``
=============

``onwind``
----------

.. csv-table::
   :header-rows: 1
   :widths: 25,7,22,30
   :file: configtables/onwind.csv

``offwind-ac``
--------------

.. csv-table::
   :header-rows: 1
   :widths: 25,7,22,30
   :file: configtables/offwind-ac.csv

``offwind-dc``
---------------

.. csv-table::
   :header-rows: 1
   :widths: 25,7,22,30
   :file: configtables/offwind-dc.csv

``solar``
---------------

.. csv-table::
   :header-rows: 1
   :widths: 25,7,22,30
   :file: configtables/solar.csv

``hydro``
---------------

.. csv-table::
   :header-rows: 1
   :widths: 25,7,22,30
   :file: configtables/hydro.csv

``lines``
=============

.. csv-table::
   :header-rows: 1
   :widths: 25,7,22,30
   :file: configtables/lines.csv

``links``
=============

.. csv-table::
   :header-rows: 1
   :widths: 25,7,22,30
   :file: configtables/links.csv

``transformers``
================

.. csv-table::
   :header-rows: 1
   :widths: 25,7,22,30
   :file: configtables/transformers.csv

``load``
=============

.. csv-table::
   :header-rows: 1
   :widths: 25,7,22,30
   :file: configtables/load.csv

``costs``
=============

.. csv-table::
   :header-rows: 1
   :widths: 25,7,22,30
   :file: configtables/costs.csv

``solving``
=============

``options``
-----------

.. csv-table::
   :header-rows: 1
   :widths: 25,7,22,30
   :file: configtables/solving-options.csv

``solver``
----------

.. csv-table::
   :header-rows: 1
   :widths: 25,7,22,30
   :file: configtables/solving-solver.csv

``plotting``
=============

.. csv-table::
   :header-rows: 1
   :widths: 25,7,22,30
   :file: configtables/plotting.csv
