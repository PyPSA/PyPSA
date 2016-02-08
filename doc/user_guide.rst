##########
User Guide
##########

This section contains advice for users.

We strongly recommend looking at the :doc:`examples`, :doc:`quick_start` and :doc:`design`.


Beginners
=========

If you're starting with Python and are unfamiliar with `pandas
<http://pandas.pydata.org/>`_, the library used for storing data, then
don't worry! You can also access component data via the object
interface

.. code:: python

    for gen in network.generators.obj:
        print(gen.p_nom)
	print(gen.p[network.now])

Experts
=======

You can use the full pandas interface.

For you, the :doc:`design` page should be comprehensive.
