##########
User Guide
##########

This section contains advice for users.

We strongly recommend looking at the :doc:`examples`, :doc:`quick_start` and :doc:`design`.


Python beginners
================

Learning python
---------------

`Learn Python <http://www.learnpython.org/>`_

`A Beginner's Python Tutorial <https://en.wikibooks.org/wiki/A_Beginner%27s_Python_Tutorial>`_

Learning pandas
---------------

`Common Excel Tasks Demonstrated in Pandas <http://pbpython.com/excel-pandas-comp.html>`_

The book `Python for Data Analysis <http://shop.oreilly.com/product/0636920023784.do>`_


PyPSA object interface
----------------------

If you're starting with Python and are unfamiliar with `pandas
<http://pandas.pydata.org/>`_, the library used for storing data, then
don't worry! You can also access component data via the object
interface

.. code:: python

    for gen in network.generators.obj:
        print(gen.p_nom)
	print(gen.p[network.now])


Python experts
==============

You can use the full pandas interface.

For you, the :doc:`design` page should be comprehensive.
