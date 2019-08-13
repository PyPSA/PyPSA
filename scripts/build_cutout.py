"""
Create cutouts with `atlite <https://atlite.readthedocs.io/en/latest/>`_.

Relevant Settings
-----------------

.. code:: yaml

    atlite:
        nprocesses:
        cutouts:
            {cutout}:

.. seealso:: 
    Documentation of the configuration file ``config.yaml`` at
    :ref:`atlite_cf`

Inputs
------

*None*

Outputs
-------

- ``cutouts/{cutout}``:

Description
-----------

"""
import os
import atlite
import logging
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=snakemake.config['logging_level'])

    cutout_params = snakemake.config['atlite']['cutouts'][snakemake.wildcards.cutout]
    for p in ('xs', 'ys', 'years', 'months'):
        if p in cutout_params:
            cutout_params[p] = slice(*cutout_params[p])

    cutout = atlite.Cutout(snakemake.wildcards.cutout,
                        cutout_dir=os.path.dirname(snakemake.output[0]),
                        **cutout_params)

    cutout.prepare(nprocesses=snakemake.config['atlite'].get('nprocesses', 4))
