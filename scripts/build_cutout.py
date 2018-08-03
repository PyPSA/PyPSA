import os
import atlite
import logging
logger = logging.getLogger(__name__)

logging.basicConfig(level=snakemake.config['logging_level'])

cutout = atlite.Cutout(snakemake.wildcards.cutout,
                       cutout_dir=os.path.dirname(snakemake.output.cutout),
                       **snakemake.config['atlite']['cutouts'][snakemake.wildcards.cutout])

cutout.prepare(nprocesses=snakemake.config['atlite'].get('nprocesses', 4))
