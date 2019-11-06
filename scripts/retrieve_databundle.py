## Copyright 2019 Fabian Hofmann (FIAS)

import logging, os, tarfile
from _helpers import progress_retrieve

logger = logging.getLogger(__name__)

if snakemake.config['tutorial']:
    url = "https://zenodo.org/record/3517921/files/pypsa-eur-tutorial-data-bundle.tar.xz"
else:
    url = "https://zenodo.org/record/3517935/files/pypsa-eur-data-bundle.tar.xz"

tarball_fn = "./bundle.tar.xz"

progress_retrieve(url, tarball_fn)

tarfile.open(tarball_fn).extractall('./data')

os.remove(tarball_fn)
