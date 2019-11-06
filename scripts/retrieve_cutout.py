## Copyright 2019 Fabian Hofmann (FIAS)

import logging, os, tarfile
from _helpers import progress_retrieve

logger = logging.getLogger(__name__)

if snakemake.config['tutorial']:
    url = "https://zenodo.org/record/3518020/files/pypsa-eur-tutorial-cutouts.tar.xz"
else:
    url = "https://zenodo.org/record/3517949/files/pypsa-eur-cutouts.tar.xz"

tarball_fn = "./cutouts.tar.xz"

progress_retrieve(url, tarball_fn)

tarfile.open(tarball_fn).extractall()

os.remove(tarball_fn)
