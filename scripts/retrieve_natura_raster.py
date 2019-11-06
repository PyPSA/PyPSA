## Copyright 2019 Fabian Hofmann (FIAS)

import logging, os
from _helpers import progress_retrieve

logger = logging.getLogger(__name__)

d = './resources'
if not os.path.exists(d):
    os.makedirs(d)

progress_retrieve("https://zenodo.org/record/3518215/files/natura.tiff",
                  "resources/natura.tiff")
                  