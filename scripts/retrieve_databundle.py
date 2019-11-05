#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 18:37:11 2019

@author: fabian
"""

import logging, os, tarfile
from _helpers import progress_retrieve

logger = logging.getLogger(__name__)


if snakemake.config['tutorial']:
    url = "https://zenodo.org/record/3517921/files/pypsa-eur-tutorial-data-bundle.tar.xz"
else:
   url = "https://zenodo.org/record/3517935/files/pypsa-eur-data-bundle.tar.xz"

file = "./bundle.tar.xz"

progress_retrieve(url, file)

# extract
tarfile.open('./bundle.tar.xz').extractall('./data')

os.remove("./bundle.tar.xz")

