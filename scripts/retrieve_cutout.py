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
    url =        "https://zenodo.org/record/3518020/files/pypsa-eur-tutorial-cutouts.tar.xz"
else:
   url = "https://zenodo.org/record/3517949/files/pypsa-eur-cutouts.tar.xz"

file = "./cutouts.tar.xz"

progress_retrieve(url, file)

# extract
tarfile.open('./cutouts.tar.xz').extractall()
os.remove("./cutouts.tar.xz")

