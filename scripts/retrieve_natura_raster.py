#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 18:37:11 2019

@author: fabian
"""

import logging, os
from _helpers import progress_retrieve

logger = logging.getLogger(__name__)

d = './resources'
if not os.path.exists(d):
    os.makedirs(d)

url = "https://zenodo.org/record/3518215/files/natura.tiff"
file = "resources/natura.tiff"
progress_retrieve(url, file)

