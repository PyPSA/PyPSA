#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 18:29:48 2022

@author: fabian
"""

import pytest
import os
import pypsa


@pytest.fixture(scope="module")
def scipy_network():
    csv_folder = os.path.join(os.path.dirname(__file__), "..",
                      "examples", "scigrid-de", "scigrid-with-load-gen-trafos")

    return pypsa.Network(csv_folder)


@pytest.fixture(scope="module")
def ac_dc_network():
    csv_folder = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "ac-dc-meshed",
        "ac-dc-data"
    )
    return pypsa.Network(csv_folder)
