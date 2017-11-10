## Demonstrate usage of logging
#
#PyPSA uses the Python standard library [logging](https://docs.python.org/3/library/logging.html).
#
#This script shows how to use it and control the logging messages from different modules.
#
#Available as a Jupyter notebook at http://www.pypsa.org/examples/logging-demo.ipynb.

#logging.basicConfig() needs to be called BEFORE importing PyPSA

#The reason is that logging.basicConfig() can only be called
#once, and it is already called in pypsa.__init__.py; further
#calls are ignored.

#Choices are ERROR, WARNING, INFO, DEBUG

import logging
logging.basicConfig(level=logging.ERROR)

import pypsa, os

csv_folder_name = (os.path.dirname(pypsa.__file__)
                   + "/../examples/ac-dc-meshed/ac-dc-data/")
network = pypsa.Network(csv_folder_name=csv_folder_name)

out = network.lopf()

out = network.lpf()

#now turn on warnings just for OPF module
pypsa.opf.logger.setLevel(logging.WARNING)

out = network.lopf()

#now turn on all messages for the PF module
pypsa.pf.logger.setLevel(logging.DEBUG)

out = network.lpf()

#now turn off all messages for the PF module again
pypsa.pf.logger.setLevel(logging.ERROR)

out = network.lpf()

