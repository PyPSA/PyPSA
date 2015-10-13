
#Relative paths seem to be necessary for Python 3

from .classes import Network, Bus, Load, Generator, Line, Transformer, Converter, SubNetwork

from . import pf,opf

Network.lpf = pf.network_lpf

SubNetwork.lpf = pf.sub_network_lpf

Network.pf = pf.network_pf

Network.lopf = opf.network_lopf
