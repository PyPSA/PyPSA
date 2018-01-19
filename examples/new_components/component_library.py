
#Create a library with its own Network class that has new components
#and a new LOPF function

#NB: This only works with Python 3 because of super()

import pypsa, pandas as pd


new_components = pd.DataFrame(data = [["chps","Combined heat and power plant.","new",None],
                                      ["methanations","Methanation plant.","new",None]],
                              index = ["CHP","Methanation"],
                              columns = ["list_name","description","type","attrs"])

chp_attrs = pd.DataFrame(data = [["string","n/a","n/a","Unique name","Input (required)"],
                                 ["string","n/a","n/a","name of bus to which generator is attached","Input (required)"],
                                 ["float","n/a",1.,"power sign","Input (optional)"],
                                 ["static or series","MW",0.,"active power set point (for PF)","Input (optional)"],
                                 ["series","MW",0.,"active power at bus (positive if net generation)","Output"]],
                         index = ["name","bus","sign","p_set","p"],
                         columns = ["type","unit","default","description","status"])


new_components.at["CHP","attrs"] = chp_attrs
new_components.at["Methanation","attrs"] = chp_attrs

class Network(pypsa.Network):

    def __init__(self,**kwargs):
        super().__init__(new_components=new_components,**kwargs)

    def lopf(self,**kwargs):

        #at this point check that all the extra links are in place for the CHPs


        #the following function should add to any extra_functionality in kwargs
        def extra_func(network, snapshots):
            print("Do something fancy")

            #at this point add the constraints for the CHPs


        super().lopf(extra_functionality=extra_func,**kwargs)
