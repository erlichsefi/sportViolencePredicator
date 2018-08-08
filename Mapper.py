import numpy as np

class Mapper:

    def __init__(self):
        self.map = {}
        self.index=0
    
    def add(self,value):
        #print(value)
        if value in self.map.keys():
            return self.map[value]
        else:
            self.index=self.index+1
            self.map[value]=np.int64(self.index)
            return np.int64(self.index)
        
    def revrese(self,value):
        return list(self.map.keys())[list(self.map.values()).index(value)]
    
def array_key(a):
    if (type(a) is np.array or type(a) is np.ndarray):
        return ",".join(map(str, a))
    else:
        return a


def key_to_array(s):
   
    if (isinstance(s, str)):
        #print(str(s))
        return [float(x) for x in s.split(",")][::-1]
    else:
        return s
    