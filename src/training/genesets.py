import pandas as pd
import itertools
from functools import reduce


class GeneSet:
    def __init__(self, geneset):
        self._geneset = geneset
        
        
class SetContainer:
    def __init__(self, sets=None):
        self.sets = {} if sets is None else sets
        
    def __getitem__(self, key):
        return self.sets[key]
    
    def __setitem__(self, key, newset):
        self.sets[key] = newset
        
    def _get_intersection(self, keys):
        return reduce(set.intersection, [self.sets[k] for k in keys])
    
    def _get_union(self, keys):
        return reduce(set.union, [self.sets[k] for k in keys])
    
    def items(self):
        return self.sets.items()
    
    def get_n_items(self):
        all_keys = list(self.sets.keys())
        _n_dic = {}
        
        for i in range(2, len(all_keys)):
            combs = itertools.combinations(all_keys, i)
            _n_dic.update({"n".join(cmb): self._get_intersection(cmb) for cmb in combs})
        return _n_dic.items()
        
    def get_u_items(self):
        all_keys = list(self.sets.keys())
        _u_dic = {}
        
        for i in range(2, len(all_keys)):
            combs = itertools.combinations(all_keys, i)
            _u_dic.update({"n".join(cmb): self._get_union(cmb) for cmb in combs})
        return _u_dic.items()
        