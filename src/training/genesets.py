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

    
def get_genesets(what="all_union"):
    mito = pd.read_csv("../results/2022May/genesets/mito.csv")["mito"].to_list()
    meta = pd.read_csv("../results/2022May/genesets/meta.csv")["meta"].to_list()
    de = pd.read_csv("../results/2022March/DEG/tables/DE_2_1.tsv", sep='\t')
    degs = de[(de["padj"] <= 0.1) & (abs(de["log2FoldChange"]) > 0.3)].index.to_series().apply(lambda x: x[:x.index(".")]).to_list()
    returned_set = {"DEG_or_meta_or_mito": set(degs) | set(meta) | set(mito)}
    indiv_sets = {
        "DEG": set(degs),
        "meta": set(meta),
        "mito": set(mito)
    }
    Or_sets = {
                "DEG_or_meta": set(degs) | set(meta),
                "DEG_or_mito": set(degs) | set(mito),
                "meta_or_mito": set(meta) | set(mito)
    }
    And_sets = {
                "DEG_and_meta": set(degs) & set(meta),
                "DEG_and_mito": set(degs) & set(mito),
                "meta_and_mito": set(meta) & set(mito),
                "DEG_and_meta_and_mito": set(degs) & set(meta) & set(mito)
    }
    
    if what == "all":
        returned_set.update(indiv_sets)
        returned_set.update(Or_sets)
        returned_set.update(And_sets)
    
    return returned_set