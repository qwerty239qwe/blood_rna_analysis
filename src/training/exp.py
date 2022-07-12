import pandas as pd
from .training import *
from .exp import *


class FeatureExp:
    def __init__(self, data, min_k, max_k):
        self._min_k if min_k > 1 else len(data.columns) * min_k
        self._max_k if max_k > 1 else len(data.columns) * max_k
        self._min_k = int(self._min_k) + (1 if int(self._min_k) < self._min_k else 0)
        self._max_k = int(self._max_k) + (1 if int(self._max_k) < self._max_k else 0)
        
        assert self._max_k < len(data.columns)
        assert self._min_k > 0
    