import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold, mutual_info_classif, SelectFdr

def transform_func(x):
    return np.log2(x + 1)


class Data:
    def __init__(self, 
                 label_file_name="../data/all_info/labels.csv", 
                 training_groups = ["G1", "G2"],
                 testing_groups = ["G3"],
                 to_testing = ['R042A', 'R055A', 'C002A', 'R035A'],
                 outliers=[]):
        labels = pd.read_csv(label_file_name, sep='\t')
        self.outliers = outliers
        gb = labels.groupby("group")["sample"].apply(lambda x: list(x))
        self.groups = {f"G{i+1}" if i != 5 else "G4": [g + "A" for g in gs] for i, gs in enumerate(gb)}
        self.groups = {k: [vi for vi in v if vi not in outliers] for k, v in self.groups.items()}
        labels.loc[:, "sample"] = labels["sample"].apply(lambda x: x+ "A")
        self.label_to_group = {s: "G" + (f"{g}" if g != 5 else "4") for s, g in zip(labels["sample"], labels["group"])}
        self.training_cols = list(set([s for s, g in self.label_to_group.items() if g in training_groups]) - set(to_testing))
        self.testing_cols = to_testing + [s for s, g in self.label_to_group.items() if g in testing_groups]
        
        self.X_dfs = {}
        self.X_train_dic, self.X_test_dic = {}, {}
        self.y_train = np.array([self.label_to_group[c] for c in self.training_cols])
        self.features = {}
        
    @property
    def X_train(self):
        return np.concatenate([x for x in self.X_train_dic.values()], axis=1)
    
    @property
    def X_test(self):
        return np.concatenate([x for x in self.X_test_dic.values()], axis=1)
        
    def register_x(self, df, prefix):
        df = df.copy()
        df.index = df.index.to_series().apply(lambda x: prefix + x)
        self.X_dfs[prefix] = df
    
    def concat_X(self, x_names, k=4000, feature_name="metabolic"):
        mg_df = pd.concat([self.X_dfs[name] for name in x_names], axis=0).fillna(0)
        mg_df = mg_df.loc[:, [c for c in mg_df.columns if c not in self.outliers]]
        tv_df, test_df = mg_df.loc[:, self.training_cols], mg_df.loc[:, self.testing_cols]
        X, y = tv_df.values.T, np.array(self.y_train)

        clf = make_pipeline(
            StandardScaler(), VarianceThreshold(), 
            SelectKBest(f_classif, k=k)
                             )

        nX_m = clf.fit_transform(X, y)
        features = tv_df.index
        self.features[feature_name] = features[clf["variancethreshold"].get_support()][clf["selectkbest"].get_support()]
        self.X_train_dic[feature_name] = nX_m
        self.X_test_dic[feature_name] = clf.transform(test_df.fillna(0).values.T)