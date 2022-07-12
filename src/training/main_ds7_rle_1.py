import pandas as pd
import numpy as np
import sys

from pathlib import Path
import argparse

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeCV

from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, VarianceThreshold
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV

from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, log_loss, roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
import optuna

from training import *


# genesets
def get_genesets():
    mito = pd.read_csv("../results/2022May/genesets/mito.csv")["mito"].to_list()
    meta = pd.read_csv("../results/2022May/genesets/meta.csv")["meta"].to_list()
    de = pd.read_csv("../results/2022March/DEG/tables/DE_2_1.tsv", sep='\t')
    degs = de[(de["padj"] <= 0.1) & (abs(de["log2FoldChange"]) > 0.3)].index.to_series().apply(lambda x: x[:x.index(".")]).to_list()
    
    
    return {
        # "DEG": set(degs),
            # "meta": set(meta),
            # "mito": set(mito),
            # "DEG_or_meta": set(degs) | set(meta),
            # "DEG_or_mito": set(degs) | set(mito),
            # "meta_or_mito": set(meta) | set(mito),
            "DEG_or_meta_or_mito": set(degs) | set(meta) | set(mito),
            # "DEG_and_meta": set(degs) & set(meta),
            # "DEG_and_mito": set(degs) & set(mito),
            # "meta_and_mito": set(meta) & set(mito),
            # "DEG_and_meta_and_mito": set(degs) & set(meta) & set(mito)
           }



def main():
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('integers', metavar='N', type=int, nargs='+',
    #                 help='an integer for the accumulator')
    
    # features = pd.read_csv("../results/2022May/datasets/fs1_rle.csv", index_col=0)
    labels = pd.read_excel("../data/all_info/labels_may_mg.xlsx")
    used_unit = "log_rle"
    ds_name = "ds7"
    data = pd.read_csv(f"../results/2022May/datasets/ds7_{used_unit}.csv", index_col=0).dropna(how="any")
    
    print("features, labels, and data are loaded.")
    
    genesets = get_genesets()
    dir_path = Path(f"../results/2022May/CLF/{ds_name}")
    pos_label = 2

    for si, used_set in genesets.items():
        non_gene = set([ng for ng in data.index if "ENSG" not in ng])
        used_data = data.loc[data.index.isin(used_set | non_gene), :]
        X = used_data.values.T
        y = labels["group"].values
        ftr = used_data.index
        results = []
        #for r in range(n_repeat):
        for c_name in ["svm"]:
            data_row = {}
            scores = []
            false_preds = []
            best_n_ftrs = []
            skf = StratifiedKFold(n_splits=5, shuffle=True)
            pr_curve_dic = {"precision": [], "recall": [], "iter": []}

            for i, (train, test) in enumerate(skf.split(X, y)):
                stud = StudyScheduler(clf_name=c_name, param_stucts=param_strs, pos_label=pos_label)
                stud.hyper_opt(X=X[train], y=y[train], ftr_names=ftr, n_trials=50, 
                               file_name_prefix=str(dir_path / f"{ds_name}_{used_unit}_{si}_{c_name}_{i}"))
                fsel = SelectKBest(mutual_info_classif, k=stud.best_n_features).fit(X[train], y[train])
                X_train, X_test = fsel.transform(X[train]), fsel.transform(X[test])
                clf = stud.best_clf.fit(X_train, y[train])
                assert X_test.shape[1] < X.shape[0]

                y_pred = clf.predict(X_test)
                y_prob = clf.predict_proba(X_test)
                precision, recall, _ = precision_recall_curve(y[test] == pos_label, y_prob[:, clf.classes_ == pos_label])
                pr_curve_dic["precision"].extend(precision)
                pr_curve_dic["recall"].extend(recall)
                #roc_curve_dic["thresholds"].extend(thresholds)
                pr_curve_dic["iter"].extend([i for _ in range(len(precision))])

                if i == 0:
                    confs = confusion_matrix(y[test], y_pred)
                else:
                    confs += confusion_matrix(y[test], y_pred)
                false_preds.append(test[y[test] != y_pred])
                best_n_ftrs.append(stud.best_n_features)
                scores.append(auc(recall, precision))
                print(f"test score: {scores[-1]}")
            pd.DataFrame(pr_curve_dic).to_csv(dir_path / f"auc_{ds_name}_{used_unit}_{si}_{c_name}.csv")

            data_row["score"] = np.mean(scores)
            data_row["false_preds"] = np.concatenate(false_preds)
            data_row["TP"] = confs[0, 0]
            data_row["TN"] = confs[1, 1]
            data_row["FP"] = confs[1, 0]
            data_row["FN"] = confs[0, 1]

            # exp info
            data_row["classifier"] = c_name
            data_row["n_k"] = best_n_ftrs
            data_row["ds"] = ds_name
            data_row["gs"] = si
            results.append(data_row)
        pd.DataFrame(results).to_csv(dir_path / f"result_{ds_name}_u_{used_unit}_gs_{si}_2.csv")
                
if __name__ == "__main__":
    main()