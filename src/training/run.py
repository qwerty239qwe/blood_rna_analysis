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

from util import *
from training import *
from genesets import get_genesets


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help ='path to the config file',
                        default='configs/run1.yml')
    args = parser.parse_args()
    config = load_config(args.filename)
    print(config)
    
    hyperparams = load_hp(Path(__file__) / "../configs/hp_search_space.yml")
    sampling_hp = load_hp(Path(__file__) / "../configs/sampling_hp_search_space.yml")
    
    labels = pd.read_excel(config["label_path"])
    used_unit = config["used_unit"]
    ds_name = config["ds_name"]
    data = pd.read_csv(config["data_path"], index_col=0).dropna(how="any")
    
    print("features, labels, and data are loaded.")
    
    genesets = get_genesets()
    dir_path = Path(config["output_dir"])
    pos_label = 2

    for si, used_set in genesets.items():
        non_gene = set([ng for ng in data.index if "ENSG" not in ng])
        used_data = data.loc[data.index.isin(used_set | non_gene), :]
        X = used_data.values.T
        y = labels["group"].values
        ftr = used_data.index
        results = []
        #for r in range(n_repeat):
        for c_name in config["trained_models"]:
            data_row = {}
            scores = []
            false_preds = []
            best_n_ftrs = []
            skf = StratifiedKFold(n_splits=5, shuffle=True)
            pr_curve_dic = {"precision": [], "recall": [], "iter": []}

            for i, (train, test) in enumerate(skf.split(X, y)):
                stud = StudyScheduler(clf_name=c_name, param_stucts=hyperparams, sampling_param_structs=sampling_hp, pos_label=pos_label)
                stud.hyper_opt(X=X[train], y=y[train], ftr_names=ftr, n_trials=100, 
                               file_name_prefix=str(dir_path / f"{ds_name}_{used_unit}_{si}_{c_name}_{i}"))
                fsel = SelectKBest(mutual_info_classif, k=stud.best_n_features).fit(X[train], y[train])
                X_train, X_test = fsel.transform(X[train]), fsel.transform(X[test])
                X_train, y_train = stud.best_sampler.fit_resample(X_train, y[train])
                clf = stud.best_clf.fit(X_train, y_train)
                
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
        m_name = 'and'.join(config["trained_models"])
        pd.DataFrame(results).to_csv(dir_path / f"result_{ds_name}_u_{used_unit}_gs_{si}_model_{m_name}.csv")
        
if __name__ == "__main__":
    main()