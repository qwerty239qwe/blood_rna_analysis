import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json

from pathlib import Path
import argparse

from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, VarianceThreshold

from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, f1_score, balanced_accuracy_score, roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay

from sklearn.preprocessing import LabelEncoder

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
    config = load_config(f"./training/configs/{args.filename}")
    print(config)
    
    hyperparams = load_hp(Path(__file__) / "../configs/hp_search_space.yml")
    sampling_hp = load_hp(Path(__file__) / "../configs/sampling_hp_search_space.yml")
    
    labels = pd.read_csv(config["label_path"])
    used_unit = config["used_unit"]
    ds_name = config["ds_name"]
    use_specified_gs, use_pca, use_flux = config["use_gs"], config["use_pca"], config["use_flux"]
    use_resampling = config["use_resampling"] if "use_resampling" in config else True
    pos_weight = float(config["pos_weight"]) if "pos_weight" in config else 50
    data = pd.read_csv(config["data_path"], index_col=0).dropna(how="any")
    
    print("features, labels, and data are loaded.")
    
    if use_specified_gs:
        print("preselected genes are used")
        genesets = get_genesets()
    else:
        genesets = {"all_genes": set(data.index)}
        
    dir_path = Path(config["output_dir"]) / Path(f"{used_unit}/{'use_gs' if use_specified_gs else 'no_gs'}/{'use_pca' if use_pca else 'use_gene'}/{'use_flux' if use_flux else 'no_flux'}")
    output_mod_dir = dir_path / "models"
    if not dir_path.is_dir():
        dir_path.mkdir(parents=True)
    if not output_mod_dir.is_dir():
        output_mod_dir.mkdir(parents=True)
    
    pos_label = 2
    m_name = 'and'.join(config["trained_models"])
    colors = ["navy", "turquoise", "darkorange", "cornflowerblue", "teal"]

    for si, used_set in genesets.items():
        non_gene = set([ng for ng in data.index if "ENSG" not in ng])
        used_data = data.loc[data.index.isin(used_set | non_gene), :]
        X = used_data.values.T
        y = labels["group"].values
        label_encoder = LabelEncoder().fit(y)
        pos_label = label_encoder.transform([pos_label])[0]
        y = label_encoder.transform(y.copy())
        ftr = used_data.index.values.reshape([1, used_data.index.values.shape[0]])
        # filter out zero vars
        val_sel = VarianceThreshold()
        X = val_sel.fit_transform(X)
        ftr = val_sel.transform(ftr).T.squeeze(-1)
        results = []
        #for r in range(n_repeat):
        for c_name in config["trained_models"]:
            data_row = {}
            scores = []
            false_preds = []
            best_n_ftrs = []
            best_params = []
            skf = StratifiedKFold(n_splits=5, shuffle=True)
            pr_curve_dic = {"precision": [], "recall": [], "iter": []}
            pr_th_dic = {"threshold": [], "iter": []}
            roc_curve_dic = {"FPR": [], "TPR": [], "iter": []}
            roc_th_dic = {"threshold": [], "iter": []}
            f1_scores = []
            balanced_acs = []
            aurocs = []
            pr_displays = []
            roc_displays = []
            
            for i, (train, test) in enumerate(skf.split(X, y)):
                
                
                stud = StudyScheduler(clf_name=c_name, param_stucts=hyperparams, sampling_param_structs=sampling_hp, 
                                      do_pca_transform=use_pca, do_resampling=use_resampling, pos_label=pos_label, pos_weight=pos_weight)
                X_train, y_train = X[train], y[train]
                stud.hyper_opt(X=X_train, y=y_train, ftr_names=ftr, n_trials=200, 
                               file_name_prefix=str(dir_path / f"{ds_name}_{c_name}_{i}"))
                if use_pca:
                    pca = PCA(stud.best_n_features)
                    pca.fit(X_train)
                    X_train, X_test = pca.transform(X_train), pca.transform(X[test])
                else:
                    fsel = SelectKBest(stud.best_ftr_sel, k=stud.best_n_features).fit(X_train, y_train)
                    X_train, X_test = fsel.transform(X_train), fsel.transform(X[test])
                if stud.best_sampler is not None:
                    X_train, y_train = stud.best_sampler.fit_resample(X_train, y_train)
                clf = stud.best_clf.fit(X_train, y_train)
                joblib.dump(clf, output_mod_dir / f"{c_name}_{i}.pkl")
                with open(str(output_mod_dir / f'{c_name}_{i}_data.json'), 'w', encoding='utf-8') as annot_f:
                    json.dump({"train_index": train.tolist(), "test_index": test.tolist(), "n_k": stud.best_n_features}, annot_f, ensure_ascii=False, indent=4)
                
                assert X_test.shape[1] < X.shape[0]

                y_pred = clf.predict(X_test)
                y_prob = clf.predict_proba(X_test)
                
                precision, recall, thres = precision_recall_curve(y[test] == pos_label, y_prob[:, clf.classes_ == pos_label])
                pr_curve_dic["precision"].extend(precision)
                pr_curve_dic["recall"].extend(recall)
                pr_curve_dic["iter"].extend([i for _ in range(len(recall))])
                pr_th_dic["threshold"].extend(thres)
                pr_th_dic["iter"].extend([i for _ in range(len(thres))])
                pr_displays.append(PrecisionRecallDisplay(precision=precision, recall=recall, pos_label=pos_label))
                
                fpr, tpr, roc_ths = roc_curve(y[test] == pos_label, y_prob[:, clf.classes_ == pos_label])
                roc_curve_dic["FPR"].extend(fpr)
                roc_curve_dic["TPR"].extend(tpr)
                roc_th_dic["threshold"].extend(roc_ths)
                roc_th_dic["iter"].extend([i for _ in range(len(roc_ths))])
                roc_curve_dic["iter"].extend([i for _ in range(len(fpr))])
                roc_displays.append(RocCurveDisplay(fpr=fpr, tpr=tpr, pos_label=pos_label))

                if i == 0:
                    confs = confusion_matrix(y[test], y_pred)
                else:
                    confs += confusion_matrix(y[test], y_pred)
                f1_scores.append(f1_score(y[test], y_pred, pos_label=pos_label))
                balanced_acs.append(balanced_accuracy_score(y[test], y_pred))
                false_preds.append(test[y[test] != y_pred])
                best_n_ftrs.append(stud.best_n_features)
                best_params.append(stud.best_params)
                
                scores.append(auc(recall, precision))
                aurocs.append(auc(fpr, tpr))
                print(f"test score: {scores[-1]}")
                pd.DataFrame(pr_curve_dic).to_csv(dir_path / f"prauc_{ds_name}_{c_name}.csv")
                pd.DataFrame(pr_th_dic).to_csv(dir_path / f"prauc_thres_{ds_name}_{c_name}.csv")
                
                pd.DataFrame(roc_curve_dic).to_csv(dir_path / f"auroc_{ds_name}_{c_name}.csv")
                pd.DataFrame(roc_th_dic).to_csv(dir_path / f"auroc_thres_{ds_name}_{c_name}.csv")
                data_row["mean score"] = np.mean(scores)
                data_row["scores"] = scores
                data_row["false_preds"] = np.concatenate(false_preds)
                data_row["TP"] = confs[0, 0]
                data_row["TN"] = confs[1, 1]
                data_row["FP"] = confs[1, 0]
                data_row["FN"] = confs[0, 1]
                data_row["F1_score"] = f1_scores
                data_row["mean_f1_score"] = np.mean(f1_scores)
                data_row["balanced_accuracy_score"] = balanced_acs
                data_row["mean_BAC"] = np.mean(balanced_acs)
                data_row["AUROC"] = aurocs
                data_row["mean_AUROC"] = np.mean(aurocs)

                # exp info
                data_row["classifier"] = c_name
                data_row["best_params"] = best_params
                data_row["n_k"] = best_n_ftrs
                data_row["ds"] = ds_name
                data_row["gs"] = si
                results.append(data_row)
                pd.DataFrame(results).to_csv(dir_path / f"result_{ds_name}_{m_name}.csv")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
            for i in range(len(roc_displays)):
                pr_displays[i].plot(ax=ax1, name=f"Outer test fold {i+1} (AUC = {scores[i]})", color=colors[i])
                roc_displays[i].plot(ax=ax2, name=f"Outer test fold {i+1} (AUC = {aurocs[i]})", color=colors[i])
            
            plt.savefig(dir_path / f"AUC_{ds_name}_{c_name}.png", dpi=600)
        
if __name__ == "__main__":
    main()