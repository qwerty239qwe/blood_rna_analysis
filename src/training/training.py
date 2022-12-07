import pandas as pd
import numpy as np

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression, LassoLars
from sklearn.manifold import TSNE, Isomap, MDS, SpectralEmbedding, LocallyLinearEmbedding
import umap

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, VarianceThreshold
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, log_loss, precision_recall_curve, auc
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE, BorderlineSMOTE
from imblearn.under_sampling import EditedNearestNeighbours , TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek

import optuna
import xgboost as xgb


IMB_SAMPLERS = {"None": None,
                "SMOTE": SMOTE, 
                "ADASYN": ADASYN, 
                "SVMSMOTE": SVMSMOTE,
                "BorderlineSMOTE": BorderlineSMOTE,
                "EditedNearestNeighbours": EditedNearestNeighbours,
                "TomekLinks": TomekLinks,
                "SMOTEENN": SMOTEENN,
                "SMOTETomek": SMOTETomek}

FTR_SELS = {"f_classif": f_classif, 
            #"mutual_info_classif": mutual_info_classif
           }

    
def get_params(trial, param_struct):
    method_map = {"c": trial.suggest_categorical,
                 "f": trial.suggest_float,
                 "i": trial.suggest_int,
                 "d": trial.suggest_discrete_uniform,
                 "lu": trial.suggest_loguniform,
                 "u": trial.suggest_uniform}
    obj = {"smote": SMOTE, 
           "enn": EditedNearestNeighbours }
    obj_dict = {}
    pops = []
    for k, v in param_struct.items():
        if ":" in k:
            obj_name, obj_params = k.split(":")
            if obj_name not in obj_dict:
                obj_dict[obj_name] = {"obj": obj[obj_name], "params": {obj_params: method_map[v["type"]](k, **{vk: vv for vk, vv in v.items() if "type" != vk})}}
            else:
                obj_dict[obj_name]["params"].update({obj_params: method_map[v["type"]](k, **{vk: vv for vk, vv in v.items() if "type" != vk})})
            pops.append(k)
    param_struct = {k: v for k, v in param_struct.items() if k not in pops}
    result = {k: method_map[v["type"]](k, **{vk: vv for vk, vv in v.items() if "type" != vk}) 
            for k, v in param_struct.items()}
    
    result.update({k: v["obj"](**v["params"]) 
                   for k, v in obj_dict.items()})
    
    return result


def objective(trial, 
              X, 
              y, 
              ps, 
              ss,
              n_f_max, 
              n_f_min, 
              ftr_names,
              file_name_prefix,
              clf,
              concat_ftrs, 
              use_sample_weight=True,
              use_evalset=False,
              do_pca_trans=False,
              do_resampling=True,
              pos_label = 2,
              pos_weight = 50,
              err_score=0.):
    param_grid = get_params(trial, ps)
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    n_features = trial.suggest_int("n_features", low=n_f_min, high=n_f_max)
    sampler_name = trial.suggest_categorical("resampler", list(IMB_SAMPLERS.keys()) if do_resampling else ["None"])
    fs_method = trial.suggest_categorical("ftr_selection", list(FTR_SELS.keys()))
    
    sampler = IMB_SAMPLERS[sampler_name]
    
    if sampler is not None:
        sampler_param_grid = get_params(trial, ss[sampler_name])
    #label_encoder = LabelEncoder().fit(y)
    #pos_label = label_encoder.transform([pos_label])[0]
    #y = label_encoder.transform(y.copy())
    ftr_sel_records = []
    pca_load_records = []
    scores = []
    for i, (train, test) in enumerate(cv.split(X, y)):
        X_train, y_train, X_test = X[train], y[train], X[test]
        
        if do_pca_trans:
            pca = PCA(n_components=n_features)
            pca.fit(X_train)
            X_train, X_test = pca.transform(X_train), pca.transform(X_test)
            
            loading_data = abs((pca.components_) * np.array([pca.explained_variance_ratio_]).T).sum(axis=0)
            
            pca_loading = pd.DataFrame(data=loading_data, index=ftr_names).T
            pca_loading["iter"] = i
            pca_load_records.append(pca_loading)
        else:
            selector = SelectKBest(FTR_SELS[fs_method], k=n_features).fit(X_train, y_train)
            X_train, X_test = selector.transform(X_train), selector.transform(X_test)
            ftr_sel_records.extend(ftr_names[selector.get_support()])
        try:
            if sampler is not None:
                X_train, y_train = sampler(**sampler_param_grid).fit_resample(X_train, y_train)
        except RuntimeError as e:
            raise optuna.TrialPruned(e)
        except ValueError as e:
            raise optuna.TrialPruned(e)
        
        if use_sample_weight:
            if pos_weight > 1:
                weight_dic = {yi: 1 if yi != pos_label else pos_weight for yi in set(y_train)}
            else:
                weight_dic = {yi: int(1 / pos_weight) if yi != pos_label else 1 for yi in set(y_train)}
            try:
                if use_evalset:
                    fitted_clf = clf(**param_grid).fit(X_train, 
                                                       y_train, 
                                                       sample_weight=[weight_dic[yi] for yi in y_train], 
                                                       eval_set=[(X_test, y[test])])
                else:
                    fitted_clf = clf(**param_grid).fit(X_train, y_train, 
                                                       sample_weight=[weight_dic[yi] for yi in y_train])
            except ValueError as e:
                raise optuna.TrialPruned(e)
        else:
            fitted_clf = clf(**param_grid).fit(X_train, y_train)
        pred_prob = fitted_clf.predict_proba(X_test)
        try:
            p, r, _ = precision_recall_curve(y[test] == pos_label, pred_prob[:, fitted_clf.classes_ == pos_label])
            scores.append(auc(r, p))
        except:
            print("problem occur during BO, return err_score !")
            print(y[test], pred_prob)
            scores.append(err_score)
    if not do_pca_trans:
        concat_ftrs += [pd.DataFrame({"features": ftr_sel_records})]
        print(f"ftr has {len(concat_ftrs)} results")
    else:
        concat_ftrs += pca_load_records
    return np.mean(scores)


class StudyScheduler:
    clfs = {"log_reg": LogisticRegression,
             "svm": SVC,
             "rf": RandomForestClassifier,
             "gbm": GradientBoostingClassifier,
             "xgb": xgb.XGBClassifier,
             "mlp": MLPClassifier,
             "knn": KNeighborsClassifier
            }
    
    def __init__(self, clf_name, param_stucts, sampling_param_structs, pos_label, pos_weight=50,
                 do_pca_transform=False, do_resampling=True):
        self.clf_name = clf_name
        self.clf = self.clfs[clf_name]
        self.param_stuct = param_stucts[clf_name]
        self.sampling_param_structs = sampling_param_structs
        self.pos_weight = pos_weight
        self.pos_label = pos_label
        self.do_pca_transform = do_pca_transform
        self.do_resampling=do_resampling
        self.concated_ftrs = []
        
    def hyper_opt(self, X, y, ftr_names, n_trials, file_name_prefix):
        self.study = optuna.create_study(direction="maximize", study_name=self.clf_name)
        func = lambda trial: objective(trial, X, y, self.param_stuct, 
                                       ss=self.sampling_param_structs,
                                       n_f_max=min(int(X.shape[0] * 0.8), X.shape[1])-1, 
                                       n_f_min=1, 
                                       ftr_names=ftr_names,
                                       concat_ftrs=self.concated_ftrs,
                                       do_pca_trans=self.do_pca_transform,
                                       do_resampling=self.do_resampling,
                                       pos_label=self.pos_label,
                                       pos_weight=self.pos_weight,
                                       use_sample_weight=(self.clf_name not in ["mlp", "knn"]),
                                       use_evalset=(self.clf_name == "xgb"),
                                       file_name_prefix=file_name_prefix, clf=self.clf)
        self.study.optimize(func, n_trials=n_trials, gc_after_trial = True)
        self._best_sampler_name = self.study.best_params["resampler"]
        self._best_fs_name = self.study.best_params["ftr_selection"]
        self._best_sampler_param_names = list(self.sampling_param_structs[self._best_sampler_name].keys()) if self._best_sampler_name != "None" else ["None"]
        self.study.trials_dataframe().to_csv(f"{file_name_prefix}_bo_trials.csv")
        if not self.do_pca_transform:
            pd.concat(self.concated_ftrs, axis=0).to_csv(f"{file_name_prefix}_bo_ftrs.csv")
        else:
            pd.concat(self.concated_ftrs, axis=0).to_csv(f"{file_name_prefix}_pca_loading.csv")
        
    @property
    def best_clf(self):
        return self.clf(**{k: v for k, v in self.study.best_params.items() if k not in ['n_features', 
                                                                                        "early_stopping_rounds", 
                                                                                        "eval_metric", 
                                                                                        "resampler", 
                                                                                        "ftr_selection"] + self._best_sampler_param_names})
    @property
    def best_sampler(self):
        if self._best_sampler_name == "None":
            return
        
        obj = {"smote": SMOTE, 
               "enn": EditedNearestNeighbours }
        sampler_params = {k: v for k, v in self.study.best_params.items() if k in self._best_sampler_param_names}
        to_pops = []
        to_join = {}
        for k, v in sampler_params.items():
            if ":" in k:
                to_pops.append(k)
                sml_name, sml_param_name = k.split(":")
                if sml_name in to_join:
                    to_join[sml_name].update({sml_param_name: v})
                else:
                    to_join[sml_name] = {sml_param_name: v}
        sampler_params = {k: v for k, v in sampler_params.items() if k not in to_pops}
        sampler_params.update({k: obj[k](**v) for k, v in to_join.items()})
        
        return IMB_SAMPLERS[self._best_sampler_name](**sampler_params)
    
    @property
    def best_params(self):
        return self.study.best_params
    
    @property
    def best_ftr_sel(self):
        return FTR_SELS[self._best_fs_name]
    
    @property
    def best_n_features(self):
        return self.study.best_params['n_features']
    
    @property
    def best_feature_idx(self):
        n = self.best_n_features
        return self._features.index.isin(self._features.sort_values()[-n:].index)
    