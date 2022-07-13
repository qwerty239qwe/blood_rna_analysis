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

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier

from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE, BorderlineSMOTE
from imblearn.under_sampling import EditedNearestNeighbours , TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek

import optuna
import xgboost as xgb


IMB_SAMPLERS = {"SMOTE": SMOTE, "ADASYN": ADASYN, "SVMSMOTE": SVMSMOTE}

    
def get_params(trial, param_struct):
    method_map = {"c": trial.suggest_categorical,
                 "f": trial.suggest_float,
                 "i": trial.suggest_int,
                 "d": trial.suggest_discrete_uniform,
                 "lu": trial.suggest_loguniform,
                 "u": trial.suggest_uniform}
    obj = {"smote": SMOTE, "enn": EditedNearestNeighbours }
    obj_dict = {}
    pops = []
    for k, v in param_struct.items():
        if ":" in k:
            obj_name, obj_params = k.split(":")
            if obj_name in obj_dict:
                obj_dict[obj_name] = {"obj": obj[obj_name], "params": {obj_params: method_map[v["type"]](k, **{vk: vv for vk, vv in v.items() if "type" != vk})}}
            else:
                obj_dict[obj_name]["params"].update({obj_params: method_map[v["type"]](k, **{vk: vv for vk, vv in v.items() if "type" != vk})})
            pops.append(k)
    param_struct = {k: v for k, v in param_struct.items() if k not in pops}
    param_struct.update({k: v["obj"](**v["params"]) for k, v in obj_dict.items()})
    
    return {k: method_map[v["type"]](k, **{vk: vv for vk, vv in v.items() if "type" != vk}) 
            for k, v in param_struct.items()}


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
              use_sample_weight=True,
              use_evalset=False,
              pos_label = 2,
              pos_weight = 50,
              err_score=0.):
    param_grid = get_params(trial, ps)
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    n_features = trial.suggest_int("n_features", low=n_f_min, high=n_f_max)
    sampler = trial.suggest_categorical("resampler", list(IMB_SAMPLERS.keys()))
    sampler_param_grid = get_params(trial, ss[sampler])
    label_encoder = LabelEncoder().fit(y)
    pos_label = label_encoder.transform([pos_label])[0]
    y = label_encoder.transform(y.copy())
    ftr_sel_records = []
    scores = []
    for i, (train, test) in enumerate(cv.split(X, y)):
        selector = SelectKBest(mutual_info_classif, k=n_features).fit(X[train], y[train])
        selected_X_train, selected_X_test = selector.transform(X[train]), selector.transform(X[test])
        ftr_sel_records.extend(ftr_names[selector.get_support()])
        if use_sample_weight:
            if use_evalset:
                fitted_clf = clf(**param_grid).fit(selected_X_train, 
                                                   y[train], 
                                                   sample_weight=[1 if yi != pos_label else pos_weight for yi in y[train]], 
                                                   eval_set=[(selected_X_test, y[test])])
            else:
                fitted_clf = clf(**param_grid).fit(selected_X_train, y[train], 
                                                   sample_weight=[1 if yi != pos_label else pos_weight for yi in y[train]])
        else:
            fitted_clf = clf(**param_grid).fit(selected_X_train, y[train])
        pred_prob = fitted_clf.predict_proba(selected_X_test)
        try:
            p, r, _ = precision_recall_curve(y[test] == pos_label, pred_prob[:, fitted_clf.classes_ == pos_label])
            scores.append(auc(r, p))
        except:
            print("problem occur during BO, return err_score !")
            print(y[test], pred_prob)
            scores.append(err_score)
    pd.DataFrame({"features": ftr_sel_records}).to_csv(f"{file_name_prefix}_bo_ftrs.csv")
    return np.mean(scores)


class StudyScheduler:
    clfs = {"log_reg": LogisticRegression,
             "svm": SVC,
             "rf": RandomForestClassifier,
             "gbm": GradientBoostingClassifier,
             "xgb": xgb.XGBClassifier,
             "mlp": MLPClassifier
            }
    
    def __init__(self, clf_name, param_stucts, sampling_param_structs, pos_label):
        self.clf_name = clf_name
        self.clf = self.clfs[clf_name]
        self.param_stuct = param_stucts[clf_name]
        self.sampling_param_structs = sampling_param_structs
        self.pos_label = pos_label
        
    def hyper_opt(self, X, y, ftr_names, n_trials, file_name_prefix):
        self.study = optuna.create_study(direction="maximize", study_name=self.clf_name)
        func = lambda trial: objective(trial, X, y, self.param_stuct, 
                                       ss=self.sampling_param_structs,
                                       n_f_max=min(X.shape[0], X.shape[1])-1, 
                                       n_f_min=1, 
                                       ftr_names=ftr_names,
                                       pos_label=self.pos_label,
                                       use_sample_weight=(self.clf_name != "mlp"),
                                       use_evalset=(self.clf_name == "xgb"),
                                       file_name_prefix=file_name_prefix, clf=self.clf)
        self.study.optimize(func, n_trials=n_trials, gc_after_trial = True)
        self.study.trials_dataframe().to_csv(f"{file_name_prefix}_bo_trials.csv")
        
    @property
    def best_clf(self):
        return self.clf(**{k: v for k, v in self.study.best_params.items() if k not in ['n_features', "early_stopping_rounds", "eval_metric"]})
    
    @property
    def best_params(self):
        return self.study.best_params
    
    @property
    def best_n_features(self):
        return self.study.best_params['n_features']
    
    @property
    def best_feature_idx(self):
        n = self.best_n_features
        return self._features.index.isin(self._features.sort_values()[-n:].index)
    