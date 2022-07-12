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
import optuna

import xgboost as xgb

#import lightgbm as lgb

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

param_strs = {"gbm": {"loss": {"type": "c", "choices": ["deviance"]},
                      "n_estimators": {"type": "c", "choices": [1000]},
                      "criterion": {"type": "c", "choices": ["friedman_mse",  "mse"]},
                      "max_features": {"type": "c", "choices": ["log2","sqrt"]},
                      "learning_rate": {"type": "f", "low": 1e-3, "high": 5e-1, "log": True},
                      "min_samples_split": {"type": "f", "low": 0.01, "high": 0.5},
                      "min_samples_leaf": {"type": "f", "low": 0.01, "high": 0.5},
                      "subsample": {"type": "f", "low": 0.1, "high": 1},
                      "max_depth": {"type": "i", "low": 2, "high": 10},
                      "n_iter_no_change": {"type": "c", "choices": [int(1e6)]}}, 
             "log_reg": {'C': {"type": "f", "low": 1e-3, "high": 1e3, "log": True}, 
                        'class_weight': {"type": "c", "choices": ["balanced"]}, 
                        'max_iter': {"type": "c", "choices": [10000]}}, 
             "svm": {'C': {"type": "f", "low": 1e-3, "high": 1e3, "log": True}, 
                   'degree': {"type": "i", "low": 1, "high": 4}, 
                   'kernel': {"type": "c", "choices": ['poly', 'rbf']}, 
                   'max_iter': {"type": "c", "choices": [5000]},
                   "probability": {"type": "c", "choices": [True]},
                   "gamma": {"type": "f", "low": 1e-3, "high": 1e3, "log": True}},
             "rf": {"n_estimators": {"type": "c", "choices": [1000]},
                 'max_features': {"type": "c", "choices": ['auto', 'sqrt']}, 
                 'max_depth': {"type": "i", "low": 5, "high": 40}, 
                 'min_samples_split': {"type": "i", "low": 3, "high": 10}, 
                 'min_samples_leaf': {"type": "i", "low": 2, "high": 10}, 
                 'bootstrap': {"type": "c", "choices": [True, False]}},
              "xgb": {"n_estimators": {"type": "c", "choices": [1000]},
                      "tree_method": {"type": "c", "choices": ["hist"]},
                      "max_depth": {"type": "i", "low": 2, "high": 15}, 
                      "learning_rate": {"type": "f", "low": 1e-3, "high": 1e-1, "log": True},
                      "gamma": {"type": "f", "low": 1e-8, "high": 10, "log": True},
                      "reg_lambda": {"type": "f", "low": 1e-8, "high": 10, "log": True},
                      "colsample_bytree": {"type": "f", "low": 0.2, "high": 0.8, "log": True},
                      "early_stopping_rounds": {"type": "c", "choices": [25]},
                      "eval_metric": {"type": "c", "choices": ["aucpr"]}
                     },
              "mlp": {"activation": {"type": "c", "choices": ["logistic", "tanh", "relu"]},
                     "hidden_layer_sizes": {"type": "c", "choices": [(512,), (512, 256), (512, 256, 128), (512, 128, 32)]},
                     "alpha": {"type": "f", "low": 1e-6, "high": 10, "log": True},
                     "early_stopping": {"type": "c", "choices": [True]},
                     "max_iter": {"type": "c", "choices": [1000]},
                      "solver": {"type": "c", "choices": ["lbfgs", "sgd", "adam"]},
                     "momentum": {"type": "f", "low": 0.2, "high": 0.9, "log": True},
                     }
             }

clf_names = {"log_reg": LogisticRegression,
             "svm": SVC,
             "rf": RandomForestClassifier,
             "gbm": GradientBoostingClassifier,
             "xgb": xgb.XGBClassifier,
             "mlp": MLPClassifier
            # "lgb": lgb.LGBMClassifier
            }


def training(X, y, clf_type=LogisticRegression, cv=5, **kwargs):
    clf = make_pipeline(clf_type(**kwargs))
    scores = cross_val_score(clf, X, y, cv=cv, verbose=1)
    sns.histplot(scores)
    print(sum(scores) / len(scores))
    return scores


def GS_training(X, y, parameters, clf_type=LogisticRegression, cv=5, **kwargs):
    clf = RandomizedSearchCV(clf_type(**kwargs), parameters, cv=cv, 
                             verbose=1, scoring='roc_auc', n_jobs=1, error_score=0.5)
    result = clf.fit(X, y)
    return result


class Trainer:
    def __init__(self, data, parameters, clf_name, **kwargs):
        self.data = data
        self.fitted_clf = GS_training(data.X_train, data.y_train, parameters, clf_type=clf_names[clf_name], **kwargs)
        self.clf_name = clf_name
    
    def score(self):
        return self.fitted_clf.best_score_
    
    def pred_test(self):
        clf = self.fitted_clf.best_estimator_.fit(self.data.X_train, self.data.y_train)
        
        test_cols = self.data.testing_cols
        y_pred = clf.predict(self.data.X_test)
        return pd.DataFrame({"sample": test_cols, f"{self.clf_name}_prediction": y_pred})
    
    
def get_params(trial, param_struct):
    method_map = {"c": trial.suggest_categorical,
                 "f": trial.suggest_float,
                 "i": trial.suggest_int,
                 "d": trial.suggest_discrete_uniform,
                 "lu": trial.suggest_loguniform,
                 "u": trial.suggest_uniform}
    
    return {k: method_map[v["type"]](k, **{vk: vv for vk, vv in v.items() if "type" != vk}) 
            for k, v in param_struct.items()}


def objective(trial, 
              X, y, 
              ps, n_f_max, n_f_min, 
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
                fitted_clf = clf(**param_grid).fit(selected_X_train, y[train], 
                                               sample_weight=[1 if yi != pos_label else pos_weight for yi in y[train]], eval_set=[(selected_X_test, y[test])])
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
    clfs = clf_names
    
    def __init__(self, clf_name, param_stucts, pos_label):
        self.clf_name = clf_name
        self.clf = self.clfs[clf_name]
        self.param_stuct = param_stucts[clf_name]
        self.pos_label = pos_label
        
    def hyper_opt(self, X, y, ftr_names, n_trials, file_name_prefix):
        self.study = optuna.create_study(direction="maximize", study_name=self.clf_name)
        func = lambda trial: objective(trial, X, y, self.param_stuct, 
                                       n_f_max=min(X.shape[0], X.shape[1])-1, n_f_min=1, ftr_names=ftr_names,
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
    