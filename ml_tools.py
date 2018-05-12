import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

from sklearn.metrics import (
    r2_score,
    explained_variance_score,
    mean_squared_error,
    mean_absolute_error,
    mean_squared_log_error,
    median_absolute_error
)

from visualisation import label_barchart


def make_cross_validation(models, scoring, X, y):
    results = OrderedDict()
    for name, model in models:
        kfold = KFold(n_splits=3)
        cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
        results[name] = cv_results

    results = pd.DataFrame(results)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    results.mean().plot.bar(ax=ax, yerr=results.std(), color='Navy')
    plt.title("Comparison of cross validation score for multiple predictors", y=-.2)
    plt.subplots_adjust(bottom=.2)
    label_barchart(plt.gca())
    return results

def pickle_models(models):

    for model in models:
        name = str(model.__class__).split(".")[-1][:-2]
        with open(f"output/{name}.pkl", 'wb') as f:
            pickle.dump(model, f)
            
            
def make_grid_search_clf(classifiers, clf_params, X_train, y_train, X_test, y_test, random=False, search_kw=None, save=True):
    best_models, scores = [], []
    results = pd.DataFrame(index=[item[0] for item in classifiers],
                           columns=["name", "params", "accuracy", "auc_score_tr", "auc_score_te",
                                    "precision", "recall", "fscore", "support", "TP", "FP", "FN", "TN"])

    if random:
        SearchCV = RandomizedSearchCV
    else:
        SearchCV = GridSearchCV
    if search_kw is None:
        search_kw = dict(n_jobs=-1, return_train_score=True)

    for i, (name, clf) in enumerate(classifiers):
        params = clf_params[name]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            gs = SearchCV(clf, params, **search_kw).fit(X_train, y_train)
        best_models.append(gs.best_estimator_)
        y_pred = gs.predict(X_test)
        precision, recall, f_score, support = precision_recall_fscore_support(y_test, y_pred)
        auc_score_te = roc_auc_score(y_test, y_pred)
        auc_score_tr = gs.best_score_
        accuracy = (y_pred == y_test).mean()
        params = gs.best_params_
        [[TP, FN], [FP, TN]] = confusion_matrix(y_test, y_pred)
        results.loc[name, :] = (name, params, accuracy, auc_score_tr, auc_score_te, precision,
                                recall, f_score, support, TP, FP, FN, TN)

        scores.append(roc_auc_score(y_test, y_pred))
        gs_results = pd.DataFrame(gs.cv_results_).drop("params", axis=1).sort_values("rank_test_score")
        print("\n{}:\n".format(name))
        print("\tAccuracy: {:.2%}".format(accuracy))
        print("\tAUC Score (Train set): {:.2%}".format(gs.best_score_))
        print("\tAUC Score (Test set): {:.2%}\n".format(scores[-1]))
        print(classification_report(y_test, y_pred))
        print(best_models[-1], "\n")
        if i + 1 < len(classifiers):
            print("#" * 100)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    results.plot.bar(ax=ax)
    if save:
        pickle_models(best_models)
    return best_models, results

def make_grid_search_reg(regressors, reg_params, X_train, y_train, X_test, y_test, random=False, search_kw=None, save=True):
    best_models, scores = [], []
    results = pd.DataFrame(index=[item[0] for item in regressors],
                           columns=["name", "params", "r2_score_tr", "r2_score_te", "explained_variance", "rmse", 
                                    "mean_absolute_error", "mean_squared_log_error", "median_absolute_error"])

    if random:
        SearchCV = RandomizedSearchCV
    else:
        SearchCV = GridSearchCV
    if search_kw is None:
        search_kw = dict(n_jobs=-1, return_train_score=True)

    for i, (name, reg) in enumerate(regressors):
        params = reg_params[name]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            gs = SearchCV(reg, params, **search_kw).fit(X_train, y_train)
        best_models.append(gs.best_estimator_)
        y_pred = gs.predict(X_test)
        r2_score_te = r2_score(y_test, y_pred)
        r2_score_tr = gs.best_score_
        metrics = []
        rmse = lambda x, y: np.sqrt(mean_squared_error(x, y))
        for func in (explained_variance_score, rmse, mean_absolute_error, mean_squared_log_error, median_absolute_error):
            metrics.append(func(y_test, y_pred))
        params = gs.best_params_
        results.loc[name, :] = (name, params, r2_score_tr, r2_score_te, *metrics)

        scores.append(r2_score_te)
        gs_results = pd.DataFrame(gs.cv_results_).drop("params", axis=1).sort_values("rank_test_score")
        print("\n{}:\n".format(name))
        print("\tRMSE: {:.2f}".format(metrics[1]))
        print("\tR2 Score (Train set): {:.2%}".format(gs.best_score_))
        print("\tR2 Score (Test set): {:.2%}\n".format(scores[-1]))
        print(best_models[-1], "\n")
        if i + 1 < len(regressors):
            print("#" * 100)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    results.plot.bar(ax=ax)
    if save:
        pickle_models(best_models)
    return best_models, results
    

def plot_roc_curve(classifiers, models, X, y):
    for name, model in zip(map(lambda x: x[0], classifiers), models):
        scores = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y.ravel(), scores.ravel())
        roc_auc = roc_auc_score(y.ravel(), scores.ravel())
        plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.2%})')

    plt.title('Receiver Operating Characteristic')
    plt.plot([0, 1], [0, 1], 'r--', label="Random predictions")
    plt.legend(loc=4)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
