import warnings
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

from visualisation import label_barchart


def make_cross_validation(models, scoring, X, y):
    results = OrderedDict()
    for name, model in models:
        kfold = KFold(n_splits=3)
        cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
        results[name] = cv_results

    results = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    plt.bar(range(results.shape[1]), results.mean(), yerr=results.std())
    plt.gca().set_xticklabels([""] + list(results.columns) + [""])
    plt.title("Comparison of AUC score for multiple classifiers", y=-.2)
    plt.subplots_adjust(bottom=.2)
    label_barchart(plt.gca())
    return results


def make_grid_search_clf(classifiers, clf_params, X_train, y_train, X_test, y_test, random=False, search_kw=None):
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

        return best_models, gs_results


def plot_roc_curve(classifiers, models, X, y):
    for name, model in zip(map(lambda x: x[0], classifiers), models):
        scores = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y.ravel(), scores.ravel())
        roc_auc = roc_auc_score(y.ravel(), scores.ravel())
        plt.plot(fpr, tpr, 'b', label=f'{name} (AUC={roc_auc:.2%})')

    plt.title('Receiver Operating Characteristic')
    plt.plot([0, 1], [0, 1], 'r--', label="Random predictions")
    plt.legend(loc=4)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
