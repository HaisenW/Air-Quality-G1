# metrics, plots, confusion/residuals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from train_baselines import *
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay, mean_absolute_error,
    root_mean_squared_error
)
from sklearn.model_selection import cross_val_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# classification metrics
def compute_metrics(y_true, y_pred, y_score=None):
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None:
        try:
            m["roc_auc"] = roc_auc_score(y_true, y_score)
        except Exception:
            m["roc_auc"] = np.nan
    else:
        m["roc_auc"] = np.nan
    return m


def plot_confusion(y_true, y_pred, labels=("class 0", "class 1")):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()


def plot_roc(y_true, y_score):
    fig = plt.figure()
    RocCurveDisplay.from_predictions(y_true, y_score)
    plt.title("ROC curve")
    plt.show()


def plot_pr(y_true, y_score):
    fig = plt.figure()
    PrecisionRecallDisplay.from_predictions(y_true, y_score)
    plt.title("Precision Recall curve")
    plt.show()


def evaluate_model(name, model, X_train, y_train, X_test, y_test, target_names):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = None

    m = compute_metrics(y_test, y_pred, y_score)
    print(f"{name} metrics")
    for k, v in m.items():
        print(f"{k}: {v:.4f}")

    plot_confusion(y_test, y_pred, labels=target_names)
    if y_score is not None:
        plot_roc(y_test, y_score)
        plot_pr(y_test, y_score)
    return m


# regression metrics
def compute_reg_metrics(model, X_train, y_train, y_test):
    y_pred = model.predict(X_test)

    # MAE on validation
    validation_mae = abs(cross_val_score(model, X_train, y_train, cv=10, scoring="neg_mean_absolute_error").mean())
    # RMSE on validation
    validation_rmse = abs(cross_val_score(model, X_train, y_train, cv=10, scoring="neg_root_mean_squared_error").mean())
    # MAE on test
    mae = mean_absolute_error(y_test, y_pred)
    # RMSE on test
    rmse = root_mean_squared_error(y_test, y_pred)

    m = {
        "validation_mae": validation_mae,
        "validation_rmse": validation_rmse,
        "test_mae": mae,
        "test_rmse": rmse
    }

    return m

if __name__ == '__main__':
    X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test = split_data()

    linear_reg_model = linear_regression(X_train, X_test, y_reg_train, y_reg_test)
    print(compute_reg_metrics(linear_reg_model, X_train, y_reg_train, y_reg_test))

    decision_tree_model = decision_tree_regression(X_train, X_test, y_reg_train, y_reg_test)
    print(compute_reg_metrics(decision_tree_model, X_train, y_reg_train, y_reg_test))