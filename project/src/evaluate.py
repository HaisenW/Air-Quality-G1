# metrics, plots, confusion/residuals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from train_baselines import *
from train_nn import *
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

    # Cross Validation scores
    validation_accuracy = cross_val_score(model, X_train, y_train, cv=10, scoring="accuracy")
    validation_roc_auc = cross_val_score(model, X_train, y_train, cv=10, scoring="roc_auc")
    print("CV accuracy: " + f"{validation_accuracy.mean():.4f}")
    print("CV roc_auc: " + f"{validation_roc_auc.mean():.4f}")

    for k, v in m.items():
        print(f"{k}: {v:.4f}")

    plot_confusion(y_test, y_pred, labels=target_names)
    if y_score is not None:
        plot_roc(y_test, y_score)
        plot_pr(y_test, y_score)
    return m


# regression metrics
def compute_reg_metrics(model, X_train, X_test, y_train, y_test):
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

def plot_reg(model, X_test, y_test, name):
    y_pred = model.predict(X_test)

    # Residuals vs predicted
    residuals = y_test - y_pred

    plt.figure()
    plt.scatter(y_pred, residuals, color='blue', alpha=0.3)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted Values of " + name + " Model")
    plt.show()

if __name__ == '__main__':
    X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test = split_data()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    #
    # linear_reg_model = linear_regression(X_train, y_reg_train)
    # print(compute_reg_metrics(linear_reg_model, X_train, X_test, y_reg_train, y_reg_test))
    # plot_reg(linear_reg_model, X_test, y_reg_test, "Linear Regression")
    #
    # decision_tree_model = decision_tree_regression(X_train, y_reg_train)
    # print(compute_reg_metrics(decision_tree_model, X_train, X_test, y_reg_train, y_reg_test))
    # plot_reg(decision_tree_model, X_test, y_reg_test, "Decision Tree Regression")
    #
    class_target = ['Good', 'Unhealthy']
    #
    # gnb = naive_bayes(X_train, y_class_train)
    # evaluate_model("Naive Bayes", gnb, X_train, y_class_train, X_test, y_class_test, class_target)
    #
    # dtc= decision_tree_classification(X_train, y_class_train)
    # evaluate_model("Decision Tree Classification", dtc, X_train, y_class_train, X_test, y_class_test, class_target)

    # mlp_reg_model = nn_reg_model(X_train, y_reg_train)
    # print(compute_reg_metrics(mlp_reg_model, X_train, X_test, y_reg_train, y_reg_test))
    # plot_reg(mlp_reg_model, X_test, y_reg_test, "NN Regression")

    mlp_class_model = nn_class_model(X_train, y_class_train)
    # evaluate_model("NN Classification", mlp_class_model, X_train, y_class_train, X_test, y_class_test, class_target)