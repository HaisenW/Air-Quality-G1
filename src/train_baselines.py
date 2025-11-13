# classical ML training for both tasks

import numpy as np
# from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer, PowerTransformer

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, accuracy_score, ConfusionMatrixDisplay, mean_squared_error
, roc_auc_score, root_mean_squared_error)
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import seaborn as sns
from features import *
from data import *
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42


def linear_regression(X_train, X_test, y_train, y_test):
    linreg = Pipeline([('scaler', StandardScaler()), ('rgs', LinearRegression())])
    linreg.fit(X_train, y_train)

    # y_pred = linreg.predict(X_test)

    # # Residuals vs predicted
    # residuals = y_test - y_pred
    #
    # plt.figure()
    # plt.scatter(y_pred, residuals, color='blue', alpha=0.3)
    # plt.axhline(y=0, color='red', linestyle='--')
    # plt.xlabel("Predicted Values")
    # plt.ylabel("Residuals")
    # plt.title("Residuals vs Predicted Values of Linear Regression Model")
    # plt.show()

    return linreg

def naive_bayes(X_train, X_test, y_train, y_test):
    gnb = Pipeline([('scaler', StandardScaler()), ('normalizer', Normalizer()), ('transformer', PowerTransformer(method='yeo-johnson')), ('clf', GaussianNB())])
    gnb.fit(X_train, y_train)

    # Accuracy on validation
    validation_accuracy = cross_val_score(gnb, X_train, y_train, cv=10, scoring="accuracy")
    print("Naive Bayes validation accuracy: ", validation_accuracy.mean())

    # ROC-AUC on validation
    validation_roc_auc = cross_val_score(gnb, X_train, y_train, cv=10, scoring="roc_auc")
    print("Naive Bayes validation AUC: ", validation_roc_auc.mean())

    # Accuracy on test
    y_pred = gnb.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print("Naive Bayes test accuracy: ", test_accuracy)

    # ROC-AUC on test
    test_roc_auc = roc_auc_score(y_test, y_pred)
    print("Naive Bayes test AUC: ", test_roc_auc)

    return gnb

def decision_tree_regression(X_train, X_test, y_train, y_test):
    dt_reg = Pipeline([('scaler', StandardScaler()), ('rgs', DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE))])
    dt_reg.fit(X_train, y_train)

    return dt_reg

def decision_tree_classification(X_train, X_test, y_train, y_test):
    dtc = Pipeline([('scaler', StandardScaler()), ('resampler', SMOTE(random_state=RANDOM_STATE)), ('clf', DecisionTreeClassifier(criterion='gini', max_depth=10, random_state=RANDOM_STATE))])
    dtc.fit(X_train, y_train)

    # Accuracy on validation
    validation_accuracy = cross_val_score(dtc, X_train, y_train, cv=10, scoring="accuracy")
    print("Decision tree classification validation accuracy: ", validation_accuracy.mean())

    # ROC-AUC on validation
    validation_roc_auc = cross_val_score(dtc, X_train, y_train, cv=10, scoring="roc_auc")
    print("Decision tree classification validation AUC: ", validation_roc_auc.mean())

    # Accuracy on test
    y_pred = dtc.predict(X_test)
    print(y_pred[:10])
    test_accuracy = accuracy_score(y_test, y_pred)
    print("Decision tree classification test accuracy: ", test_accuracy)

    # ROC-AUC on test
    test_roc_auc = roc_auc_score(y_test, y_pred)
    print("Decision tree classification validation AUC: ", test_roc_auc)

    # Confusion Matrix part
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Good", "Unhealthy"])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix of Decision Tree Classifier")
    plt.show()

    return dtc

if __name__ == '__main__':
    X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test = split_data()

    linear_regression(X_train, X_test, y_reg_train, y_reg_test)
    # decision_tree_regression(X_train, X_test, y_train, y_test)
    # naive_bayes(X_train, X_test, y_class_train, y_class_test)
    # decision_tree_classification(X_train, X_test, y_class_train, y_class_test)