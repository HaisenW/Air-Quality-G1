# classical ML training for both tasks

import numpy as np
# from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer, PowerTransformer, PolynomialFeatures

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
from sklearn.model_selection import GridSearchCV

RANDOM_STATE = 42

def gridsearch_linear_regression(X_train, y_train):
    linreg = Pipeline([('scaler', StandardScaler()),
                       ('poly', PolynomialFeatures(include_bias=False)),
                       ('rgs', LinearRegression())])
    param_grid = {
        "poly__degree": [1, 2, 3, 4]
    }
    grid = GridSearchCV(
        estimator=linreg,
        param_grid=param_grid,
        cv=10,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    print("Best CV score:", -grid.best_score_)

def linear_regression(X_train, y_train):
    linreg = Pipeline([('scaler', StandardScaler()),
                       ('poly', PolynomialFeatures(include_bias=False, degree=3)),
                       ('rgs', LinearRegression())])
    linreg.fit(X_train, y_train)

    return linreg

def gridsearch_naive_bayes(X_train, y_train):
    gnb = Pipeline([('scaler', StandardScaler()),
                    ('normalizer', Normalizer()),
                    ('transformer', PowerTransformer(method='yeo-johnson')),
                    ('clf', GaussianNB())])
    param_grid = {
        "clf__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    }
    grid = GridSearchCV(
        estimator=gnb,
        param_grid=param_grid,
        cv=10,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    print("Best CV score:", grid.best_score_)

def naive_bayes(X_train, y_train):
    gnb = Pipeline([('scaler', StandardScaler()),
                    ('normalizer', Normalizer()),
                    ('transformer', PowerTransformer(method='yeo-johnson')),
                    ('clf', GaussianNB(var_smoothing=0.001))])
    gnb.fit(X_train, y_train)

    return gnb

def gridsearch_decision_tree_regression(X_train, y_train):
    dt_reg = Pipeline([('scaler', StandardScaler()),
                       ('rgs', DecisionTreeRegressor(random_state=RANDOM_STATE))])
    param_grid = {
                "rgs__max_depth": [None, 3, 5, 7, 10, 15],
                "rgs__min_samples_split": [2, 5, 10, 20],
                "rgs__min_samples_leaf": [1, 2, 4, 8],
                "rgs__max_features": [None, "sqrt", "log2"],
                "rgs__max_leaf_nodes": [None, 10, 20, 50, 100]
    }
    grid = GridSearchCV(
        estimator=dt_reg,
        param_grid=param_grid,
        cv=10,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    print("Best CV score:", -grid.best_score_)

def decision_tree_regression(X_train, y_train):
    dt_reg = Pipeline([('scaler', StandardScaler()),
                       ('rgs', DecisionTreeRegressor(max_depth=10,
                                                     max_features=None,
                                                     max_leaf_nodes=None,
                                                     min_samples_leaf=4,
                                                     min_samples_split=20,
                                                     random_state=RANDOM_STATE))])
    dt_reg.fit(X_train, y_train)

    return dt_reg

def gridsearch_decision_tree_classification(X_train, y_train):
    dtc = Pipeline([('scaler', StandardScaler()),
                    ('resampler', SMOTE(random_state=RANDOM_STATE)),
                    ('clf', DecisionTreeClassifier(random_state=RANDOM_STATE))])
    param_grid = {
                "clf__criterion": ["gini", "entropy"],
                "clf__max_depth": [None, 3, 5, 7, 10, 15],
                "clf__min_samples_split": [2, 5, 10, 20],
                "clf__min_samples_leaf": [1, 2, 4, 8],
                "clf__max_features": [None, "sqrt", "log2"],
                "clf__max_leaf_nodes": [None, 10, 20, 50, 100]
    }
    grid = GridSearchCV(
        estimator=dtc,
        param_grid=param_grid,
        cv=10,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    print("Best CV score:", grid.best_score_)

def decision_tree_classification(X_train, y_train):
    dtc = Pipeline([('scaler', StandardScaler()),
                    ('resampler', SMOTE(random_state=RANDOM_STATE)),
                    ('clf', DecisionTreeClassifier(criterion='gini',
                                                   max_depth=None,
                                                   max_features=None,
                                                   max_leaf_nodes=100,
                                                   min_samples_leaf=1,
                                                   min_samples_split=2,
                                                   random_state=RANDOM_STATE))])
    dtc.fit(X_train, y_train)

    return dtc

if __name__ == '__main__':
    X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test = split_data()

    # Gridsearch for best parameters
    # Best params: {'poly__degree': 3}
    # gridsearch_linear_regression(X_train, y_reg_train)

    # Best params: {'clf__var_smoothing': 0.001}
    # gridsearch_naive_bayes(X_train, y_class_train)

    # Best params: {'rgs__max_depth': 10, 'rgs__max_features': None, 'rgs__max_leaf_nodes': None, 'rgs__min_samples_leaf': 4, 'rgs__min_samples_split': 20}
    # gridsearch_decision_tree_regression(X_train, y_reg_train)

    # Best params: {'clf__criterion': 'gini', 'clf__max_depth': None, 'clf__max_features': None, 'clf__max_leaf_nodes': 100, 'clf__min_samples_leaf': 1, 'clf__min_samples_split': 2}
    # gridsearch_decision_tree_classification(X_train, y_class_train)

    # linear_regression(X_train, y_reg_train)
    # decision_tree_regression(X_train, y_reg_train)
    # naive_bayes(X_train, y_class_train)
    # decision_tree_classification(X_train, y_class_train)