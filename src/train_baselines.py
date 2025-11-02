import numpy as np
from sklearn.preprocessing import StandardScaler

import data
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


def linear_regression():
    csv_data = data.load_data()
    feature_cols = ["hour", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]

    X = csv_data[feature_cols]
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    y = csv_data["pm2.5"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    linreg = LinearRegression()
    linreg.fit(X_train, y_train)

    y_pred = linreg.predict(X_test)

    # MAE on validation
    validation_mae = cross_val_score(linreg, X_train, y_train, cv=10, scoring="neg_mean_absolute_error")
    print("Linear regression validation MAE: ", validation_mae.mean())

    # RMSE on validation
    validation_rmse = cross_val_score(linreg, X_train, y_train, cv=10, scoring="neg_root_mean_squared_error")
    print("Linear regression validation RMSE: ", validation_rmse.mean())

    # MAE on test
    mae = metrics.mean_absolute_error(y_test, y_pred)
    print("Linear regression test MAE: ", mae)

    # RMSE on test
    rmse = root_mean_squared_error(y_test, y_pred)
    print("Linear regression test RMSE: ", rmse)

    # Residuals vs predicted
    residuals = y_test - y_pred

    plt.figure()
    plt.scatter(y_pred, residuals, color='blue', alpha=0.3)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted Values of Linear Regression Model")
    plt.show()

def naive_bayes():
    csv_data = data.load_data()
    feature_cols = ["hour", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
    X = csv_data[feature_cols]
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    y = csv_data["class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    gnb = GaussianNB()
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

def decision_tree_regression():
    csv_data = data.load_data()
    feature_cols = ["hour", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
    X = csv_data[feature_cols]
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    y = csv_data["pm2.5"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    dt_reg = DecisionTreeRegressor()
    dt_reg.fit(X_train, y_train)

    # MAE on validation
    validation_accuracy = cross_val_score(dt_reg, X_train, y_train, cv=10, scoring="neg_mean_absolute_error")
    print("Decision tree regression validation accuracy: ", validation_accuracy.mean())

    # RMSE on validation
    validation_roc_auc = cross_val_score(dt_reg, X_train, y_train, cv=10, scoring="neg_root_mean_squared_error")
    print("Decision tree validation RMSE: ", validation_roc_auc.mean())

    y_pred = dt_reg.predict(X_test)
    print(y_pred[:10])

    # MAE on test
    mae = metrics.mean_absolute_error(y_test, y_pred)
    print(mae)
    print("Decision tree test MAE: ", mae)

    # RMSE on test
    rmse = root_mean_squared_error(y_test, y_pred)
    print(rmse)
    print("Decision tree test RMSE: ", rmse)

def decision_tree_classification():
    csv_data = data.load_data()
    feature_cols = ["hour", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
    X = csv_data[feature_cols]
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    y = csv_data["class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    dtc = DecisionTreeClassifier(max_depth=5, random_state=1)
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

if __name__ == '__main__':
    linear_regression()
    decision_tree_regression()
    naive_bayes()
    decision_tree_classification()