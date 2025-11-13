# loading, cleaning, splitting

import csv
import pandas as pd
from pandas import read_csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from features import *

RANDOM_STATE = 42

def load_data():
    air_quality_filename = '../data/PRSA_data_2010.1.1-2014.12.31.csv'
    data = read_csv(air_quality_filename, index_col=0)
    data = data[data['pm2.5'].notna()]
    data = data.reset_index()

    # tagging instances to good and unhealthy
    data = classify_instances(data)

    # encode cbwd to number
    data = encode_wind_direction(data)

    return data

def split_data():
    data = load_data()
    feature_cols = ["hour", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir", "cbwd_NE", "cbwd_NW", "cbwd_SE", "cbwd_cv"]

    X = data[feature_cols]
    y_reg = data["pm2.5"]
    y_class = data["class"]

    X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.3,random_state=RANDOM_STATE)
    y_class_train = y_class.loc[y_reg_train.index]
    y_class_test = y_class.loc[y_reg_test.index]

    return (X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test)

def distri_class(data):
    data: pd.DataFrame
    data['class'].value_counts().sort_index().plot(
        kind='bar',
        color='lightblue',
        edgecolor='black',
    )
    plt.title('Distribution of Air Quality Classification')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()

def correlation_matrix(data):
    data: pd.DataFrame
    feature_cols = ["hour", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
    features = data[feature_cols]

    corr_matrix = features.corr()
    plt.figure()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Heatmap for Key Numeric Features')
    plt.show()


if __name__ == '__main__':
    data = load_data()
    # distri_class(load_data())
    # correlation_matrix(load_data())