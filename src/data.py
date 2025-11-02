import csv

import pandas as pd
from pandas import read_csv
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    air_quality_filename = '../data/PRSA_data_2010.1.1-2014.12.31.csv'
    data = read_csv(air_quality_filename, index_col=0)
    data = data[data['pm2.5'].notna()]

    data['class'] = data['pm2.5'].apply(lambda x: 0 if x <= 150 else 1)

    return data

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
    load_data()
    # distri_class(load_data())
    correlation_matrix(load_data())