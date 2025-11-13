# preprocessing and engineered features

from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def classify_instances(data):
    data['class'] = data['pm2.5'].apply(lambda x: 0 if x <= 150 else 1)
    return data

def encode_wind_direction(data):
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_cbwd = enc.fit_transform(data[['cbwd']])
    encoded_colums = enc.get_feature_names_out()

    encoded_data = pd.DataFrame(encoded_cbwd, columns=encoded_colums)
    data = pd.concat([data, encoded_data], axis=1).drop(['cbwd'], axis=1)

    return data