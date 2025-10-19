import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def clean_data(df):
    df = df.dropna()
    df = df[df['yield'] > 0]
    return df

def normalize(df):
    scaler = MinMaxScaler()
    numeric_cols = df.select_dtypes(include=['float64','int64']).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df
