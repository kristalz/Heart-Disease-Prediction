import pandas as pd 
import numpy as np
from sklearn import preprocessing 

def remove_outliers(df):
    numeric_col=[col for col in df if df[col].dtype !="object" ]
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    IQR = q3-q1
    lower = q1 - 1.5 * IQR
    upper = q3 + 1.5 * IQR
    df = df[~((df[numeric_col] < (lower)) |(df[numeric_col] > (upper))).any(axis=1)]
    return df

def label_encoding(df,col_name):
    label_encoder = preprocessing.LabelEncoder()
    df[col_name]= label_encoder.fit_transform(df[col_name])
    df[col_name].unique()
