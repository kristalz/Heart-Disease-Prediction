import pandas as pd
from sklearn import datasets, linear_model, metrics
from sklearn import preprocessing 
from sklearn import decomposition
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_rows",None)


def knn_model(df,k):
    def label_endocing(col_name):
        label_encoder = preprocessing.LabelEncoder()
        df[col_name]= label_encoder.fit_transform(df[col_name])
        df[col_name].unique()

    categorical_col=[col for col in df if df[col].dtype=="object"]
    for col in categorical_col:
        label_endocing(col)
    scaler = StandardScaler()
    scaler.fit(df.drop('HeartDisease',axis = 1))
    scaled_features = scaler.transform(df.drop('HeartDisease',axis = 1))
    feature_df = pd.DataFrame(scaled_features,columns = df.columns[:-1])
    X = feature_df
    y = df['HeartDisease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    cmat = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cmat, columns = ['No Heart Disease', 'Heart Disease'], index=['No Heart Disease', 'Heart Disease'])
    TN =cm_df.iloc[0, 0]
    FP = cm_df.iloc[0, 1]
    FN = cm_df.iloc[1, 0]
    TP = cm_df.iloc[1, 1]
    accuracy = round((TP + TN) / (TP + FP + TN + FN) * 100, 2)
    sensitivity = round(TP / (TP + FN) * 100, 2)
    specificity = round(TN / (TN + FP) * 100, 2)
    precision = round(TP / (TP + FP) * 100, 2)
    c = round(TP / (TP + FP) * 100, 2)
    return accuracy, sensitivity, specificity, precision, cm_df


def RFC(df,n):
    def label_endocing(col_name):
        label_encoder = preprocessing.LabelEncoder()
        df[col_name]= label_encoder.fit_transform(df[col_name])
        df[col_name].unique()

    categorical_col=[col for col in df if df[col].dtype=="object"]
    for col in categorical_col:
        label_endocing(col)
    scaler = StandardScaler()
    scaler.fit(df.drop('HeartDisease',axis = 1))
    scaled_features = scaler.transform(df.drop('HeartDisease',axis = 1))
    feature_df = pd.DataFrame(scaled_features,columns = df.columns[:-1])
    X = feature_df
    y = df['HeartDisease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    forest= RandomForestClassifier(n_estimators = n, random_state = 0)
    forest.fit(X_train,y_train)  
    y_pred = forest.predict(X_test)
    cmat = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cmat, columns = ['No Heart Disease', 'Heart Disease'], index=['No Heart Disease', 'Heart Disease'])
    TN =cm_df.iloc[0, 0]
    FP = cm_df.iloc[0, 1]
    FN = cm_df.iloc[1, 0]
    TP = cm_df.iloc[1, 1]
    accuracy = round((TP + TN) / (TP + FP + TN + FN) * 100, 2)
    sensitivity = round(TP / (TP + FN) * 100, 2)
    specificity = round(TN / (TN + FP) * 100, 2)
    precision = round(TP / (TP + FP) * 100, 2)
    return accuracy, sensitivity, specificity, precision, cm_df
