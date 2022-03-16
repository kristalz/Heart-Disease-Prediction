import pandas as pd
from sklearn import datasets, linear_model, metrics
from sklearn import preprocessing 
from sklearn import decomposition
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from data_preprocess import remove_outliers, label_encoding
import pickle
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_rows",None)

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
    acc = accuracy_score(y_test, y_pred)
    print(f'Successfully trained model with an accuracy of {acc:.2f}')

    return forest

if __name__ == '__main__':
    heart_data = pd.read_csv("BIS_634_HW/final project/heart_disease.csv")
    new_heart_data = remove_outliers(heart_data)
    forest_mdl = RFC(new_heart_data,19)
    with open('modelforest.pkl', 'wb') as file:
        pickle.dump(forest_mdl, file)  