from flask import Flask, render_template, request, jsonify
import json
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, plot, iplot
import plotly.figure_factory as ff
from sklearn import preprocessing 
from data_preprocess import remove_outliers, label_encoding
from models import knn_model, RFC
import pickle

app = Flask(__name__)

heart_data = pd.read_csv("BIS_634_HW/final project/heart_disease.csv")

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/background")
def background():
    return render_template('background.html')

@app.route("/data")
def data_description():
    heart_data['HeartDisease'].replace({1:'Yes', 0:'No'}, inplace=True) 
    heart_data['ExerciseAngina'].replace({'Y':'Yes','N':'No'}, inplace=True)
    heart_data['FastingBS'].replace({1: '> 120 mg/dl', 0: 'Normal'}, inplace=True)
    fig = go.Figure(data=[go.Table(
                                 header=dict(
                                              values=list(heart_data.columns), # header values
                                              line_color='black', # line Color of header 
                                              fill_color='pink', # background color of header
                                              align='center', # align header at center
                                              height=40, # height of Header
                                              font=dict(color='white', size=10), # font size & color of header text
                                             ), cells=dict(values=[
                                                     heart_data.Age , # cclumn values
                                                     heart_data.Sex, 
                                                     heart_data.ChestPainType,
                                                     heart_data.RestingBP, 
                                                     heart_data.Cholesterol,
                                                     heart_data.FastingBS,
                                                     heart_data.RestingECG,
                                                     heart_data.MaxHR,
                                                     heart_data.ExerciseAngina,
                                                     heart_data.Oldpeak,
                                                     heart_data.ST_Slope,
                                                     heart_data.HeartDisease
                                                    ],line_color='black', # line color of the cell
                                            fill_color='lightblue', # color of the cell
                                            align='left'  # align text to left in cell
                                           )
                                    )
                        ]
                    )
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('data.html', graphJSON=graphJSON)

@app.route("/summary_statistics")
def stat():
    return render_template('summary_statistics.html')

@app.route("/univariate_analysis")
def univariate():
    colors = px.colors.cyclical.Twilight
    fig = make_subplots(rows=1,cols=2,
                        subplot_titles=('Counts',
                                        'Percentages'),
                        specs=[[{"type": "xy"},
                                {'type':'domain'}]])
    fig.add_trace(go.Bar(y = heart_data['Sex'].value_counts().values.tolist(), 
                        x = heart_data['Sex'].value_counts().index, 
                        text=heart_data['Sex'].value_counts().values.tolist(),
                textfont=dict(size=15),
                        textposition = 'outside',
                        showlegend=False,
                marker = dict(color = colors,
                                line_color = 'black',
                                line_width=3)),row = 1,col = 1)
    fig.add_trace((go.Pie(labels=heart_data['Sex'].value_counts().keys(),
                                values=heart_data['Sex'].value_counts().values,textfont = dict(size = 16),
                        hole = .4,
                        marker=dict(colors=colors),
                        textinfo='label+percent',
                        hoverinfo='label')), row = 1, col = 2)
    fig.update_yaxes(range=[0,1000])
    fig.update_layout(
                        title=dict(text = f"Gender Distribution",x=0.5,y=0.95),
                        title_font_size=30
                        )
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)  
    return render_template('univariate_analysis.html',graphJSON=graphJSON)

@app.route("/bivariate_analysis")
def plot_biv():
    new_heart_data = remove_outliers(heart_data)
    categorical_col=[col for col in new_heart_data if new_heart_data[col].dtype=="object"] 
    numeric_col=[col for col in new_heart_data if new_heart_data[col].dtype !="object" ]
    fig1 = px.scatter(new_heart_data, 
                    x=new_heart_data['Age'], 
                    y=new_heart_data['RestingBP'], 
                    color=new_heart_data['HeartDisease'], 
                    facet_col=new_heart_data['FastingBS'],
                    facet_row=new_heart_data['Sex'],
                    color_discrete_map={1: "#FF5722",0: "#7CB342"},
                    width=1000, 
                    height=800)

    fig1.update_layout(
                        plot_bgcolor= "#dcedc1",
                        paper_bgcolor="#FFFDE7",
                    )   
    
    fig2 = px.scatter(new_heart_data, 
                    x=new_heart_data['Age'], 
                    y=new_heart_data['Cholesterol'], 
                    color=new_heart_data['HeartDisease'], 
                    facet_col=new_heart_data['FastingBS'],
                    facet_row=new_heart_data['Sex'],
                    color_discrete_map={1: "#FF5722",0: "#7CB342"},
                    width=1000, 
                    height=800)

    fig2.update_layout(
                        plot_bgcolor= "#dcedc1",
                        paper_bgcolor="#FFFDE7",
                    )   
    graphJSON1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('bivariate_analysis.html',graphJSON1=graphJSON1, graphJSON2=graphJSON2)


@app.route("/feature_corr")
def plot_corr(): 
    new_heart_data = remove_outliers(heart_data)
    categorical_col=[col for col in new_heart_data if new_heart_data[col].dtype=="object"] 
    for col in categorical_col:
        label_encoding(new_heart_data,col)
    new_heart_data_corr = new_heart_data.corr()
    fig = px.imshow(new_heart_data_corr)
    fig.update_layout(
    title='12 attributes correlation',
    width=800,
    height=600,
    )
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('feature_corr.html', graphJSON=graphJSON)

@app.route("/model_prediction")
def model_select():
    return render_template('model_prediction.html')

@app.route('/analyze_knn', methods=["POST"])
def knn():
    usertext = request.form["usertext"]
    k = int(usertext)
    new_heart_data = remove_outliers(heart_data)
    accuracy, sensitivity, specificity, precision, cm_df = knn_model(new_heart_data, k)
    fig = px.imshow(cm_df,
                    labels=dict(x="Predicted label ", y="True lable"),
                    x=['No Heart Disease', 'Heart Disease'],
                    y=['No Heart Disease', 'Heart Disease']
                )
    fig.update_layout(
        title=f'The confusion matrix for k={k} is:',
        width=800,
        height=600,
        )
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('analyze_knn.html', accuracy=accuracy, sensitivity=sensitivity, specificity=specificity, precision=precision, graphJSON=graphJSON)

@app.route('/analyze_rfc', methods=["POST"])
def rfc():
    usertext = request.form["usertext"]
    n = int(usertext)
    new_heart_data = remove_outliers(heart_data)
    accuracy, sensitivity, specificity, precision, cm_df = RFC(new_heart_data, n)
    fig = px.imshow(cm_df,
                    labels=dict(x="Predicted label ", y="True lable"),
                    x=['No Heart Disease', 'Heart Disease'],
                    y=['No Heart Disease', 'Heart Disease']
                )
    fig.update_layout(
        title=f'The confusion matrix for n_estimators={n} is:',
        width=800,
        height=600,
        )
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('analyze_rfc.html', accuracy=accuracy, sensitivity=sensitivity, specificity=specificity, precision=precision, graphJSON=graphJSON)


@app.route('/predictknn')
def predictknn():
    return render_template('predictknn.html')

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 11)
    loaded_model = pickle.load(open("modelknn.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/resultknn', methods=['POST'])
def result_knn():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)       
        if int(result)== 1:
            prediction ='Sorry, you have heart Disease'
        else:
            prediction ='You are healthy!'           
        return render_template("resultknn.html", prediction = prediction)

@app.route('/predictrfc')
def predictrfc():
    return render_template('predictrfc.html')

def ValuePredictor_rfc(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 11)
    loaded_model = pickle.load(open("modelforest.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/resultrfc', methods=['POST'])
def result_rfc():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor_rfc(to_predict_list)       
        if int(result)== 1:
            prediction ='Sorry, you have heart Disease'
        else:
            prediction ='You are healthy!'           
        return render_template("resultrfc.html", prediction = prediction)

if __name__ == "__main__":
    app.run(debug=True)