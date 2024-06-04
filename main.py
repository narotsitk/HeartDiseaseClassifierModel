from flask import Flask
from flask import render_template
from flask import request,redirect,session

import numpy as np
import json
import joblib
import pandas as pd
import os
import pickle

from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,MinMaxScaler,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier,VotingClassifier
from xgboost import XGBClassifier



age_categories=[['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older']]
gen_health_categories = [['Poor','Fair','Good','Very good','Excellent']]

meta_data_file = "../metadata.json"

meta_data = {}
with open(meta_data_file,"r") as jsonFile:
    meta_data = json.load(jsonFile)

class HeartDiseasePredictionPipeline:

    THRESHOLD = 0.7
    COLUMN_ORDER = ['Asthma', 'PhysicalHealth', 'Smoking', 
                    'AlcoholDrinking', 'DiffWalking', 'KidneyDisease', 
                    'Stroke', 'Diabetic', 'SleepTime', 'PhysicalActivity', 
                    'MentalHealth', 'SkinCancer', 'BMI', 'Race', 'Sex',
                      'AgeCategory', 'GenHealth'] 
    def __init__(self,model_path:os.path,preprocessor_path:os.path,*args,**kwargs):

        self.model = joblib.load(model_path)


        with open(preprocessor_path,"rb") as processorFile:
            self.preprocessor = pickle.load(processorFile)
            

    def preprocess(self,data:pd.DataFrame,*args,**kwargs) -> pd.DataFrame:
        data[meta_data['numerical_columns']] = data[meta_data['numerical_columns']].astype('float64')
        self.preprocessor.set_output(transform='pandas')

        # pass data through a column transformer
        data_encoded = self.preprocessor.transform(data)
        data_encoded.columns = list(map(lambda x: x.replace("scaled__","").replace("encoded_0__","").replace("encoded_1__","").replace("encoded_2__","").replace("remainder__",""),\
                                data_encoded.columns))

        # the data must be reindexed how it was sent during training 
        data_encoded = data_encoded.drop(columns = "HeartDisease")
        data_encoded = data_encoded.reindex(columns =  self.model.feature_names_in_)
        return data_encoded
    
    def predict(self,data_encoded,*args,**kwargs) -> tuple:
        predict_proba = self.model.predict_proba(data_encoded)
        
        # prediction is true if the probability > THRESHOLD
        if predict_proba[0][1] >= self.THRESHOLD:
            predict_proba = predict_proba[0][1]
            prediction = True 
        else:
            predict_proba = predict_proba[0][0]
            prediction = False

        return prediction, round(predict_proba*100,2)
    
    def column_name_transformer(self,column,*args,**kwargs):
        if column in meta_data["numerical_columns"]:
            return "scaled__"+column
        elif column == "AgeCategory":
            return "encoded_1__"+column
        elif column == "GenHealth":
            return "encoded_2__"+column
        elif column in (set(meta_data["categorical_columns"]) - {"AgeCategory","GenHealth"}):
            return "encoded_0__"+column
        else:
            return "remainder__"+column






MODEL_VERSION = "09e44cef72" #"430593fd95"
model_path_folder = "../model/"
model_path = os.path.join(model_path_folder,f"heart_disease_classifier_{MODEL_VERSION}.pkl")
                          
preprocessor_path_folder = "../model/"
preprocessor_path = os.path.join(preprocessor_path_folder,"preprocessor_v2.pkl")

app = Flask(__name__)
import os

app.secret_key = os.urandom(32).hex()


modeler = HeartDiseasePredictionPipeline(model_path=model_path,preprocessor_path=preprocessor_path)



@app.route("/",methods=["GET","POST"])
def index():
    if request.method == "GET" or request.method != "POST":
        return render_template("index.html",metadata=meta_data)
    elif request.method == "POST":
        # get all form data
        form_data = [  key for key in meta_data.keys() if key in meta_data['categorical_columns']] + meta_data['numerical_columns']
        
        data = [ request.form[inpt] if request.form[inpt] else np.nan for inpt in form_data ]


        data = pd.DataFrame([data],columns=form_data)
        data['HeartDisease'] = 0
        if data.isna().sum().sum() > 0 :
            session['error'] = "Please fill up all form data"
            return redirect("/")
        else:
            try:
                renamed_columns = list(map(modeler.column_name_transformer,data.columns))
                # data[renamed_columns] = np.nan
                data_encoded = modeler.preprocess(data)
                result,probability = modeler.predict(data_encoded)
                
                return render_template("index.html",metadata=meta_data,prediction=result,probability=probability)
    
            except Exception as ex:
                print(ex)
                session["error"] = "Some error occurred please try again"
                return redirect("/")
    else:
        return redirect("/")

if __name__ == '__main__':
    app.run(debug=True)
