import pandas as pd 
from flask import Flask, request, Response
from house.House import House
import pickle

#  carregando modelo 
model = pickle.load(open('/home/andre/repos/House-Prices/model/house_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/house/predict', methods=['POST'])

def house_predict():
    test_json = request.get_json()
    
    
    if test_json: #tem dados
        if isinstance(test_json, dict): # exemplo único 
            test_raw = pd.Dataframe(test_json, index=[0])
        
        else:
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
        # instanciando classe House 
        pipeline = House()
        
        # limpeza dos dados
        df1 = pipeline.data_cleaning(test_raw)
        
        # atributo dos dados 
        df2 = pipeline.feature_engineer(df1)
        
        # preparação dos dados
        df3 = pipeline.data_preparation(df1, df2)
        
        # predição dos dados
        df_response = pipeline.get_prediction(model, test_raw, df3)
                               
        
        return df_response
    
    
    else:
        return Response('{}', status=200, mimetype='application/json')
    
    print(test_json)
    
if __name__ == '__main__':
    app.run('127.0.0.1')
