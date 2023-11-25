# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 13:18:21 2023

@author: irvin
"""

import config
from flask import Flask,request, url_for, redirect, render_template, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
from google.cloud import bigquery



"""


# Crea un DataFrame de pandas con valores dummy
data = {
    'avg_Open': [100.0],
    'avg_High': [105.0],
    'avg_Low': [95.0],
    'avg_Close': [100.5],
    'avg_Volume': [1000000],
    'std_Open': [2.0],
    'std_High': [2.5],
    'std_Low': [1.5],
    'std_Close': [2.0],
    'std_Volume': [50000]
}

results_info = pd.DataFrame(data)
results_modelo = pd.read_csv('df_acum.csv')

"""


# Calcular la fecha actual y la fecha hace 30 días
fecha_actual = datetime.now().strftime('%Y-%m-%d')
fecha_30_dias_atras = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

# Construir la consulta SQL para la información del stock
sql_query = f"""
    SELECT
        AVG(Open_WALMEXMX) AS avg_Open,
        AVG(High_WALMEXMX) AS avg_High,
        AVG(Low_WALMEXMX) AS avg_Low,
        AVG(Close_WALMEXMX) AS avg_Close,
        AVG(Volume_WALMEXMX) AS avg_Volume,
        STDDEV_POP(Open_WALMEXMX) AS std_Open,
        STDDEV_POP(High_WALMEXMX) AS std_High,
        STDDEV_POP(Low_WALMEXMX) AS std_Low,
        STDDEV_POP(Close_WALMEXMX) AS std_Close,
        STDDEV_POP(Volume_WALMEXMX) AS std_Volume
    FROM
        `test-proyecto-final-406120.stock_dataset.stock_data_model_table`
    WHERE
        FECHA BETWEEN '{fecha_30_dias_atras}' AND '{fecha_actual}'
"""

# Ejecuta la consulta
query_job = client.query(sql_query)

# Recupera los resultados
results_info = query_job.to_dataframe()


# Construye la consulta SQL para la información de que se utiliza en el modelo
sql_query = """
    SELECT *
    FROM `test-proyecto-final-406120.stock_dataset.stock_data_model_table`
    ORDER BY FECHA DESC
    LIMIT 1
"""

# Ejecuta la consulta
query_job = client.query(sql_query)

# Recupera los resultados
results_modelo = query_job.to_dataframe()


nombre_archivo_modelo = 'modelo_logistico_produccion.pkl'

with open(nombre_archivo_modelo, 'rb') as archivo_modelo:
    modelo_cargado = pickle.load(archivo_modelo)


# Definición de indices para cálculo de métricas

index_train = 411
index_tot_train = 457

list_columns_model = list(modelo_cargado.feature_names_in_)

# Separación de muestras

X_train = results_modelo.iloc[:index_train][list_columns_model]
y_train = results_modelo.iloc[:index_train]['Target']

X_test = results_modelo.iloc[index_train:index_tot_train][list_columns_model]
y_test = results_modelo.iloc[index_train:index_tot_train]['Target']

X_test2 = results_modelo.iloc[index_train:][list_columns_model]
y_test2 = results_modelo.iloc[index_train:]['Target']

# Cálculo de métricas

acc_train = modelo_cargado.score(X_train,y_train)
acc_test = modelo_cargado.score(X_test,y_test)
acc_test2 = modelo_cargado.score(X_test2,y_test2)


app = Flask(__name__)
"""
	position: absolute;
	top: 25%;
	left: 50%;
    transform: translate(-50%, -50%);	width:600px;
    width: 600px;
	height:200px;

"""

@app.route('/')
def home():
    
    datos_tabla = [
        {'Parametro': 'Promedio', 
         'Open': round(results_info['avg_Open'][0],1), 
         'High': round(results_info['avg_High'][0],1), 
         'Low': round(results_info['avg_Low'][0],1), 
         'Close': round(results_info['avg_Close'][0],1), 
         'Volume': '{:,.0f}'.format(round(results_info['avg_Volume'][0],0))
         },
        {'Parametro': 'Desviación Estandar', 
         'Open': round(results_info['std_Open'][0],1), 
         'High': round(results_info['std_High'][0],1), 
         'Low': round(results_info['std_Low'][0],1), 
         'Close': round(results_info['std_Close'][0],1), 
         'Volume': '{:,.0f}'.format(round(results_info['std_Volume'][0],0))
         }
    ]    
    
    fecha_actual = datetime.now().strftime('%Y-%m-%d')


    datos_tabla_2 = [
        {'Metrica': 'Accuracy', 
         'train': round(acc_train,2), 
         'test': round(acc_test,2), 
         'test_act': round(acc_test2,2) 
         }
    ]    
    
    return render_template("home.html", 
                           datos_tabla=datos_tabla,
                           datos_tabla_2=datos_tabla_2, 
                           fecha_actual=fecha_actual)

@app.route('/predict',methods=['POST'])
def predict():
    datos_tabla = [
        {'Parametro': 'Promedio', 
         'Open': round(results_info['avg_Open'][0],1), 
         'High': round(results_info['avg_High'][0],1), 
         'Low': round(results_info['avg_Low'][0],1), 
         'Close': round(results_info['avg_Close'][0],1), 
         'Volume': '{:,.0f}'.format(round(results_info['avg_Volume'][0],0))
         },
        {'Parametro': 'Desviación Estandar', 
         'Open': round(results_info['std_Open'][0],1), 
         'High': round(results_info['std_High'][0],1), 
         'Low': round(results_info['std_Low'][0],1), 
         'Close': round(results_info['std_Close'][0],1), 
         'Volume': '{:,.0f}'.format(round(results_info['std_Volume'][0],0))
         }
    ]    
    
    fecha_actual = datetime.now().strftime('%Y-%m-%d')


    datos_tabla_2 = [
        {'Metrica': 'Accuracy', 
         'train': round(acc_train,2), 
         'test': round(acc_test,2), 
         'test_act': round(acc_test2,2) 
         }
    ]    


    prediction = .0934
    return render_template('home.html',
                           pred=f'La probabilidad de crecimiento al 2023-11-02 es: {round(prediction,2)}'.format(prediction),
                           datos_tabla=datos_tabla,
                           datos_tabla_2=datos_tabla_2, 
                           fecha_actual=fecha_actual)



@app.route('/predict_api',methods=['POST'])
def predict_api():
    prediction = 10
    output = prediction
    return jsonify(output)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG_MODE)