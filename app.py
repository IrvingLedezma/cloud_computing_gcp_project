# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 13:18:21 2023

@author: irvin
"""


import config
import pipeline
from flask import Flask,request, url_for, redirect, render_template, jsonify
import pandas as pd
import numpy as np
from datetime import datetime


app = Flask(__name__)


nombre_archivo_modelo = 'modelo_logistico_produccion.pkl'

with open(nombre_archivo_modelo, 'rb') as archivo_modelo:
    modelo_cargado = pickle.load(archivo_modelo)

index_train = 457





@app.route('/')
def home():
    datos_tabla = [
        {'nombre': 'John', 'edad': 30, 'ciudad': 'Ciudad A'},
        {'nombre': 'Jane', 'edad': 25, 'ciudad': 'Ciudad B'},
        {'nombre': 'Bob', 'edad': 35, 'ciudad': 'Ciudad C'},
        {'nombre': 'Bob', 'edad': 35, 'ciudad': 'Ciudad C'},
        {'nombre': 'Bob', 'edad': 35, 'ciudad': 'Ciudad C'}
    ]    
    
    fecha_actual = datetime.now().strftime('%Y-%m-%d')
    
    return render_template("home.html", 
                           datos_tabla=datos_tabla,
                           fecha_actual=fecha_actual)

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = 10

    datos_tabla = [
        {'nombre': 'John', 'edad': 30, 'ciudad': 'Ciudad A'},
        {'nombre': 'Jane', 'edad': 25, 'ciudad': 'Ciudad B'},
        {'nombre': 'Bob', 'edad': 35, 'ciudad': 'Ciudad C'}
    ]

    return render_template('home.html',pred='Expected Bill will be {}'.format(prediction),
                           datos_tabla=datos_tabla)

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = 10
    output = prediction
    return jsonify(output)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG_MODE)