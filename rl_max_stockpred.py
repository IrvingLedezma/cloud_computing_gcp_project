# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import random2 as random

from sklearn.preprocessing import  StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import pickle

import re


import os
os.chdir('C:/Users/irvin/OneDrive/Desktop/python_scripts/cloud_computing')


class Get_stock:
    def __init__(self,start_date,end_date, portfolio):
        self.start_date = start_date
        self.end_date = end_date
        self.portfolio = portfolio
        self.interval = "1d"
        self.dict_df = {}

    def get_data(self):
        for key in self.portfolio.keys():
            ticker=yf.Ticker(key)
            self.dict_df[key] = ticker.history(start=self.start_date, end=self.end_date, interval=self.interval)


portfolio = {
    'WALMEX.MX':0,
    '^IXIC':0,
    '^GSPC':0,
    '^DJI':0,
    '^NYA':0}


"""=========================================================================================
    Importación de datos para entrenamiento
========================================================================================="""

start_date = dt.datetime(2022, 1, 1)
end_date = dt.datetime.now()
stock = Get_stock(start_date,end_date,portfolio)
stock.get_data()


"""=========================================================================================
    Unión de series de datos
========================================================================================="""

dict_col = {}
col_chan = ['Open', 'High', 'Low', 'Close', 'Volume']
for stock_name in portfolio.keys():
    stock.dict_df[stock_name]['FECHA'] = pd.to_datetime(stock.dict_df[stock_name].index.strftime('%Y-%m-%d'))
    stock.dict_df[stock_name].rename(columns = {
        i:i + f'_{stock_name}' for i in col_chan
        }, inplace=True)
    dict_col[stock_name] = ['FECHA'] + [i + f'_{stock_name}' for i in col_chan]


df_acum = pd.DataFrame()
df_acum = pd.merge(
    stock.dict_df['WALMEX.MX'][dict_col['WALMEX.MX']],
    stock.dict_df['^IXIC'][dict_col['^IXIC']],
    on ='FECHA'
    )

df_acum = pd.merge(
    df_acum,
    stock.dict_df['^GSPC'][dict_col['^GSPC']],
    on ='FECHA'
    )

df_acum = pd.merge(
    df_acum,
    stock.dict_df['^DJI'][dict_col['^DJI']],
    on ='FECHA'
    )

df_acum = pd.merge(
    df_acum,
    stock.dict_df['^NYA'][dict_col['^NYA']],
    on ='FECHA'
    )

patron = re.compile(r'[^a-zA-Z_]')

def limpiar_string(s):
    return re.sub(patron, '', s)

lista_strings = df_acum.columns
lista_limpiada = list(map(limpiar_string, lista_strings))
df_acum.columns = lista_limpiada


all_field =[i for i in dict_col['^IXIC'] + dict_col['WALMEX.MX']+
            dict_col['^GSPC'] + dict_col['^DJI'] +
            dict_col['^NYA'] if i not in ['FECHA']]

all_field = list(map(limpiar_string, all_field))


all_field_lags = []

for col in all_field :
    df_acum[col + f'_L1'] = df_acum[col].shift(1)
    df_acum[col + f'_L2'] = df_acum[col].shift(2)
    df_acum[col + f'_L3'] = df_acum[col].shift(3)

    all_field_lags.append(col + f'_L1')
    all_field_lags.append(col + f'_L2')
    all_field_lags.append(col + f'_L3')


"""=========================================================================================
    Cálculo de target
========================================================================================="""

def get_target(x):
    if x>0: return 1
    else: return 0

df_acum['Target'] = np.vectorize(get_target)(df_acum['Close_WALMEXMX'] - df_acum['Close_WALMEXMX_L1'])


"""=========================================================================================
    Estandarización de columnas
========================================================================================="""

map_scaler = {}
for col in all_field_lags:
    map_scaler[col] = StandardScaler()
    df_acum[col] = map_scaler[col].fit_transform(df_acum[[col]])


with open('map_scaler_produccion.pkl', 'wb') as file_map:
    pickle.dump(map_scaler, file_map)


"""=========================================================================================
    Búsqueda de mejores combinaciones lineales
========================================================================================="""

threshold_corre = 0.11
map_features = {}
features = []
list_corr = []

for seed in range(50000):
    random.seed(seed)
    subset = random.sample(all_field_lags , 3)

    df_acum[f'feature_1_{seed}'] = (df_acum[subset[2]] - df_acum[subset[1]]) / (df_acum[subset[0]] - df_acum[subset[1]])
    df_acum[f'feature_2_{seed}'] = (df_acum[subset[0]] - df_acum[subset[1]]) * df_acum[subset[2]]
    df_acum[f'feature_3_{seed}'] = (df_acum[subset[0]] - df_acum[subset[2]]) / (df_acum[subset[1]] - df_acum[subset[2]])

    df_corr_1 = df_acum[['Target'] + [f'feature_1_{seed}'] ].corr().abs()
    df_corr_2 = df_acum[['Target'] + [f'feature_2_{seed}'] ].corr().abs()
    df_corr_3 = df_acum[['Target'] + [f'feature_3_{seed}'] ].corr().abs()

    corr_1 = round(abs(df_corr_1[f'feature_1_{seed}'][0]),4)
    corr_2 = round(abs(df_corr_2[f'feature_2_{seed}'][0]),4)
    corr_3 = round(abs(df_corr_3[f'feature_3_{seed}'][0]),4)

    if corr_1 > threshold_corre and corr_1 not in list_corr:
        features.append(f'feature_1_{seed}')
        list_corr.append(corr_1)
        map_features[seed] = f'{subset[0]} + {subset[1]} + {subset[2]}'
    else:
        del df_acum[f'feature_1_{seed}']

    if corr_2 > threshold_corre and corr_2 not in list_corr:
        features.append(f'feature_2_{seed}')
        list_corr.append(corr_2)
        map_features[seed] = f'{subset[0]} + {subset[1]} + {subset[2]}'
    else:
        del df_acum[f'feature_2_{seed}']

    if corr_3 > threshold_corre and corr_3 not in list_corr:
        features.append(f'feature_3_{seed}')
        list_corr.append(corr_3)
        map_features[seed] = f'{subset[0]} + {subset[1]} + {subset[2]}'
    else:
        del df_acum[f'feature_3_{seed}']




"""=========================================================================================
    Entrenar modelo con mejores variables
========================================================================================="""


df_corr = df_acum[['Target'] + features + all_field_lags ].corr().abs()
df_corr.to_csv('df_correlacion.csv')
len(df_corr)


"""=========================================================================================
    Entrenar modelo con mejores variables
========================================================================================="""


hyperparameters = []
num_features = []
mean_accuracy = []
std_accuracy = []
train_accuracy = []
test_accuracy = []


df_acum.dropna(inplace=True)
cut_train = int(len(df_acum)*.9)
df_train = df_acum.iloc[0:cut_train].copy()
df_test = df_acum.iloc[cut_train:].copy()


for index_corre in range(2,269):

    best_var = list(df_corr['Target'].sort_values(ascending=False).iloc[1:index_corre].index)

    X_train = df_train[best_var]
    y_train = df_train['Target']

    X_test = df_test[best_var]
    y_test = df_test['Target']

    """=======================================================
        Logistic Regression
    ======================================================="""

    for solver in ['liblinear','saga']:
        for penalty in ['l1', 'l2']:
            for C in [0.0001, 0.001, 0.01, 0.1, 1] :
                model_final = LogisticRegression(C=C, penalty=penalty,solver=solver )

                metrics         = []
                best_model      = []
                best_evaluation = 0
                kf = KFold(n_splits = 4)
                try:
                    for train, test in kf.split(X_train):
                        X_train_cv = X_train.iloc[train]
                        X_test_cv = X_train.iloc[test]
                        y_train_cv = y_train.values[train]
                        y_test_cv = y_train.values[test]
                        model_final.fit(X_train_cv,y_train_cv)
                        evaluation = model_final.score(X_test_cv,y_test_cv)
                        if evaluation>best_evaluation:
                            best_model = model_final
                            best_evaluation = evaluation
                        metrics.append(evaluation)
                    mean_accuracy.append(np.mean(metrics))
                    std_accuracy.append(np.std(metrics))
                    train_accuracy.append(best_model.score(X_train,y_train))
                    test_accuracy.append(best_model.score(X_test,y_test))
                except:
                    mean_accuracy.append(0)
                    std_accuracy.append(0)
                    train_accuracy.append(0)
                    test_accuracy.append(0)

                hyperparameters.append(f'penalty: {penalty}, C: {C}, solver: {solver}')
                num_features.append(index_corre)



metricas_features = pd.DataFrame({
    'hyperparameters':hyperparameters,
    'num_features':num_features,
    'mean_accuracy':mean_accuracy,
    'std_accuracy':std_accuracy,
    'train_accuracy':train_accuracy,
    'test_accuracy':test_accuracy
    })


metricas_features.to_csv('metricas_features.csv')



"""=========================================================================================
    Exportar mejor modelo obtenido. 
    
    Número de caracteristicas:
            169 
    Mejores hiperparametros (lr):
            best_hiper = {penalty: 'l2', C: 0.001, solver: 'liblinear'}
========================================================================================="""


hyperparameters = []
num_features = []
mean_accuracy = []
std_accuracy = []
train_accuracy = []
test_accuracy = []


df_acum.dropna(inplace=True)
cut_train = int(len(df_acum)*.9)
df_train = df_acum.iloc[0:cut_train].copy()
df_test = df_acum.iloc[cut_train:].copy()


for index_corre in [169]:

    best_var = list(df_corr['Target'].sort_values(ascending=False).iloc[1:index_corre].index)

    X_train = df_train[best_var]
    y_train = df_train['Target']

    X_test = df_test[best_var]
    y_test = df_test['Target']
    
    best_hiper = {'penalty': 'l2', 'C': 0.001, 'solver': 'liblinear'}

    """=======================================================
        Logistic Regression
    ======================================================="""

    model_final = LogisticRegression(**best_hiper)

    metrics         = []
    best_model      = []
    best_evaluation = 0
    kf = KFold(n_splits = 4)
    
    for train, test in kf.split(X_train):
        X_train_cv = X_train.iloc[train]
        X_test_cv = X_train.iloc[test]
        y_train_cv = y_train.values[train]
        y_test_cv = y_train.values[test]
        model_final.fit(X_train_cv,y_train_cv)
        evaluation = model_final.score(X_test_cv,y_test_cv)
        if evaluation>best_evaluation:
            best_model = model_final
            best_evaluation = evaluation
        metrics.append(evaluation)
    mean_accuracy.append(np.mean(metrics))
    std_accuracy.append(np.std(metrics))
    train_accuracy.append(best_model.score(X_train,y_train))
    test_accuracy.append(best_model.score(X_test,y_test))

    num_features.append(index_corre)



metricas_features_final = pd.DataFrame({
    'num_features':num_features,
    'mean_accuracy':mean_accuracy,
    'std_accuracy':std_accuracy,
    'train_accuracy':train_accuracy,
    'test_accuracy':test_accuracy
    })


with open('modelo_logistico_produccion.pkl', 'wb') as archivo_modelo:
    pickle.dump(best_model, archivo_modelo)

































# Definicion de esquema de tabla informacion stock

list_tb_info = ['Open_WALMEXMX', 'High_WALMEXMX', 'Low_WALMEXMX','Close_WALMEXMX', 'Volume_WALMEXMX']
esquema_tb_info = [{'name': col, 'type': 'FLOAT64', 'mode': 'NULLABLE'} for col in list_tb_info if col not in ['FECHA', 'Target']]


# Definicion de esquema de tabla transformaciones lista para alimentar a RL

df_corr = pd.read_csv('df_correlacion.csv')
lista_best_features = df_corr['Unnamed: 0'].to_list()
esquema_tb_trans = [{'name': col, 'type': 'FLOAT64', 'mode': 'NULLABLE'} for col in lista_best_features if col not in ['FECHA', 'Target']]


portfolio = {
    'WALMEX.MX':0,
    '^IXIC':0,
    '^GSPC':0,
    '^DJI':0,
    '^NYA':0}


class Get_stock:
    def __init__(self,start_date,end_date, portfolio):
        self.start_date = start_date
        self.end_date = end_date
        self.portfolio = portfolio
        self.interval = "1d"
        self.dict_df = {}

    def get_data(self):
        for key in self.portfolio.keys():
            ticker=yf.Ticker(key)
            self.dict_df[key] = ticker.history(start=self.start_date, end=self.end_date, interval=self.interval)


patron = re.compile(r'[^a-zA-Z_]')
def limpiar_string(s):
    return re.sub(patron, '', s)



def get_target(x):
    if x>0: return 1
    else: return 0


    
"""=========================================================================================
    Importación de datos para entrenamiento
========================================================================================="""

start_date = dt.datetime(2022, 1 , 1)
end_date = dt.datetime.now()
stock = Get_stock(start_date,end_date,portfolio)
stock.get_data()


"""=========================================================================================
    Unión de series de datos
========================================================================================="""

dict_col = {}
col_chan = ['Open', 'High', 'Low', 'Close', 'Volume']
for stock_name in portfolio.keys():
    stock.dict_df[stock_name]['FECHA'] = pd.to_datetime(stock.dict_df[stock_name].index.strftime('%Y-%m-%d'))
    stock.dict_df[stock_name].rename(columns = {
        i:i + f'_{stock_name}' for i in col_chan
        }, inplace=True)
    dict_col[stock_name] = ['FECHA'] + [i + f'_{stock_name}' for i in col_chan]


df_acum = pd.DataFrame()
df_acum = pd.merge(
    stock.dict_df['WALMEX.MX'][dict_col['WALMEX.MX']],
    stock.dict_df['^IXIC'][dict_col['^IXIC']],
    on ='FECHA'
    )

df_acum = pd.merge(
    df_acum,
    stock.dict_df['^GSPC'][dict_col['^GSPC']],
    on ='FECHA'
    )

df_acum = pd.merge(
    df_acum,
    stock.dict_df['^DJI'][dict_col['^DJI']],
    on ='FECHA'
    )

df_acum = pd.merge(
    df_acum,
    stock.dict_df['^NYA'][dict_col['^NYA']],
    on ='FECHA'
    )
    
lista_strings = df_acum.columns
lista_limpiada = list(map(limpiar_string, lista_strings))
df_acum.columns = lista_limpiada

df_acum_info = df_acum[['FECHA'] + list_tb_info ].copy()


"""=========================================================================================
    Cálculo de lags
========================================================================================="""


all_field =[i for i in dict_col['^IXIC'] + dict_col['WALMEX.MX']+
            dict_col['^GSPC'] + dict_col['^DJI'] +
            dict_col['^NYA'] if i not in ['FECHA']]

all_field = list(map(limpiar_string, all_field))


all_field_lags = []

for col in all_field :
    df_acum[col + f'_L1'] = df_acum[col].shift(1)
    df_acum[col + f'_L2'] = df_acum[col].shift(2)
    df_acum[col + f'_L3'] = df_acum[col].shift(3)

    all_field_lags.append(col + f'_L1')
    all_field_lags.append(col + f'_L2')
    all_field_lags.append(col + f'_L3')



"""=========================================================================================
    Cálculo de target
========================================================================================="""


df_acum['Target'] = np.vectorize(get_target)(df_acum['Close_WALMEXMX'] - df_acum['Close_WALMEXMX_L1'])


"""=========================================================================================
    Estandarización de columnas
========================================================================================="""

for col in all_field_lags:
    df_acum[col] = map_scaler[col].transform(df_acum[[col]])


"""=========================================================================================
    Lectura de archivo de mejores caracteristicas (best_random)
========================================================================================="""

best_random = [int(re.search(r'feature_\d+_(\d+)', elemento).group(1)) for elemento in lista_best_features if elemento.startswith('feature')]


"""=========================================================================================
    Cálculo de mejores caracteristicas
========================================================================================="""

for seed in best_random:
    random.seed(seed)
    subset = random.sample(all_field_lags , 3)

    df_acum[f'feature_1_{seed}'] = (df_acum[subset[2]] - df_acum[subset[1]]) / (df_acum[subset[0]] - df_acum[subset[1]])
    df_acum[f'feature_2_{seed}'] = (df_acum[subset[0]] - df_acum[subset[1]]) * df_acum[subset[2]]
    df_acum[f'feature_3_{seed}'] = (df_acum[subset[0]] - df_acum[subset[2]]) / (df_acum[subset[1]] - df_acum[subset[2]])


# Eliminar filas vacias
df_acum.dropna(inplace=True)
df_acum[['FECHA'] + lista_best_features].to_csv('df_acum.csv', index=False)










