from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.providers.google.cloud.transfers.gcs_to_bigquery import GCSToBigQueryOperator
from airflow.providers.google.cloud.operators.gcs import GCSCreateBucketOperator
from airflow.providers.google.cloud.operators.gcs import GCSDeleteBucketOperator
import uuid
from datetime import timedelta
import datetime as dt
from airflow.utils.dates import days_ago
import fnmatch
import yfinance as yf
from google.cloud import storage
import pandas as pd 
import re
import pickle
from sklearn.preprocessing import  StandardScaler
import random2 as random
import numpy as np


PROJECT_ID="test-proyecto-final-406120"
STAGING_DATASET = "stock_dataset"
LOCATION = "us-central1"


default_args = {
    'owner': 'Amara',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'start_date':  days_ago(1),
    'retry_delay': timedelta(minutes=5),
}


# Inicio de sesión

storage_client = storage.Client()

nombre_bucket = 'us-central1-demo-environmen-24f1181f-bucket'
bucket = storage_client.bucket(nombre_bucket)
map_scaler_nm = 'map_scaler_produccion.pkl'
df_corre_nm = 'df_correlacion.csv'

# Descarga de map scaler en pkl

blob = bucket.blob(f'dags/{map_scaler_nm}')
blob.download_to_filename(map_scaler_nm)

with open(map_scaler_nm, 'rb') as file_map:
    map_scaler = pickle.load(file_map)

# Descarga de dataframe de mejores correlaciones

blob = bucket.blob(f'dags/{df_corre_nm}')
blob.download_to_filename(df_corre_nm)



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


def get_data():
    

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

    # Convert the data to CSV and encode 
    data_info = df_acum_info.to_csv(index=False).encode()
    data_model = df_acum[['FECHA'] + lista_best_features].to_csv(index=False).encode()


    # Create a storage client
    storage_client = storage.Client()

    # Get a list of all buckets
    buckets = list(storage_client.list_buckets())

    # Filter the list of buckets to only include those with the desired prefix
    buckets_with_prefix = [bucket for bucket in buckets if fnmatch.fnmatch(bucket.name, 'the_demo_*')]

    #Choose the matching buckets to upload the data to
    bucket = buckets_with_prefix[0]

    # Upload the data to the selected bucket
    blob = bucket.blob('stock_data.csv')
    blob.upload_from_string(data_info)

    blob = bucket.blob('stock_data_model.csv')
    blob.upload_from_string(data_model)
    
    print(f"data sucessfully uploadesd to {bucket}")
    


with DAG('Stock_data',
         start_date=days_ago(1), 
         schedule_interval="@once",
         catchup=False, 
         default_args=default_args, 
         tags=["gcs", "bq"]
) as dag:

    generate_uuid = PythonOperator(
            task_id="generate_uuid", 
            python_callable=lambda: "the_demo_" + str(uuid.uuid4()),
        )

    create_bucket = GCSCreateBucketOperator(
            task_id="create_bucket",
            bucket_name="{{ task_instance.xcom_pull('generate_uuid') }}",
            project_id=PROJECT_ID,
        )

    pull_stock_data_to_gcs = PythonOperator(
        task_id = 'pull_stock_data_to_gcs',
        python_callable = get_data,
        )

    load_to_bq_1 = GCSToBigQueryOperator(
        task_id = 'load_to_bq_1',
        bucket = "{{ task_instance.xcom_pull('generate_uuid') }}",
        source_objects = ['stock_data.csv'],
        destination_project_dataset_table = f'{PROJECT_ID}:{STAGING_DATASET}.stock_data_table',
        write_disposition='WRITE_TRUNCATE',
        source_format = 'csv',
        allow_quoted_newlines = 'true',
        skip_leading_rows = 1,
        schema_fields=[
        {'name': 'FECHA', 'type': 'DATE', 'mode': 'NULLABLE'}] + 
        esquema_tb_info,
        )

    load_to_bq_2 = GCSToBigQueryOperator(
        task_id = 'load_to_bq_2',
        bucket = "{{ task_instance.xcom_pull('generate_uuid') }}",
        source_objects = ['stock_data_model.csv'],
        destination_project_dataset_table = f'{PROJECT_ID}:{STAGING_DATASET}.stock_data_model_table',
        write_disposition='WRITE_TRUNCATE',
        source_format = 'csv',
        allow_quoted_newlines = 'true',
        skip_leading_rows = 1,
        schema_fields=[
        {'name': 'FECHA', 'type': 'DATE', 'mode': 'NULLABLE'},
        {'name': 'Target', 'type': 'INT64', 'mode': 'NULLABLE'}] + 
        esquema_tb_trans,
        )
    
    delete_bucket = GCSDeleteBucketOperator(
            task_id="delete_bucket",
            bucket_name="{{ task_instance.xcom_pull('generate_uuid') }}",
        )

    (
        generate_uuid
        >> create_bucket
        >> pull_stock_data_to_gcs
        >> load_to_bq_1
        >> load_to_bq_2
        >> delete_bucket
    )
















