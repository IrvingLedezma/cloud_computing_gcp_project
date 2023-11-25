# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 12:45:38 2023

@author: irvin
"""

from google.cloud import bigquery

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
        `test-proyecto-final.stock_dataset.stock_data_table`
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
    FROM `test-proyecto-final.stock_dataset.stock_data_table`
    ORDER BY FECHA DESC
    LIMIT 1
"""

# Ejecuta la consulta
query_job = client.query(sql_query)

# Recupera los resultados
results_modelo = query_job.to_dataframe()


print(results_info)
print(results_modelo)