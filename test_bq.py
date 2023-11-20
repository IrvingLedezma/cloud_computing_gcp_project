# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 12:45:38 2023

@author: irvin
"""

from google.cloud import bigquery




# Configura la conexión a BigQuery (asegúrate de tener las credenciales configuradas)
client = bigquery.Client(project="test-proyecto-final")

# Construye la consulta SQL
sql_query = """
    SELECT *
    FROM `test-proyecto-final.stock_dataset.stock_data_table`
    ORDER BY FECHA DESC
    LIMIT 1
"""

# Ejecuta la consulta
query_job = client.query(sql_query)

# Recupera los resultados
results = query_job.result()

# Itera sobre los resultados (debería haber solo un resultado, ya que limitamos a 1)
for row in results:
    print(row)
