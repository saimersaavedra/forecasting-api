import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

# Cargar variables de entorno si no estás en producción
if os.getenv("ENV") != "production":
    load_dotenv()

API_URL = os.getenv("API_URL")

def get_and_clean_data():
    endpoint = f"{API_URL.rstrip('/')}/category/sales-category"  
    response = requests.get(endpoint)
    response.raise_for_status()

    raw_data = response.json()
    df = pd.DataFrame(raw_data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df

def get_and_clean_product_data(product_id: str) -> pd.DataFrame:
    """
    Fetch y limpia datos de ventas semanales por producto.
    Elimina semanas futuras a la fecha actual.
    """
    endpoint = f"{API_URL.rstrip('/')}/product/weekly-sales/{product_id}"
    resp = requests.get(endpoint)
    resp.raise_for_status()

    raw = resp.json()
    df = pd.DataFrame(raw)
    # renombrar columnas para Prophet
    df = df.rename(columns={'week': 'date', 'totalSales': 'value'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    # eliminar semanas futuras
    today = datetime.now()
    df = df[df['date'] <= today].reset_index(drop=True)
    return df