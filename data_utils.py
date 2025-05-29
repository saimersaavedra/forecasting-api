import os
import requests
import pandas as pd
from dotenv import load_dotenv

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
