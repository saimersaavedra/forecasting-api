import requests
import pandas as pd

API_URL = "https://farmaciabe-production.up.railway.app/api/dashboard/sales-category"

def get_and_clean_data():
    # 1. Obtener datos de la API
    response = requests.get(API_URL)
    response.raise_for_status()  # Error si falla

    raw_data = response.json()

    # 2. Convertir a DataFrame
    df = pd.DataFrame(raw_data)

    # 3. Convertir la fecha a datetime
    df['date'] = pd.to_datetime(df['date'])

    # 4. Ordenar por fecha ascendente
    df = df.sort_values('date').reset_index(drop=True)

    return df
