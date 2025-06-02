import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import timedelta, datetime

# Cargar variables de entorno si no estás en producción
if os.getenv("ENV") != "production":
    load_dotenv()

API_URL = os.getenv("API_URL")

def get_and_clean_category_data():
    try:
        # 1) Llamada al endpoint
        endpoint = f"{API_URL.rstrip('/')}/category/sales-category"
        response = requests.get(endpoint)
        response.raise_for_status()

        # 2) Convertir JSON a DataFrame
        raw_data = response.json()
        df = pd.DataFrame(raw_data)

        # 3) Renombrar 'week' a 'date'
        df = df.rename(columns={'week': 'date'})

        # 4) Convertir 'date' a datetime y ordenar
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)

        # 5) Imprimir cuántas filas trajo y las primeras 5 filas
        print(f"[get_and_clean_category_data] Datos obtenidos: {len(df)} filas.")
        print("Primeras 5 filas del DataFrame de categorías ya limpio:")
        print(df.head().to_string(index=False))

        return df
    except Exception as e:
        print(f"[get_and_clean_category_data] Error: {e}")
        return pd.DataFrame()


def get_and_clean_product_data(product_id: str) -> pd.DataFrame:
    try:
        endpoint = f"{API_URL.rstrip('/')}/product/weekly-sales/{product_id}"
        resp = requests.get(endpoint)
        resp.raise_for_status()

        raw = resp.json()
        df = pd.DataFrame(raw)

        # Renombrar columnas
        df = df.rename(columns={'week': 'date', 'totalSales': 'value'})
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        # 1) Imprimir cuántas filas trajo y las primeras 5 filas:
        print(f"[get_and_clean_product_data] Producto {product_id}: {len(df)} filas limpias.")
        print("Primeras 5 filas del DataFrame de producto:")
        print(df.head().to_string(index=False))

        return df
    except Exception as e:
        print(f"[get_and_clean_product_data] Error en producto {product_id}: {e}")
        return pd.DataFrame()


def get_all_products() -> list[dict]:
    try:
        endpoint = f"{API_URL.rstrip('/')}/product"
        resp = requests.get(endpoint)
        resp.raise_for_status()
        all_products = resp.json()
        return [{"id": p["id"], "name": p["name"]} for p in all_products]
    except Exception as e:
        print(f"[get_all_products] Error: {e}")
        return []
