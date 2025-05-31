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
    endpoint = f"{API_URL.rstrip('/')}/category/sales-category"
    response = requests.get(endpoint)
    response.raise_for_status()

    raw_data = response.json()
    df = pd.DataFrame(raw_data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # 1) Imprimir cuántas filas trajo y las primeras 5 filas:
    print(f"[get_and_clean_category_data] Datos obtenidos: {len(df)} filas.")
    print("Primeras 5 filas del DataFrame de categorías ya limpio:")
    print(df.head().to_string(index=False))  # .to_string() para mostrar en bloque

    # 2) Opcional: si quieres ver TODO, quita el .head() (cuidado si es muy grande)
    # print("DataFrame completo:\n", df.to_string(index=False))

    return df


def get_and_clean_product_data(product_id: str) -> pd.DataFrame:
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

    # 2) Si necesitas ver todo el DataFrame (pero puede ser muchas filas):
    # print("DataFrame completo:\n", df.to_string(index=False))

    return df

def get_all_products() -> list[dict]:
    """
    Consume el endpoint GET /product de la API interna
    y devuelve la lista de productos (cada uno con id y name).
    """
    endpoint = f"{API_URL.rstrip('/')}/product"
    resp = requests.get(endpoint)
    resp.raise_for_status()
    all_products = resp.json()
    # Simplificamos a sólo id y name:
    return [{"id": p["id"], "name": p["name"]} for p in all_products]
