# scripts/generate_forecasts.py

import os
import json
import numpy as np
from datetime import datetime
from predictor import (
    predict_category_sales,
    predict_product_sales,
    prepare_category_df,
    prepare_product_df,
    es_forecast_inestable
)
from data_utils import get_and_clean_category_data, get_and_clean_product_data, get_all_products

CATEGORIES_CACHE_PATH = "cache/categories_forecast.json"
PRODUCTS_CACHE_PATH   = "cache/products_forecast.json"
WEEKS = 4

def generate_category_forecasts():
    # 1) Obtener y limpiar datos de categorías
    df = get_and_clean_category_data()
    categories = [col for col in df.columns if col != "date"]
    results = []

    for category in categories:
        # 2) Construir “history” para cada fila del DataFrame
        history = [
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "value": int(round(row[category]))
            }
            for _, row in df.iterrows()
        ]

        # 3) Predecir con la función robusta
        df_fore = predict_category_sales(df, category, weeks=WEEKS)

        # 4) Construir lista “forecasting” a partir de df_fore
        forecasting = [
            {
                "date": row["ds"].strftime("%Y-%m-%d"),
                "value": int(round(row["yhat"]))
            }
            for _, row in df_fore.iterrows()
        ]

        results.append({
            "category":    category,
            "history":     history,
            "forecasting": forecasting,
            "weeks":       WEEKS
        })

    # 5) Asegurarse de que exista la carpeta “cache”
    os.makedirs(os.path.dirname(CATEGORIES_CACHE_PATH), exist_ok=True)

    # 6) Guardar JSON (lista de categorías)
    print(f"[generate_category_forecasts] Guardando {len(results)} categorías en '{CATEGORIES_CACHE_PATH}'")
    with open(CATEGORIES_CACHE_PATH, "w", encoding="utf-8") as f:
        # Guardamos directamente la lista; main.py lo acepta así.
        json.dump(results, f, ensure_ascii=False, indent=2)

def generate_product_forecasts():
    # 1) Obtener lista de todos los productos
    products = get_all_products()
    results = []

    for prod in products:
        pid  = prod["id"]
        name = prod.get("name", "")

        # 2) Obtener historial de ventas de ese producto
        df = get_and_clean_product_data(pid)
        if df.empty:
            continue

        # 3) Construir “history” para cada fila del DataFrame
        history = [
            {
                "date":  row["date"].strftime("%Y-%m-%d"),
                "value": int(round(row["value"]))
            }
            for _, row in df.iterrows()
        ]

        # 4) Predecir con la función robusta
        #    Primero preparamos df para Prophet:
        df_prod_prepared = prepare_product_df(df)
        df_fore = predict_product_sales(df_prod_prepared, weeks=WEEKS)

        # 5) Verificar inestabilidad: 
        history_values  = [int(round(r["value"])) for r in history]
        forecast_values = df_fore["yhat"].tolist()
        if es_forecast_inestable(history_values, forecast_values):
            # Fallback: promedio de últimas 3 semanas
            avg3 = int(round(np.mean(history_values[-3:])))
            forecasting = [
                {
                    "date": row["ds"].strftime("%Y-%m-%d"),
                    "value": avg3
                }
                for _, row in df_fore.iterrows()
            ]
        else:
            # Normal: redondear cada valor de yhat
            forecasting = [
                {
                    "date": row["ds"].strftime("%Y-%m-%d"),
                    "value": int(round(row["yhat"]))
                }
                for _, row in df_fore.iterrows()
            ]

        results.append({
            "product_id":  pid,
            "name":        name,
            "history":     history,
            "forecasting": forecasting,
            "weeks":       WEEKS
        })

    # 6) Asegurarse de que exista la carpeta “cache”
    os.makedirs(os.path.dirname(PRODUCTS_CACHE_PATH), exist_ok=True)

    # 7) Guardar JSON (lista de productos)
    print(f"[generate_product_forecasts] Guardando {len(results)} productos en '{PRODUCTS_CACHE_PATH}'")
    with open(PRODUCTS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # Crear carpeta “cache” si no existe
    os.makedirs("cache", exist_ok=True)

    generate_category_forecasts()
    generate_product_forecasts()
    print("✅ Forecasts generados y guardados")
