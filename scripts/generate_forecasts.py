# scripts/generate_forecasts.py

import os
import json
import numpy as np
from datetime import datetime
from predictor import (
    predict_next_weeks,
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
    # Obtener y limpiar datos de categorías
    df = get_and_clean_category_data()
    categories = [col for col in df.columns if col != "date"]
    results = []

    for category in categories:
        # Construir “history” para cada fila del DataFrame
        history = [
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "value": int(round(row[category]))
            }
            for _, row in df.iterrows()
        ]

        # Preparar DataFrame para Forecast y predecir las próximas WEEKS semanas
        df_cat = prepare_category_df(df, category)
        forecast_df = predict_next_weeks(df_cat, weeks=WEEKS)

        # Construir lista “forecasting” con ds → yhat
        forecasting = [
            {
                "date": row["ds"].strftime("%Y-%m-%d"),
                "value": int(round(row["yhat"]))
            }
            for _, row in forecast_df.iterrows()
        ]

        results.append({
            "category":   category,
            "history":    history,
            "forecasting": forecasting,
            "weeks":      WEEKS
        })

    # Asegurarse de que exista la carpeta
    os.makedirs(os.path.dirname(CATEGORIES_CACHE_PATH), exist_ok=True)

    # Siempre se escribe, incluso si “results” está vacío (será “[]” en JSON)
    print(f"[generate_category_forecasts] Guardando {len(results)} categorías en '{CATEGORIES_CACHE_PATH}'")
    with open(CATEGORIES_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def generate_product_forecasts():
    # Obtener lista de todos los productos (id, name, etc.)
    products = get_all_products()
    results = []

    for prod in products:
        pid  = prod["id"]
        name = prod.get("name", "")

        df = get_and_clean_product_data(pid)
        # Si no hay datos históricos de ese producto, saltar directamente
        if df.empty:
            continue

        # Construir “history” para cada fila del DataFrame
        history = [
            {
                "date":  row["date"].strftime("%Y-%m-%d"),
                "value": int(round(row["value"]))
            }
            for _, row in df.iterrows()
        ]

        # Preparar DataFrame para Forecast
        df_prod = prepare_product_df(df)
        forecast_df = predict_product_sales(df_prod, weeks=WEEKS)

        # Extraer lista de valores pronosticados e históricos
        forecast_values = forecast_df["yhat"].tolist()
        history_values  = [row["value"] for _, row in df.iterrows()]

        # Si la función de inestabilidad detecta “forecast inestable”:
        if es_forecast_inestable(history_values, forecast_values):
            # Tomar el promedio de las últimas 3 mediciones históricas
            avg3 = int(round(np.mean(history_values[-3:])))
            forecasting = [
                {
                    "date":  row["ds"].strftime("%Y-%m-%d"),
                    "value": avg3
                }
                for _, row in forecast_df.iterrows()
            ]
        else:
            # Normal: redondear cada yhat
            forecasting = [
                {
                    "date":  row["ds"].strftime("%Y-%m-%d"),
                    "value": int(round(row["yhat"]))
                }
                for _, row in forecast_df.iterrows()
            ]

        results.append({
            "product_id": pid,
            "name":       name,
            "history":    history,
            "forecasting": forecasting,
            "weeks":      WEEKS
        })

    # Asegurarse de que exista la carpeta
    os.makedirs(os.path.dirname(PRODUCTS_CACHE_PATH), exist_ok=True)

    # Siempre se escribe, incluso si “results” está vacío (será “[]” en JSON)
    print(f"[generate_product_forecasts] Guardando {len(results)} productos en '{PRODUCTS_CACHE_PATH}'")
    with open(PRODUCTS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    # Crear carpeta “cache” si no existe
    os.makedirs("cache", exist_ok=True)

    generate_category_forecasts()
    generate_product_forecasts()
    print("✅ Forecasts generados y guardados")
