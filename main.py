# data_utils.py
import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

# Cargar variables de entorno si no estás en producción
if os.getenv("ENV") != "production":
    load_dotenv()

API_URL = os.getenv("API_URL")


def get_and_clean_category_data():
    """
    Fetch y limpia datos de ventas por categoría.
    """
    endpoint = f"{API_URL.rstrip('/')}/category/sales-category"
    resp = requests.get(endpoint)
    resp.raise_for_status()

    raw = resp.json()
    df = pd.DataFrame(raw)
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


# predictor.py
from prophet import Prophet
import pandas as pd
import numpy as np

def prepare_category_df(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """
    Prepara df para Prophet:
     - renombra date→ds y category→y
     - asegura valores ≥0 y aplica log1p
    """
    df_cat = df[['date', category]].copy()
    df_cat = df_cat.rename(columns={'date': 'ds', category: 'y'})
    df_cat['y'] = df_cat['y'].clip(lower=0)
    df_cat['y'] = np.log1p(df_cat['y'])
    return df_cat


def prepare_product_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara df para Prophet para producto:
     - renombra date→ds y value→y
     - asegura valores ≥0 y aplica log1p
    """
    df_prod = df[['date', 'value']].copy()
    df_prod = df_prod.rename(columns={'date': 'ds', 'value': 'y'})
    df_prod['y'] = df_prod['y'].clip(lower=0)
    df_prod['y'] = np.log1p(df_prod['y'])
    return df_prod


def predict_next_weeks(df: pd.DataFrame, weeks: int = 4) -> pd.DataFrame:
    """
    Entrena Prophet sin estacionalidades y con tendencia suavizada,
    genera forecast y revierte log1p→expm1, clip y round.
    """
    model = Prophet(
        weekly_seasonality=False,
        daily_seasonality=False,
        yearly_seasonality=False,
        changepoint_prior_scale=0.1,
        seasonality_mode='additive'
    )
    model.fit(df)

    future = model.make_future_dataframe(periods=weeks, freq='W')
    forecast = model.predict(future)

    # Revertir transformación y limpiar resultado
    yhat = np.expm1(forecast['yhat'].clip(lower=0))
    forecast['yhat'] = yhat.clip(lower=0).round().astype(int)

    return forecast[['ds', 'yhat']].tail(weeks)


# main.py
import os
from fastapi import FastAPI, HTTPException, Depends, Security, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from data_utils import get_and_clean_category_data, get_and_clean_product_data
from predictor import (
    prepare_category_df,
    prepare_product_df,
    predict_next_weeks
)
import uvicorn

# Carga la API Key desde las variables de entorno
API_KEY = os.environ.get("API_KEY", "")
if not API_KEY:
    raise RuntimeError("La variable de entorno API_KEY no está definida")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(status_code=401, detail="Invalid or missing API Key")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Modelos de respuesta ---
class Point(BaseModel):
    date: str
    value: int

class ForecastCategoryResponse(BaseModel):
    category: str
    history: list[Point]
    forecasting: list[Point]
    weeks: int

class ForecastProductResponse(BaseModel):
    product_id: str
    history: list[Point]
    forecasting: list[Point]
    weeks: int

@app.get("/", response_model=dict)
def root():
    return {"message": "API funcionando correctamente"}

@app.post(
    "/forecast/category/{category}",
    response_model=ForecastCategoryResponse,
    dependencies=[Depends(get_api_key)]
)
def get_forecast(
    category: str = Path(..., description="Nombre de la categoría a predecir"),
    weeks: int = 4
):
    df = get_and_clean_category_data()
    if category not in df.columns:
        raise HTTPException(status_code=404, detail=f"Category '{category}' no encontrada.")
    history = [
        Point(date=row["date"].strftime("%Y-%m-%d"), value=int(round(row[category])))
        for _, row in df.iterrows()
    ]
    df_cat = prepare_category_df(df, category)
    forecast_df = predict_next_weeks(df_cat, weeks=weeks)
    forecasting = [
        Point(date=row["ds"].strftime("%Y-%m-%d"), value=int(round(row["yhat"])))
        for _, row in forecast_df.iterrows()
    ]
    return ForecastCategoryResponse(
        category=category,
        history=history,
        forecasting=forecasting,
        weeks=weeks
    )

@app.get(
    "/forecast/category/all",
    response_model=dict,
    dependencies=[Depends(get_api_key)]
)
def forecast_all_categories(weeks: int = 4):
    df = get_and_clean_category_data()
    cats = [col for col in df.columns if col != "date"]
    results = []
    for cat in cats:
        history = [
            Point(date=row["date"].strftime("%Y-%m-%d"), value=int(round(row[cat])))
            for _, row in df.iterrows()
        ]
        df_cat = prepare_category_df(df, cat)
        forecast_df = predict_next_weeks(df_cat, weeks=weeks)
        forecasting = [
            Point(date=row["ds"].strftime("%Y-%m-%d"), value=int(round(row["yhat"])))
            for _, row in forecast_df.iterrows()
        ]
        results.append({"category": cat, "history": history, "forecasting": forecasting, "weeks": weeks})
    return {"forecasts": results}

@app.get(
    "/product/forecast/{product_id}",
    response_model=ForecastProductResponse,
    dependencies=[Depends(get_api_key)]
)
def forecast_product(
    product_id: str = Path(..., description="ID del producto para predicción"),
    weeks: int = 4
):
    df = get_and_clean_product_data(product_id)
    if df.empty:
        raise HTTPException(status_code=404, detail="No se encontraron ventas para este producto")
    history = [
        Point(date=row["date"].strftime("%Y-%m-%d"), value=int(round(row["value"])))
        for _, row in df.iterrows()
    ]
    df_prod = prepare_product_df(df)
    forecast_df = predict_next_weeks(df_prod, weeks=weeks)
    forecasting = [
        Point(date=row["ds"].strftime("%Y-%m-%d"), value=int(round(row["yhat"])))
        for _, row in forecast_df.iterrows()
    ]
    return ForecastProductResponse(
        product_id=product_id,
        history=history,
        forecasting=forecasting,
        weeks=weeks
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=os.environ.get("ENV") != "production"
    )