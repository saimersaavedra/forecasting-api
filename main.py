import os
from fastapi import FastAPI, HTTPException, Depends, Security, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
import numpy as np
from pydantic import BaseModel
from data_utils import get_and_clean_category_data, get_and_clean_product_data, get_all_products
from predictor import (
    prepare_category_df,
    prepare_product_df,
    predict_next_weeks,
    predict_product_sales,
    es_forecast_inestable
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

class ForecastCategoryRequest(BaseModel):
    category: str
    weeks: int = 4  # Valor por defecto

class ForecastProductResponse(BaseModel):
    product_id: str
    history: list[Point]
    forecasting: list[Point]
    weeks: int = 4

class ForecastProductRequest(BaseModel):
    product_id: str
    weeks: int = 4

# --- Rutas de la API ---
@app.post(
    "/forecast/category",
    response_model=ForecastCategoryResponse,
    dependencies=[Depends(get_api_key)]
)
def get_forecast(body: ForecastCategoryRequest):
    category = body.category
    weeks = body.weeks

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
    "/forecast/category",
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

from fastapi import Body

@app.post(
    "/forecast/product",
    response_model=ForecastProductResponse,
    dependencies=[Depends(get_api_key)]
)
def forecast_product(request: ForecastProductRequest = Body(...)):
    product_id = request.product_id
    weeks = request.weeks

    df = get_and_clean_product_data(product_id)
    if df.empty:
        raise HTTPException(status_code=404, detail="No se encontraron ventas para este producto")
    history = [
        Point(date=row["date"].strftime("%Y-%m-%d"), value=int(round(row["value"])))
        for _, row in df.iterrows()
    ]
    df_prod = prepare_product_df(df)
    forecast_df = predict_product_sales(df_prod, weeks=weeks)
    forecast_values = forecast_df["yhat"].tolist()
    history_values = [row["value"] for _, row in df.iterrows()]

    if es_forecast_inestable(history_values, forecast_values):
        promedio = int(round(np.mean(history_values[-3:])))
        forecasting = [
            Point(date=row["ds"].strftime("%Y-%m-%d"), value=promedio)
            for _, row in forecast_df.iterrows()
        ]
    else:
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

@app.get(
    "/forecast/product",
    response_model=dict,
    dependencies=[Depends(get_api_key)]
)
def forecast_all_products(weeks: int = 4):
    """
    Genera pronósticos para todos los productos (id+name)
    devolviendo sólo los que tienen historial.
    """
    products = get_all_products()
    results = []

    for prod in products:
        pid = prod["id"]
        name = prod["name"]
        df = get_and_clean_product_data(pid)
        if df.empty:
            continue

        # Construir historial
        history = [
            Point(date=row["date"].strftime("%Y-%m-%d"), value=int(row["value"]))
            for _, row in df.iterrows()
        ]

        # Forecast con lógica existente
        df_prod = prepare_product_df(df)
        fc_df = predict_product_sales(df_prod, weeks)
        yhat_vals = fc_df["yhat"].tolist()
        hist_vals = [pt.value for pt in history]

        if es_forecast_inestable(hist_vals, yhat_vals):
            avg3 = int(round(np.mean(hist_vals[-3:])))
            forecasting = [
                Point(date=r["ds"].strftime("%Y-%m-%d"), value=avg3)
                for _, r in fc_df.iterrows()
            ]
        else:
            forecasting = [
                Point(date=r["ds"].strftime("%Y-%m-%d"), value=int(r["yhat"]))
                for _, r in fc_df.iterrows()
            ]

        results.append({
            "product_id": pid,
            "name": name,
            "history": history,
            "forecasting": forecasting,
            "weeks": weeks
        })

    return {"forecasts": results}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=os.environ.get("ENV") != "production"
    )