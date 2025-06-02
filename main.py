# main.py

import os
import json
from fastapi import FastAPI, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from pydantic import BaseModel

from scripts.generate_forecasts import (
    generate_category_forecasts,
    generate_product_forecasts
)

class Point(BaseModel):
    date: str
    value: int

class ForecastCategoryRequest(BaseModel):
    category: str
    weeks: int = 4

class ForecastCategoryResponse(BaseModel):
    category: str
    history: list[Point]
    forecasting: list[Point]
    weeks: int

class ForecastProductRequest(BaseModel):
    product_id: str
    weeks: int = 4

class ForecastProductResponse(BaseModel):
    product_id: str
    history: list[Point]
    forecasting: list[Point]
    weeks: int

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.environ.get("API_KEY", "")
if not API_KEY:
    raise RuntimeError("La variable de entorno API_KEY no está definida")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(status_code=401, detail="Invalid or missing API Key")

CATEGORIES_CACHE_PATH = "cache/categories_forecast.json"
PRODUCTS_CACHE_PATH   = "cache/products_forecast.json"

scheduler = AsyncIOScheduler(timezone="America/Bogota")

@app.on_event("startup")
async def startup_event():
    try:
        if not scheduler.running:
            trigger = CronTrigger(
                day_of_week="mon",
                hour=0,
                minute=0,
                timezone="America/Bogota"
            )
            scheduler.add_job(generate_category_forecasts, trigger, id="cat_job", replace_existing=True)
            scheduler.add_job(generate_product_forecasts,  trigger, id="prod_job", replace_existing=True)
            scheduler.start()
    except Exception as e:
        print(f"[startup_event] Error al iniciar el scheduler: {e}")

def ensure_cache(path: str, generate_fn):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if (not os.path.isfile(path)) or (os.path.getsize(path) == 0):
            print(f"[ensure_cache] '{path}' no existe o está vacío → generando ahora.")
            generate_fn()
            return

        try:
            with open(path, 'r', encoding='utf-8') as f:
                _ = json.load(f)
        except Exception as e:
            print(f"[ensure_cache] '{path}' contiene JSON inválido ({e}) → regenerando.")
            generate_fn()
            return
    except Exception as e:
        print(f"[ensure_cache] Error general: {e}")

def load_cache(path: str, generate_fn) -> list:
    try:
        ensure_cache(path, generate_fn)

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"[load_cache] ERROR leyendo '{path}' incluso después de generar: {e}")
            return []

        if isinstance(data, dict) and 'forecasts' in data:
            return data['forecasts']
        if isinstance(data, list):
            return data

        print(f"[load_cache] '{path}' no contiene ni lista ni 'forecasts'.")
        return []
    except Exception as e:
        print(f"[load_cache] Error inesperado: {e}")
        return []

@app.get("/cached/forecast/category", dependencies=[Depends(get_api_key)])
def get_cached_categories():
    try:
        cache = load_cache(CATEGORIES_CACHE_PATH, generate_category_forecasts)
        return {"forecasts": cache}
    except Exception as e:
        print(f"[get_cached_categories] Error: {e}")
        raise HTTPException(status_code=500, detail="Error interno al obtener categorías en cache")

@app.get("/cached/forecast/product", dependencies=[Depends(get_api_key)])
def get_cached_products():
    try:
        cache = load_cache(PRODUCTS_CACHE_PATH, generate_product_forecasts)
        return {"forecasts": cache}
    except Exception as e:
        print(f"[get_cached_products] Error: {e}")
        raise HTTPException(status_code=500, detail="Error interno al obtener productos en cache")

@app.post(
    "/forecast/category",
    response_model=ForecastCategoryResponse,
    dependencies=[Depends(get_api_key)]
)
def post_forecast_category(req: ForecastCategoryRequest = Body(...)):
    try:
        cache = load_cache(CATEGORIES_CACHE_PATH, generate_category_forecasts)
        for item in cache:
            if item.get('category') == req.category and item.get('weeks') == req.weeks:
                history     = [Point(**p) for p in item['history']]
                forecasting = [Point(**p) for p in item['forecasting']]
                return ForecastCategoryResponse(
                    category    = req.category,
                    history     = history,
                    forecasting = forecasting,
                    weeks       = req.weeks
                )
        raise HTTPException(
            status_code=404,
            detail=f"Category '{req.category}' con weeks={req.weeks} no encontrada en cache."
        )
    except Exception as e:
        print(f"[post_forecast_category] Error: {e}")
        raise HTTPException(status_code=500, detail="Error interno al procesar categoría")

@app.post(
    "/forecast/product",
    response_model=ForecastProductResponse,
    dependencies=[Depends(get_api_key)]
)
def post_forecast_product(req: ForecastProductRequest = Body(...)):
    try:
        cache = load_cache(PRODUCTS_CACHE_PATH, generate_product_forecasts)
        for item in cache:
            if item.get('product_id') == req.product_id and item.get('weeks') == req.weeks:
                history     = [Point(**p) for p in item['history']]
                forecasting = [Point(**p) for p in item['forecasting']]
                return ForecastProductResponse(
                    product_id  = req.product_id,
                    history     = history,
                    forecasting = forecasting,
                    weeks       = req.weeks
                )
        raise HTTPException(
            status_code=404,
            detail=f"Product '{req.product_id}' con weeks={req.weeks} no encontrada en cache."
        )
    except Exception as e:
        print(f"[post_forecast_product] Error: {e}")
        raise HTTPException(status_code=500, detail="Error interno al procesar producto")

@app.get("/", tags=["Root"])
def read_root():
    return {
        "message": "🟢 API de forecasting disponible.",
        "endpoints": [
            "/cached/forecast/category",
            "/cached/forecast/product",
            "/forecast/category",
            "/forecast/product"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=int(os.getenv("PORT", 8000)),
            reload=os.getenv("ENV") != "production"
        )
    except Exception as e:
        print(f"[main] Error al iniciar el servidor: {e}")