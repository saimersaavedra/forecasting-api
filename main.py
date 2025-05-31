# main.py

import os
import json
from fastapi import FastAPI, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from scripts.generate_forecasts import (
    generate_category_forecasts,
    generate_product_forecasts
)
from pydantic import BaseModel

# --- Modelos Pydantic ---
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


# --- Configuración FastAPI y CORS ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Key (obligatorio) ---
API_KEY = os.environ.get("API_KEY", "")
if not API_KEY:
    raise RuntimeError("La variable de entorno API_KEY no está definida")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(status_code=401, detail="Invalid or missing API Key")


# --- Rutas de cache en disco ---
CATEGORIES_CACHE_PATH = "cache/categories_forecast.json"
PRODUCTS_CACHE_PATH   = "cache/products_forecast.json"


# --- Scheduler para regenerar cache cada lunes a las 03:00 (Bogotá) ---
scheduler = AsyncIOScheduler(timezone="America/Bogota")

@app.on_event("startup")
async def startup_event():
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

def ensure_cache(path: str, generate_fn):
    """
    1) Si el archivo no existe, lo crea y genera forecast.
    2) Si existe pero está vacío (size == 0), vuelve a generar.
    3) Si existe pero no es JSON válido, vuelve a generar.
    Después de cualquiera de estos casos, al final fuerza la generación y lectura.
    """
    # 1) Crear carpeta contenedora si no existe
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # 2) Si no existe o está vacío → generar
    if (not os.path.isfile(path)) or (os.path.getsize(path) == 0):
        print(f"[ensure_cache] '{path}' no existe o está vacío → generando ahora.")
        generate_fn()
        return

    # 3) Si el archivo existe y tiene contenido, verificar que sea JSON válido
    try:
        with open(path, 'r', encoding='utf-8') as f:
            _ = json.load(f)
    except Exception as e:
        print(f"[ensure_cache] '{path}' contiene JSON inválido ({e}) → regenerando.")
        generate_fn()
        return

    # Si llegamos aquí, el archivo ya existía y tenía JSON válido; no es necesario regenerar.


def load_cache(path: str, generate_fn) -> list:
    """
    Se asegura de que exista un JSON válido en `path`.
    Si no existía o estaba corrupto/ vacío, llama a generate_fn() para
    crearlo. Luego lee y devuelve la lista resultante.
    """
    ensure_cache(path, generate_fn)

    # Leer por segunda vez, ahora que estamos seguros de que hay JSON válido
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[load_cache] ERROR leyendo '{path}' incluso después de generar: {e}")
        return []

    # Si el JSON está envuelto en {"forecasts": [...]}, desempaquetar
    if isinstance(data, dict) and 'forecasts' in data:
        return data['forecasts']

    # Si es una lista directamente, devolverla
    if isinstance(data, list):
        return data

    # Si no es ni dict ni list, devolvemos lista vacía
    print(f"[load_cache] '{path}' no contiene ni una lista ni diccionario con 'forecasts'.")
    return []


# --- Endpoints GET cacheados ---
@app.get("/cached/forecast/category", dependencies=[Depends(get_api_key)])
def get_cached_categories():
    cache = load_cache(CATEGORIES_CACHE_PATH, generate_category_forecasts)
    return {"forecasts": cache}


@app.get("/cached/forecast/product", dependencies=[Depends(get_api_key)])
def get_cached_products():
    cache = load_cache(PRODUCTS_CACHE_PATH, generate_product_forecasts)
    return {"forecasts": cache}


# --- Endpoints POST que usan el cache ---
@app.post(
    "/forecast/category",
    response_model=ForecastCategoryResponse,
    dependencies=[Depends(get_api_key)]
)
def post_forecast_category(req: ForecastCategoryRequest = Body(...)):
    cache = load_cache(CATEGORIES_CACHE_PATH, generate_category_forecasts)
    for item in cache:
        if item.get('category') == req.category and item.get('weeks') == req.weeks:
            history    = [Point(**p) for p in item['history']]
            forecasting = [Point(**p) for p in item['forecasting']]
            return ForecastCategoryResponse(
                category=req.category,
                history=history,
                forecasting=forecasting,
                weeks=req.weeks
            )
    raise HTTPException(
        status_code=404,
        detail=f"Category '{req.category}' con weeks={req.weeks} no encontrada en cache."
    )


@app.post(
    "/forecast/product",
    response_model=ForecastProductResponse,
    dependencies=[Depends(get_api_key)]
)
def post_forecast_product(req: ForecastProductRequest = Body(...)):
    cache = load_cache(PRODUCTS_CACHE_PATH, generate_product_forecasts)
    for item in cache:
        if item.get('product_id') == req.product_id and item.get('weeks') == req.weeks:
            history    = [Point(**p) for p in item['history']]
            forecasting = [Point(**p) for p in item['forecasting']]
            return ForecastProductResponse(
                product_id=req.product_id,
                history=history,
                forecasting=forecasting,
                weeks=req.weeks
            )
    raise HTTPException(
        status_code=404,
        detail=f"Product '{req.product_id}' con weeks={req.weeks} no encontrado en cache."
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENV") != "production"
    )
