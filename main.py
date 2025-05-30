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

# --- API Key ---
API_KEY = os.environ.get("API_KEY", "")
if not API_KEY:
    raise RuntimeError("La variable de entorno API_KEY no está definida")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(status_code=401, detail="Invalid or missing API Key")

# --- Scheduler ---
scheduler = AsyncIOScheduler(timezone="America/Bogota")
@app.on_event("startup")
async def startup_event():
    if not scheduler.running:
        trigger = CronTrigger(day_of_week="mon", hour=3, minute=0)
        scheduler.add_job(generate_category_forecasts, trigger, id="cat_job", replace_existing=True)
        scheduler.add_job(generate_product_forecasts, trigger, id="prod_job", replace_existing=True)
        scheduler.start()

# --- Helpers para cache ---
def load_cache(path: str) -> list:
    # Generar cache si no existe o está vacío/corrupto
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if 'category' in path:
            generate_category_forecasts()
        else:
            generate_product_forecasts()
    content = open(path, 'r', encoding='utf-8').read().strip()
    if not content:
        if 'category' in path:
            generate_category_forecasts()
        else:
            generate_product_forecasts()
        content = open(path, 'r', encoding='utf-8').read().strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        if 'category' in path:
            generate_category_forecasts()
        else:
            generate_product_forecasts()
        data = json.loads(open(path, 'r', encoding='utf-8').read().strip())
    # Si dict con clave 'forecasts', desempaquetar
    if isinstance(data, dict) and 'forecasts' in data:
        return data['forecasts']
    return data

# --- Endpoints GET cacheados ---
@app.get("/cached/forecast/category", dependencies=[Depends(get_api_key)])
def get_cached_categories():
    cache = load_cache("cache/categories_forecast.json")
    return {"forecasts": cache}

@app.get("/cached/forecast/product", dependencies=[Depends(get_api_key)])
def get_cached_products():
    cache = load_cache("cache/products_forecast.json")
    return {"forecasts": cache}

# --- Endpoints POST que usan el cache ---
@app.post(
    "/forecast/category",
    response_model=ForecastCategoryResponse,
    dependencies=[Depends(get_api_key)]
)
def post_forecast_category(req: ForecastCategoryRequest = Body(...)):
    cache = load_cache("cache/categories_forecast.json")
    for item in cache:
        if item.get('category') == req.category and item.get('weeks') == req.weeks:
            history = [Point(**p) for p in item['history']]
            forecasting = [Point(**p) for p in item['forecasting']]
            return ForecastCategoryResponse(
                category=req.category,
                history=history,
                forecasting=forecasting,
                weeks=req.weeks
            )
    raise HTTPException(status_code=404, detail=f"Category '{req.category}' con weeks={req.weeks} no encontrada en cache.")

@app.post(
    "/forecast/product",
    response_model=ForecastProductResponse,
    dependencies=[Depends(get_api_key)]
)
def post_forecast_product(req: ForecastProductRequest = Body(...)):
    cache = load_cache("cache/products_forecast.json")
    for item in cache:
        if item.get('product_id') == req.product_id and item.get('weeks') == req.weeks:
            history = [Point(**p) for p in item['history']]
            forecasting = [Point(**p) for p in item['forecasting']]
            return ForecastProductResponse(
                product_id=req.product_id,
                history=history,
                forecasting=forecasting,
                weeks=req.weeks
            )
    raise HTTPException(status_code=404, detail=f"Product '{req.product_id}' con weeks={req.weeks} no encontrado en cache.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=os.getenv("ENV")!="production")
