import os
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from data_utils import get_and_clean_data
from predictor import prepare_category_df, predict_next_weeks
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

class ForecastRequest(BaseModel):
    category_id: int

@app.get("/categories")
def list_categories():
    """
    Devuelve la lista de categorías disponibles con su ID para que el cliente sepa qué enviar.
    """
    df = get_and_clean_data()
    cats = [col for col in df.columns if col != "date"]
    return [{"id": idx, "name": cat} for idx, cat in enumerate(cats)]

@app.get("/")
def root():
    return {"message": "API funcionando correctamente"}

@app.post(
    "/forecast",
    dependencies=[Depends(get_api_key)]
)
def get_forecast(request: ForecastRequest):
    # 1. Carga y limpieza de datos
    df = get_and_clean_data()

    # 2. Extraer lista de categorías y validar ID
    cats = [col for col in df.columns if col != "date"]
    if request.category_id < 0 or request.category_id >= len(cats):
        raise HTTPException(
            status_code=400,
            detail=f"category_id '{request.category_id}' inválido. Debe estar entre 0 y {len(cats)-1}."
        )
    category_name = cats[request.category_id]

    # 3. Construir historial con datos originales
    history_list = [
        {
            "date": row["date"].strftime("%Y-%m-%d"),
            "value": int(round(row[category_name]))
        }
        for _, row in df.iterrows()
    ]

    # 4. Preparar DF para Prophet (log-transform)
    df_cat = prepare_category_df(df, category_name)
    # 5. Predicción para 4 semanas fijas
    forecast_df = predict_next_weeks(df_cat, weeks=4)

    # 6. Formatear forecast
    forecasting_list = [
        {
            "date": row["ds"].strftime("%Y-%m-%d"),
            "value": int(round(row["yhat"]))
        }
        for _, row in forecast_df.iterrows()
    ]

    return {
        "category_id": request.category_id,
        "category_name": category_name,
        "history": history_list,
        "forecasting": forecasting_list,
        "weeks": 4
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=os.environ.get("ENV") != "production"
    )
