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
    category: str
    weeks: int = 4

@app.get("/")
def root():
    return {"message": "API funcionando correctamente"}

@app.post(
    "/forecast",
    dependencies=[Depends(get_api_key)]  # <- Protección con API Key
)
def get_forecast(request: ForecastRequest):
    # 1. Carga y limpieza de datos
    df = get_and_clean_data()

    # 2. Validación de categoría
    if request.category not in df.columns:
        raise HTTPException(status_code=400, detail=f"Category '{request.category}' not found.")

    # 3. Construir historial con datos originales
    history_list = [
        {
            "date": row["date"].strftime("%Y-%m-%d"),
            "value": int(round(row[request.category]))
        }
        for _, row in df.iterrows()
    ]

    # 4. Preparar DF para Prophet (log-transform)
    df_cat = prepare_category_df(df, request.category)
    # 5. Predicción
    forecast_df = predict_next_weeks(df_cat, request.weeks)

    # 6. Formatear forecast
    forecasting_list = [
        {
            "date": row["ds"].strftime("%Y-%m-%d"),
            "value": int(round(row["yhat"]))
        }
        for _, row in forecast_df.iterrows()
    ]

    return {
        "category": request.category,
        "history": history_list,
        "forecasting": forecasting_list,
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=os.environ.get("ENV") != "production"
    )
