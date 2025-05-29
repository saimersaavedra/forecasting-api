import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from data_utils import get_and_clean_data
from predictor import prepare_category_df, predict_next_weeks

app = FastAPI()

# CORS abierto para cualquier cliente
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schema para el body de /forecast
class ForecastRequest(BaseModel):
    category: str
    weeks: int = 4

# Ruta raíz para comprobar que la API está viva
@app.get("/")
def root():
    return {"message": "API funcionando correctamente"}

# Endpoint principal
@app.post("/forecast")
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

# Arranque de Uvicorn
if __name__ == "__main__":
    import uvicorn

    # Puerto asignado por la plataforma (p.ej. Render, Heroku)
    port = int(os.environ.get("PORT", 8000))
    # En producción NO usar reload=True
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=os.environ.get("ENV") != "production")
