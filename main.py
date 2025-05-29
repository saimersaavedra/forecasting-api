from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from data_utils import get_and_clean_data
from predictor import prepare_category_df, predict_next_weeks
import os

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

@app.post("/forecast")
def get_forecast(request: ForecastRequest):
    df = get_and_clean_data()

    if request.category not in df.columns:
        raise HTTPException(status_code=400, detail=f"Category '{request.category}' not found.")

    # History con valores originales sin transformación
    history_list = [
        {
            "date": row["date"].strftime("%Y-%m-%d"),
            "value": int(round(row[request.category]))
        }
        for _, row in df.iterrows()
    ]

    df_cat = prepare_category_df(df, request.category)
    forecast = predict_next_weeks(df_cat, request.weeks)

    forecasting_list = [
        {
            "date": row["ds"].strftime("%Y-%m-%d"),
            "value": int(round(row["yhat"]))
        }
        for _, row in forecast.iterrows()
    ]

    return {
        "category": request.category,
        "history": history_list,
        "forecasting": forecasting_list,
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
