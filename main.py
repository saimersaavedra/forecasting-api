from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from data_utils import get_and_clean_data
from predictor import prepare_category_df, predict_next_weeks

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body schema
class ForecastRequest(BaseModel):
    category: str
    weeks: int = 4

@app.post("/forecast")
def get_forecast(request: ForecastRequest):
    # 1. Load data
    df = get_and_clean_data()

    # 2. Validate category
    if request.category not in df.columns:
        raise HTTPException(status_code=400, detail=f"Category '{request.category}' not found.")

    # 3. Prepare category data
    df_cat = prepare_category_df(df, request.category)

    # 4. Predict future weeks
    forecast = predict_next_weeks(df_cat, request.weeks)

    # 5. Format response
    return {
        "category": request.category,
        "history": [
            {
                "date": row["ds"].strftime("%Y-%m-%d"),
                "value": int(round(float(row["y"])))
            }
            for _, row in df_cat.iterrows()
        ],
        "forecasting": [
            {
                "date": row["ds"].strftime("%Y-%m-%d"),
                "value": int(round(float(row["yhat"])))
            }
            for _, row in forecast.iterrows()
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
