from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="AI Prediction API", description="Endpoints for Attrition and Fraud Detection Models")

# --- Load Models ---
try:
    attrition_model = joblib.load("attrition_model.pkl")
    print("✅ Attrition model loaded successfully.")
except Exception as e:
    print(f"⚠️ Failed to load attrition_model.pkl: {e}")




# ==========================================================
# 🔹 ATTRITION PREDICTION API
# ==========================================================

class AttritionInput(BaseModel):
    Age: int
    MonthlyIncome: float
    OverTime: int  # 0 = No, 1 = Yes


@app.post("/predict_attrition/")
def predict_attrition(data: AttritionInput):
    try:
        input_df = pd.DataFrame([data.dict()])
        prediction = attrition_model.predict(input_df)[0]
        return {"Attrition_Prediction": int(prediction)}
    except Exception as e:
        return {"detail": f"Attrition prediction failed: {str(e)}"}


