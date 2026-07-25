import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

  # load (in a different script, e.g. your API)
df=pd.read_csv("employees_ml.csv")
X=[["experience_years", "education_years", "hours_per_week"],]
app = FastAPI()
model = joblib.load("salary_model.joblib")
joblib.dump(model, "salary_model.joblib")   # save
model = joblib.load("salary_model.joblib")
predictions_array=[]
class PredictionInput(BaseModel):
    experience_years: float
    education_years: float
    hours_per_week: float

@app.post("/predict")
def predict(data: PredictionInput):
    X_new = [[data.experience_years, data.education_years, data.hours_per_week]]
    prediction = model.predict(X_new)
    predictions_array.append(prediction)
    return {"predicted_salary": prediction[0]}
