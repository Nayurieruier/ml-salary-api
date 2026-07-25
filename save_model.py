# save_model.py — run this FIRST, on its own, just once
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv("employees_ml.csv")
df["education_years"] = df["education_years"].fillna(df["education_years"].median())
df["hours_per_week"] = df["hours_per_week"].fillna(df["hours_per_week"].median())

X = df[["experience_years", "education_years", "hours_per_week"]]
y = df["salary"]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "salary_model.joblib")
print("Model saved!")