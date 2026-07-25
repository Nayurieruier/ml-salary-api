# Salary Prediction API

A REST API that predicts employee salary based on years of experience, education level, and hours worked per week.

## Stack
- scikit-learn (LinearRegression)
- FastAPI + Pydantic
- Docker
- joblib

## Project structure
- `save_model.py` — trains the LinearRegression model and saves it with joblib
- `FastAPi.py` — FastAPI app that loads the saved model and serves predictions
- `Dockerfile` — containerizes the API
- `requirements.txt` — minimal dependencies for deployment

## Running locally

### Without Docker
```bash
pip install fastapi uvicorn scikit-learn joblib pandas
python save_model.py
uvicorn FastAPi:app --reload
```

### With Docker
```bash
docker build -t salary-api .
docker run -p 8000:8000 salary-api
```

Then visit http://127.0.0.1:8000/docs to try the API interactively.

## Example request
```json
POST /predict
{
  "experience_years": 5,
  "education_years": 16,
  "hours_per_week": 40
}
```

## Example response
```json
{
  "predicted_salary": 72450.83
}
```