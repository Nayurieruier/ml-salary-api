import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier

df = pd.read_csv("employees_ml.csv")
df["education_years"] = df["education_years"].fillna(df["education_years"].median())
df["hours_per_week"] = df["hours_per_week"].fillna(df["hours_per_week"].median())

X = df[["experience_years", "education_years", "hours_per_week"]]
y = df["performance"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.3, random_state=42)
model.fit(X_train, y_train)              # ← the missing step
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("Confusion matrix:\n", confusion_matrix(y_test, predictions))
print("Feature importance's:", dict(zip(X.columns, model.feature_importances_)))
cr=cross_val_score(model, X, y_encoded, cv=5)


for i in range(3):
    for j in range(len(cr)):
        print(cr[j], end=" ")


def nums():
    return list(range(10))
i=nums()
print(i)