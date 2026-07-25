import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
#Question 1

df=pd.read_csv("employees_ml.csv")
#print(pd.get_dummies(df["department"]))

#Question 2

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), ["experience_years", "education_years", "hours_per_week"]),
    ("cat", OneHotEncoder(), ["department"])
])

#Question 3
X=df[["experience_years", "education_years","hours_per_week","department" ]]

y=df["performance"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression())
])

pipe.fit(X_train, y_train)        # does preprocessing AND fits the model, in order
pipe.predict(X_test)                # applies the SAME preprocessing, then predicts


#Question 4

