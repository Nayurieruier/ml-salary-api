import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
df = pd.read_csv('employees_ml.csv')
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
#Question 1
# Since there was no mention of any scikit learn libraries we are going to use the pandas to fix missing values

df["education_years"] = df["education_years"].fillna(df["education_years"].median())
df["hours_per_week"] = df["hours_per_week"].fillna(df["hours_per_week"].median())
#print(df.isnull().sum())


X=df[["experience_years", "education_years","hours_per_week" ]]
y=df["salary"]


#Question 2



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Question 3

model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
# Regression
from sklearn.linear_model import LinearRegression


# Classification


mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
#print("MSE:", mse)
#print("R²:", r2)

#Question 6

#Question Fixes


X_2 = df[["experience_years", "education_years", "hours_per_week"]]
y_2 = df["performance"]

X_2_train, X_2_test, y_2_train, y_2_test = train_test_split(
    X_2, y_2, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_2_train, y_2_train)
predictions = model.predict(X_2_test)

acc = accuracy_score(y_2_test, predictions)
cm = confusion_matrix(y_2_test, predictions)
print("Accuracy:", acc)
print("Confusion matrix:\n", cm)






