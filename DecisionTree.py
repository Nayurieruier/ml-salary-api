from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
#df = pd.read_csv('employees_ml.csv')
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


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
#print("Accuracy:", acc)
#print("Confusion matrix:\n", cm)





clf=DecisionTreeClassifier(max_depth=4,random_state=42)
clf.fit(X_2_train, y_2_train)
predictions = clf.predict(X_2_test)
cm_2 = confusion_matrix(y_2_test, predictions)
accuracy = accuracy_score(y_2_test, predictions)
#print("Accuracy:", accuracy)
#print("Confusion matrix:\n", cm_2)
#Better accuracy and yes higher values

#3. Max_depth=2 has the better accuracy, it does worse at max_depth=none

#4
clf_2=RandomForestClassifier(n_estimators=100,max_depth=4,random_state=42)
clf_2.fit(X_2_train, y_2_train)
clf_2.predict(X_2_test)
cm_3 = confusion_matrix(y_2_test, clf_2.predict(X_2_test))
acc_3 = accuracy_score(y_2_test, clf_2.predict(X_2_test))
#print("Accuracy for Random forest:", acc_3)
#print("Confusion matrix:\n", cm_3)
#print(clf_2.feature_importances_)

scores=cross_val_score(model,X_2,y_2, cv=5)

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

param_grid = {
    "max_depth": [2, 4, 6, None],
    "min_samples_split": [2, 5, 10]
}

grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
grid.fit(X_2_train, y_2_train)
print(grid.best_params_)
print(grid.best_score_)
grid.predict(X_2_test)
print(grid.score(X_2_test, y_2_test))

param_grid_2={
    "n_estimators": [50,100,200],
    "max_depth": [2, 4, 6],
}
grid_2=GridSearchCV(RandomForestClassifier(random_state=42), param_grid_2, cv=5)
grid_2.fit(X_2_train, y_2_train)
print(grid_2.best_params_)
print(grid_2.best_score_)