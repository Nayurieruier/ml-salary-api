
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = {
    'name': ['Alice','Bob','Carol','Dave','Eve','Frank','Grace','Heidi','Ivan','Judy'],
    'department': ['Sales','Eng','Sales','Eng','Marketing','Eng','Marketing','Sales','Eng','Marketing'],
    'age': [25, 32, 29, np.nan, 41, 38, 27, 33, np.nan, 45],
    'salary': [52000, 85000, 58000, 91000, 62000, 88000, 59000, 61000, 95000, 67000],
    'years_at_company': [2, 5, 3, 7, 4, 6, 1, 3, 8, 9]
}
df = pd.DataFrame(data)
df["age"]=df["age"].fillna().median()
#Question 1

#print(df.head())
#print(df.info())
#print(df.describe())
#print(df.shape)
#print(df.isnull().sum())

#Question 2

filter_df=df.loc[df["department"] == "Eng","name"]
#print(filter_df)

#Question 3

#print(df.isna().sum())
fil=df["age"].fillna(df["age"].median())
#print(fil)

#Question 4

avg=df.groupby("department").agg(
    avg=("salary", "mean")
)

#Question 5

avg_2=df.groupby("department").agg(
    avg=("salary", "mean"),
    count=("name", "count")
)

# Question 6

#df["salary_per_year"]=df["salary"]/(df["years_at_company"]+1)
#print(df)
#Question 7
#top3=df.nlargest(3,"salary")[["salary","name"]]
#print(top3)

#Question 8

# Syntax: df.loc[row_condition, column_selection]
result = df.loc[df['years_at_company'] >= 3, ['name', 'department','salary']].groupby(

    'department').agg(
    avg=("salary", "mean"),
    count=("name", "count")
)

#print(df.columns.tolist())

#print(result)


dept_info = {
    'department': ['Sales', 'Eng', 'Marketing', 'HR'],
    'manager': ['Linda', 'Sam', 'Priya', 'Tom'],
    'budget': [200000, 500000, 150000, 90000]
}
dept_df = pd.DataFrame(dept_info)

#Question 9

merged_df = pd.merge(df, dept_df, on='department', how='inner')
#print(merged_df)

#Question 10
merged_df_left = pd.merge(df, dept_df, on='department', how='left')
#Shows the left rows and the rows at the right that matches and shows null at the left column at the rows which don't match
#print(merged_df_left)
merged_df_outer = pd.merge(df, dept_df, on='department', how='outer')
#Shows records that match either left or right and if there is no match it fills in the data with null
#print(merged_df_outer)


#Question 11

df["salary_per_year"]=df["salary"]/(df["years_at_company"]+1)

w=df.pivot_table(values="salary", index="department", aggfunc=["mean", "max"])
#print(w)

#Question 12
question_12 = df[['name', 'salary', 'years_at_company']].copy()
new = question_12.melt(id_vars=['name'], value_vars=['salary', 'years_at_company'])
print()
pn=["ksn"]