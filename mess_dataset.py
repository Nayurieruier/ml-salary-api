
import pandas as pd
df = pd.read_csv('messy_orders.csv')

#Question 1

#print(df.info())
#print(df.isnull().sum())
#print(df.duplicated())

#Question 2

df['customer'] = df['customer'].str.title()
df['region']=df['region'].str.title()

#Question 3

df['price']=df['price'].astype(float)

#Question 4

df.drop_duplicates(subset='order_id', keep='first', inplace=True)
#print(df)

#Question 5

#customer with no name is deleted from question 4

df["product"]=df["product"].fillna("none")

df["quantity"]=df["quantity"].fillna(0)

#Question 6
# negative quantity is a sign error — fix it
df["quantity"] = df["quantity"].abs()

# quantity = 100 — investigate before deciding
d=df[df["quantity"] == 100]
print(d)

#Question 7

df["order_date"] = pd.to_datetime(df["order_date"], format='mixed')

#Question 8
df["total_price"]=df["price"]*df["quantity"]
print(df.info())