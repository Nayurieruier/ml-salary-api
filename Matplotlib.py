import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#Barplot
df = pd.read_csv('clean_orders.csv')
#sns.barplot(data=df, x="region", y="total_price",estimator="sum")

#plt.show()

#Histomgram

#sns.histplot(data=df, x="region", y="total_price")
#plt.show()

#Scatter Plot

#sns.scatterplot(data=df, x="quantity", y="total_price")
#plt.show()

#sns.boxplot(x="total_price", y="region", data=df)
#plt.show()

#df["product"].value_counts()
#sns.countplot(x="product", data=df)
#plt.show()

sns.heatmap(df["price","quantity","total_price"])
plt.show()