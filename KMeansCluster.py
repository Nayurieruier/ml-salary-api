from matplotlib import pyplot as plt
from seaborn import scatterplot
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
#Question 1
df=pd.read_csv('employees_ml.csv')
X=df[["experience_years","education_years","hours_per_week"]]

scaler = StandardScaler()
scaler.fit(X)

#Question 2

# Initialize imputer (strategy can be 'mean', 'median', or 'most_frequent')
imputer = SimpleImputer(strategy='mean')

# Fit and transform your data matrix
X = imputer.fit_transform(X)
inertia=[]
for k in range(1,8):
    km=KMeans(n_clusters=k,random_state=42)
    km.fit(X)
    inertia.append(km.inertia_)
print(inertia)

#Question 3

kmeans = KMeans(n_clusters=4,random_state=42)

kmeans.fit(X)

#Question 4


#Question 5

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)    # collapses your features into 2 new "components"
scatterplot(X_reduced)
plt.show()