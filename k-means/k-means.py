import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering #HC

veriler = pd.read_csv('./data/medical.csv')

X= veriler.iloc[:, [0,2,3]]

kmeans = KMeans(n_clusters = 3, init = 'k-means++')
kmeans.fit(X)

print(kmeans.cluster_centers_)

sonuclar = []
for i in range(1,10):
  kmeans = KMeans (n_clusters=i, init='k-means++', random_state=123)
  kmeans.fit(X)
  sonuclar.append(kmeans.inertia_)
plt.plot(range(1,10), sonuclar)
plt.show()

#Karşılaştır
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=123)
Y_tahmin = kmeans.fit_predict(X)
print(Y_tahmin)
plt.scatter(X[Y_tahmin==0].iloc[:,0], X[Y_tahmin==0].iloc[:,1], s=100, c='red')
plt.scatter(X[Y_tahmin==1].iloc[:,0], X[Y_tahmin==1].iloc[:,1], s=100, c='blue')
plt.scatter(X[Y_tahmin==2].iloc[:,0], X[Y_tahmin==2].iloc[:,1], s=100, c='green')
plt.title("KMeans")
plt.show()



#HC
ac = AgglomerativeClustering(n_clusters=3, affinity='euclidean',  linkage='ward')
Y_tahmin = ac.fit_predict(X)
print(Y_tahmin)

plt.scatter(X[Y_tahmin==0].iloc[:,0], X[Y_tahmin==0].iloc[:,1], s=100, c='red')
plt.scatter(X[Y_tahmin==1].iloc[:,0], X[Y_tahmin==1].iloc[:,1], s=100, c='blue')
plt.scatter(X[Y_tahmin==2].iloc[:,0], X[Y_tahmin==2].iloc[:,1], s=100, c='green')
plt.title("HC")
plt.show()

# Dendrogram
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.show()