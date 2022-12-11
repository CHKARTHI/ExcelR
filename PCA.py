# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 19:58:12 2022

@author: DELL
"""


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

a=pd.read_csv("wine.csv")
print(a.describe())
a.head()
a['Type'].value_counts()
Wine= a.iloc[:,1:]
Wine
Wine.shape
Wine.info()
# Converting data to numpy array
wine_ar=Wine.values
wine_ar
# Normalizing the  numerical data
wine_norm=scale(wine_ar)
wine_norm

# Applying PCA Fit Transform to dataset
pca = PCA()
pca_values = pca.fit_transform(wine_norm)
pca_values
# PCA Components matrix or convariance Matrix
pca.components_
# The amount of variance that each PCA has
var = pca.explained_variance_ratio_
var
# Cummulative variance of each PCA
Var = np.cumsum(np.round(var,decimals= 4)*100)
Var
plt.plot(Var,color="blue")
# Final Dataframe
final_df=pd.concat([a['Type'],pd.DataFrame(pca_values[:,0:3],columns=['PC1','PC2','PC3'])],axis=1)
final_df

# Visualization of PCAs
import seaborn as sns
fig=plt.figure(figsize=(6,4))
sns.scatterplot(data=final_df)
sns.scatterplot(data=final_df, x='PC1', y='PC2', hue='Type')
pca_values[: ,0:1]
x= pca_values[:,0:1]
y= pca_values[:,1:2]
plt.scatter(x,y)
#Hierarchical Clustering
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
plt.figure(figsize=(10,8))
dendrogram=sch.dendrogram(sch.linkage(wine_norm,'complete'))
# Create Clusters
hclusters=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
hclusters
y=pd.DataFrame(hclusters.fit_predict(wine_norm),columns=['clustersid'])
y['clustersid'].value_counts()
wine2=a.copy()
wine2['clustersid']=hclusters.labels_
wine2

#K-Means
wcss=[]
for i in range (1,6):
    kmeans=KMeans(n_clusters=i,random_state=2)
    kmeans.fit(wine_norm)
    wcss.append(kmeans.inertia_)
# Plot K values range vs WCSS to get Elbow graph for choosing K (no. of clusters)
plt.plot(range(1,6),wcss)
plt.title('Elbow Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')

# Cluster algorithm , K is 3
clusters3=KMeans(3,random_state=30).fit(wine_norm)
clusters3
clusters3.labels_
wine3=a.copy()
wine3['clusters3id']=clusters3.labels_
wine3

wine3['clusters3id'].value_counts()























