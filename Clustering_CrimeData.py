# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 23:27:41 2022

@author: DELL
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
x=pd.read_csv('crime_data.csv')
x
#Normalized data fuction
def norm(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)
x_norm=norm(x.iloc[:,1:])
x_norm
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(x_norm,method='average'))

#Hierarchical
from sklearn.cluster import AgglomerativeClustering
clusters=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='complete')
clusters
y_clusters=clusters.fit_predict(x_norm)
y_clusters
x['clusters']=clusters.labels_
x

#Kmeans
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
crim1=pd.read_csv('crime_data.csv')
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)                       #normalize can't be used
df_norm=norm_func(crim1.iloc[:,1:])
# Elbow curv
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow curv')
plt.xlabel('number of clusters')
plt.ylabel('WCSS')
plt.show()
# selecting clusters from above scree plot
model=KMeans(n_clusters=4)
model.fit(df_norm)
model.labels_
x=pd.Series(model.labels_)
crim1['Clust']=x
crim1
crim1.iloc[:,1:5].groupby(crim1.Clust).mean()

#DBSCAN

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

crim3=pd.read_csv('crime_data.csv')
crim3.info()
df=crim3.iloc[:,1:5]
df.values
stscaler=StandardScaler().fit(df.values)
x=stscaler.transform(df.values)
x
dbscan=DBSCAN(eps=2,min_samples=5)
dbscan.fit(x)
dbscan.labels_
cl=pd.DataFrame(dbscan.labels_,columns=['cluster'])
cl
pd.concat([crim3,cl],axis=1)



















































