# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 22:54:22 2022

@author: DELL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


x=pd.read_csv('EastWestAirlines.csv')
x
x.info()
x1=x.drop(['ID#'],axis=1)
x1
from sklearn.preprocessing import normalize
x1_norm=pd.DataFrame(normalize(x1),columns=x1.columns)
x1_norm
#dendogram
import scipy.cluster.hierarchy as sch
plt.figure(figsize=(10, 7))  
dendograms=sch.dendrogram(sch.linkage(x1_norm,'single'))


from sklearn.cluster import AgglomerativeClustering

#Clusters
clusters=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='complete')
clusters
y=pd.DataFrame(clusters.fit_predict(x1_norm),columns=['clusters'])
y['clusters'].value_counts()
x1['clusters']=clusters.labels_
x1
x1.groupby('clusters').agg(['mean']).reset_index()
# Plot Clusters
plt.figure(figsize=(10, 7))  
plt.scatter(x1['clusters'],x1['Balance'], c=clusters.labels_) 

#Kmeans
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
Airdata=pd.read_csv('EastWestAirlines.csv')
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)                       #normalize can't be used
df_norm=norm_func(Airdata.iloc[:,1:])
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
Airdata['Clust']=x
Airdata
Airdata.iloc[:,1:10].groupby(Airdata.Clust).mean()

#DBSCAN

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

Airdata.info()
df=Airdata.iloc[:,1:10]
df.values
stscaler=StandardScaler().fit(df.values)
x=stscaler.transform(df.values)
x
dbscan=DBSCAN(eps=2,min_samples=5)
dbscan.fit(x)
dbscan.labels_
cl=pd.DataFrame(dbscan.labels_,columns=['cluster'])
cl
pd.concat([Airdata,cl],axis=1)






















