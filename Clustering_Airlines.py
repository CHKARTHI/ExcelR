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























