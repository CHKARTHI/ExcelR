# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 21:17:24 2022

@author: DELL
"""
##Q6
def expected_value(X,Y):
    values = np.asarray(X)
    weights = np.asarray(Y)
    return (values * weights).sum()

X=[1,4,3,5,6,2]
Y=[0.015,0.20,0.65,0.005,0.01,0.120]
expected_value(X,Y)

#Q7
X=[108,110,123,134,135,145,167,187,199]

Y=[1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9]
expected_value(X,Y)


#......................................
#Q7: ) Calculate Mean, Median, Mode, Variance, Standard Deviation, Range &     
#comment about the values / draw inferences

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
cars=pd.read_csv("Q7.csv")
cars
# mean
cars.mean()
# Median
cars.median()
# Mode
cars.mode()
cars.Points.mode() 
cars.Score.mode()
cars.Weigh.mode()
# Variance
cars.var()
# Satndard Deviation
cars.std()
# Range
cars.describe()
Points_Range=cars.Points.max()-cars.Points.min()
Points_Range
Score_Range=cars.Score.max()-cars.Score.min()
Score_Range
Weigh_Range=cars.Weigh.max()-cars.Weigh.min()
Weigh_Range

#Plotting
plt.subplots(figsize=(5,2))
plt.subplot(1,3,1)
plt.boxplot(cars.Points)
plt.title('Points')
plt.subplot(1,3,2)
plt.boxplot(cars.Score)
plt.title('Score')
plt.subplot(1,3,3)
plt.boxplot(cars.Weigh)
plt.title('Weigh')

#histogram
plt.hist(cars["Points"], bins = 10, edgecolor= 'black')
plt.hist(cars["Score"], bins = 10, edgecolor= 'green')
plt.hist(cars["Weigh"], bins = 10, edgecolor= 'red')
#bargraph
plt.figure(figsize=(5,5))
plt.barh(cars["Cars"],cars["Points"])
#plt.barh(cars["Cars"],cars["Score"])
#plt.barh(cars["Cars"],cars["Weigh"])
plt.yticks(fontsize=5)
plt.show()
#####...............................................................

#Q9

X=pd.read_csv("Q9_a.csv")
X["speed"]
X["dist"]

X.speed.skew()
X.dist.skew()
X.speed.kurt()
X.dist.kurt()

print("Speed skewness:",X.speed.skew().round(2))
print("Speed Kurtosis:",X.speed.kurt().round(2))
print("dist skewness:",X.dist.skew().round(2))
print("dist kurtosis:",X.dist.kurt().round(2))

import matplotlib.pyplot
plt.figure(figsize=(3,3))
plt.bar(X.speed,X.dist)
#plt.yticks()
plt.hist(X.speed,histtype='step')
plt.hist(X.dist)

#......................
#Sp and WT


Xa=pd.read_csv("Q9_b.csv")
Xa["SP"]
Xa["WT"]

Xa.SP.skew()
Xa.WT.skew()
Xa.SP.kurt()
Xa.WT.kurt()

print("SP skewness:",Xa.SP.skew().round(2))
print("SP Kurtosis:",Xa.SP.kurt().round(2))
print("WT skewness:",Xa.WT.skew().round(2))
print("WT kurtosis:",Xa.WT.kurt().round(2))

import matplotlib.pyplot
plt.figure(figsize=(3,3))
plt.bar(Xa.SP,Xa.WT)
#plt.yticks()
plt.hist(Xa.SP,histtype='step')
plt.hist(Xa.WT)


#Q10....

plt.bar(Xa.SP,Xa.WT)
plt.ylabel(ylabel="WT")
plt.xlabel(xlabel="SP")
plt.subplots(figsize=(5,2))
plt.subplot(1,2,1)
plt.boxplot(Xa.SP)
plt.subplot(1,2,2)
plt.boxplot(Xa.WT)

from scipy import stats
# Avg. weight of Adult in Mexico with 94% CI
#scale 
SD=(30/np.sqrt(2000))
stats.norm.interval(0.94,200,SD)

#98%
stats.norm.interval(0.98,200,SD)
#96%
stats.norm.interval(0.96,200,SD)

##############################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
Marks=np.array([34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56])
Marks.mean()

Marks.var()
Marks.std()
pd.DataFrame(Marks).median()
sns.distplot(pd.DataFrame(Marks))
plt.boxplot(Marks,vert=False)
Marks.skew()
pd.DataFrame(Marks).skew()
IQR=((Marks.max()-Marks.min())*0.75)-((Marks.max()-Marks.min())*0.25)

##########

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

cars=pd.read_csv("Cars.csv")
cars
sns.distplot(cars.MPG, label='Cars-MPG')

B=pd.read_csv("wc-at.csv")
sns.distplot(B.AT, label='AT')
sns.distplot(B.Waist, label='Waist')
B.mean()
B.median()

#########Q22
from scipy import stats
stats.norm.ppf(0.95)
stats.norm.ppf(0.97)
stats.norm.ppf(0.8)

######Q23
stats.t.ppf(0.975,24)
stats.t.ppf(0.98,24)
stats.t.ppf(0.995,24)


####Q24

# Assume Null Hypothesis is: Ho = Avg life of Bulb >= 260 days
# Alternate Hypothesis is: H1 = Avg life of Bulb < 260 days
t=(260-270)/(90/np.sqrt(18))
p_value=1-stats.t.cdf(abs(t),df=17)































































