# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 13:08:54 2022

@author: DELL
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#####
x=pd.Series([24.23,25.53,25.41,24.14,29.62,28.25,25.81,24.39,40.26,32.95,91.36,25.99,39.42,26.71,35.00])
name=['Allied Signal','Bankers Trust','General Mills','ITT Industries','J.P.Morgan & Co.','Lehman Brothers',
      'Marriott','MCI','Merrill Lynch','Microsoft','Morgan Stanley','Sun Microsystems','Travelers','US Airways',
      'Warner-Lambert']
# Pie Plot
plt.figure(figsize=(6,8))
plt.pie(x,labels=name,autopct='%1.0f%%')
plt.show()

plt.boxplot(x,vert=False)
x.mean()
x.var()
x.std()

####
from scipy import stats
p=1/200
n=5
q=199/200

1-stats.binom(5,1/200).cdf(0)


#####################3

a = np.array([-2000,-1000,0,1000,2000,3000])
b = np.array([0.1,0.1,0.2,0.2,0.3,0.1])
c=np.array([a*b])
C=pd.DataFrame({'x':[-2000,-1000,0,1000,2000,3000],'Px':[0.1,0.1,0.2,0.2,0.3,0.1],'Y':[-200., -100.,    0.,  200.,  600.,  300.]})
C.Y.sum()
d=c=np.array([a*a*b])
d.sum()
np.sqrt(d.sum()-(C.Y.sum()*C.Y.sum()))
########################################

import scipy
from scipy import stats

1-stats.norm(45,8).cdf(50)

stats.norm(38,6).cdf(44)-stats.norm(38,6).cdf(34)
1-stats.norm(38,6).cdf(44)

stats.norm(38,6).cdf(30)*400

#######

stats.norm.interval(0.99,100,20)


#####
import numpy as np
Mean = (5+7)*45
SD = (np.sqrt((9)+(16)))*45
stats.norm.interval(0.95,Mean,SD)
Mean+SD*-1.645
stats.norm.cdf(0,5,3)
stats.norm.cdf(0,7,4)
####################
z_scores=(0.046-0.05)/(np.sqrt((0.05*(1-0.05))/2000))

p_value=1-stats.norm.cdf(abs(z_scores))
p_value


####

SD= 40/np.sqrt(100)

Z45 = (45-50)/SD
Z55 = (55-50)/SD
A=stats.norm.cdf(Z55)-stats.norm.cdf(Z45)
1-A
stats.norm.interval(0.7887,loc=50,scale=SD)

#1.96= 5/40/np.sqrt(n)
15.68*15.68


































































































