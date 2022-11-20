# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 15:07:32 2022

@author: DELL
"""

##ToyotaCorolla

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

x=pd.read_csv("ToyotaCorolla.csv",encoding='latin1')
list(x.head())
x.drop('Id',axis=1,inplace=True)    
plt.boxplot(x['Price'],vert=False)
sns.distplot(x['Price'])
q3 = np.percentile(x['Price'],75)
q1=np.percentile(x['Price'],25)
iqr=q3-q1
uw=q3+1.5*iqr

x.drop(x[x.Price>uw].index,inplace=True)
x
x.corr()
sns.distplot(x['Price'])
plt.hist(x['Price'])
Y=x['Price']
X=x.loc[:,["Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]

from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
#ss.fit_tranform(x)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=50,test_size=0.3)
from sklearn.linear_model import Lasso
LS=Lasso(alpha=12)
LS.fit(X,Y)
Y_pred_train = LS.predict(X_train)
Y_pred_test = LS.predict(X_test)
from sklearn.metrics import mean_squared_error
train_error = mean_squared_error(Y_train,Y_pred_train)
print("Training Error:" ,np.sqrt(train_error).round(3))
test_error = mean_squared_error(Y_test,Y_pred_test)
print("Test Error:" ,np.sqrt(test_error).round(3))
pd.DataFrame(LS.coef_)
pd.DataFrame(X.columns)
pd.concat([pd.DataFrame(LS.coef_),pd.DataFrame(X.columns)],axis=1)

#KM can be exluded from above data
X_new = x.loc[:,["Age_08_04","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]

X_new_train, X_new_test, Y_train, Y_test = train_test_split(X_new,Y,random_state=50,test_size=0.4)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_new,Y)
Y_train_predict = lr.predict(X_new_train)
Y_test_predict = lr.predict(X_new_test)
from sklearn.metrics import mean_squared_error, r2_score
train_error=mean_squared_error(Y_train_predict, Y_train)
print("train_error:", np.sqrt(train_error).round(3))
test_error=mean_squared_error(Y_test_predict, Y_test)
print("test_error:", np.sqrt(test_error).round(3))
train_r2 = r2_score(Y_train_predict, Y_train)
print("train_r2:", train_r2.round(3))
test_r2 = r2_score(Y_test_predict, Y_test)
print("test_r2:", test_r2.round(3))
#now model is good fit

#### 50_Startups
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


a = pd.read_csv("50_Startups.csv")
a
list(a.head())
list(a)
a.drop(index=a[a['R&D Spend'] == 0].index, inplace=True)
a.drop(index=a[a['Administration'] == 0].index, inplace=True)
a.drop(index=a[a['Marketing Spend'] == 0].index, inplace=True)
a.drop(index=a[a['State'] == 0].index, inplace=True)
a
plt.boxplot(a['Profit'],vert=False)
sns.distplot(a['Profit'])
a.corr()
a.describe()
sns.set_style(style='darkgrid')
sns.pairplot(a)
from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
a["State"]=l.fit_transform(a["State"])
a
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x=ss.fit_transform(a)
X=pd.DataFrame(x)
X.columns =['R&D Spend', 'Administration', 'Marketing Spend', 'State','Profit']
X
X1=X.loc[:,["R&D Spend","Administration","Marketing Spend","State"]]
Y1=X["Profit"]
#Evaluating using cross validation
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
lr=LinearRegression()
lv=LeaveOneOut()
results=abs(cross_val_score(lr,X1,Y1,cv=lv, scoring="neg_mean_squared_error"))
results
np.mean(results)

#regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
    lr=LinearRegression()
    lr.fit(X1,Y1)
r2_training=[]
r2_test=[]
for i in range (1 , 40):
  X_train, X_test, Y_train, Y_test = train_test_split(X1,Y1,random_state=i,test_size=0.3)

Y_train_predict = lr.predict(X_train)
Y_test_predict = lr.predict(X_test)
r2_training.append(r2_score(Y_train_predict, Y_train))
r2_test.append(r2_score(Y_test_predict, Y_test))
TE0 = pd.DataFrame(r2_training)
TE1 = pd.DataFrame(r2_test)
frames = [TE0,TE1]
a=pd.concat(frames)
a
##model is good fit






















