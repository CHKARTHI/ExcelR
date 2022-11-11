# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 12:20:58 2022

@author: DELL
"""


### problem 1

import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import shapiro





X=pd.read_csv('delivery_time.csv')
X.info()
stat, pval = shapiro(X) 
sns.distplot(X['Delivery Time'])
sns.distplot(X['Sorting Time'])
import matplotlib.pyplot as plt
plt.subplots(figsize=(5,5))
plt.subplot(1,2,1)
plt.boxplot(X['Delivery Time'],vert=False)
plt.title('Delivery Time')
plt.subplot(1,2,2)
plt.boxplot(X['Sorting Time'],vert=False)
plt.title('Sorting Time') #no outliers


X.corr()
pd.DataFrame(np.array(X['Delivery Time']).reshape(-1,1))
sns.regplot(x=X['Delivery Time'],y=X['Sorting Time'])
y=X['Delivery Time']
x=X['Sorting Time']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3)

print(x_train.shape)
print(x_test.shape)

print(y_train.shape)
print(y_test.shape)
from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(np.array(x_train).reshape(-1,1),np.array(y_train))
lr.intercept_
lr.coef_
y_pred_train = pd.DataFrame(lr.predict(np.array(x_train).reshape(-1,1)))
y_pred_train = pd.DataFrame(lr.predict(np.array(x_train).reshape(-1,1)))
y_pred_test = pd.DataFrame(lr.predict(np.array(x_test).reshape(-1,1)))
from sklearn.metrics import mean_squared_error , r2_score
mse_train = mean_squared_error(y_train,y_pred_train)
mse_test = mean_squared_error(y_test,y_pred_test)
mse_train
mse_test
sns.regplot(y_train,y_pred_train )
sns.regplot(y_test,y_pred_test )
plt.subplots(figsize=(5,5))
plt.subplot(1,2,1)
plt.boxplot(sns.regplot(y_train,y_pred_train ))
plt.title('Train model')
plt.subplot(1,2,2)
plt.boxplot(sns.regplot(y_test,y_pred_test))
plt.title('test model')
print('RMSE_train:',np.sqrt(mse_train).round(2))
print('RMSE_test:',np.sqrt(mse_test).round(2))
print('R2_train:',r2_score(y_train,y_pred_train).round(2))
print('R2_test:',r2_score(y_test,y_pred_test).round(2))



#problem 2



A=pd.read_csv('Salary_Data.csv')
A.corr()
a=A[['Salary']]
b=A[['YearsExperience']]
#EDA
stat, pval = shapiro(A) 
sns.distplot(a)
sns.distplot(b)
import matplotlib.pyplot as plt
plt.subplots(figsize=(5,5))
plt.subplot(1,2,1)
plt.boxplot(a,vert=False)
plt.title('Salary')
plt.subplot(1,2,2)
plt.boxplot(b,vert=False)
plt.title('Exp') #no outliers
sns.regplot(b,a)
#standarddising the data 
#from sklearn.preprocessing import LabelEncoder
#lb=LabelEncoder()
#a1=lb.fit_transform(a)
#Splitting the data
from sklearn.model_selection import train_test_split
a_train, a_test, b_train, b_test = train_test_split(a,b,test_size=0.3)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(a,b)

a_train_pred = lr.predict(b_train)
a_test_pred = lr.predict(b_test)

from sklearn.metrics import mean_squared_error, r2_score
RMSE_train = np.sqrt(mean_squared_error(a_train_pred,a_train))
RMSE_test = np.sqrt(mean_squared_error(a_test_pred,a_test))
print('RMSE_train:',RMSE_train)
print('RMSE_test:',RMSE_test)
print('r2_train:',abs(r2_score(a_train_pred,a_train)))
print('r2_train:',abs(r2_score(a_test_pred,a_test)))

#validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kf=KFold(n_splits=5)
result=abs(cross_val_score(lr,a,b,scoring="neg_mean_squared_error"))
result.mean()

#regularization 
from sklearn.linear_model import Ridge
r=Ridge(alpha=5)
r.fit(a_train,b_train)
a_train_pred = r.predict(b_train)
a_test_pred = r.predict(b_test)
from sklearn.metrics import mean_squared_error, r2_score
RMSE_train = np.sqrt(mean_squared_error(a_train_pred,a_train))
RMSE_test = np.sqrt(mean_squared_error(a_test_pred,a_test))
print('RMSE_train:',RMSE_train)
print('RMSE_test:',RMSE_test)
print('r2_train:',abs(r2_score(a_train_pred,a_train)))
print('r2_train:',abs(r2_score(a_test_pred,a_test)))

#the models are good fit.





































































































