# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 15:54:17 2022

@author: DELL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
x=pd.read_csv('bank-full.csv',sep=';')
x
#EDA
x.info()
x1=pd.get_dummies(x,columns=['job','marital','education','contact','poutcome','month'])
x1
pd.set_option("display.max.columns", None)
x1
x1.info()
x1['default'] = np.where(x1['default'].str.contains("yes"), 1, 0)
x1['housing'] = np.where(x1['housing'].str.contains("yes"), 1, 0)
x1['loan'] = np.where(x1['loan'].str.contains("yes"), 1, 0)
x1['y'] = np.where(x1['y'].str.contains("yes"), 1, 0)
x1
x1.info()
#Model
X=pd.concat([x1.iloc[:,0:10],x1.iloc[:,11:]],axis=1)
Y=x1.iloc[:,10]
classifier=LogisticRegression()
classifier.fit(X,Y)
Y_pred=classifier.predict(X)
Y_pred
Y_pred_df=pd.DataFrame({'actual_Y':Y,'Y_pred_prob':Y_pred})
Y_pred_df
confusion_matrix = confusion_matrix(Y,Y_pred)
confusion_matrix
# The model accuracy is calculated by (a+d)/(a+b+c+d)
(39156+1162)/(39156+766+4127+1162)
classifier.predict_proba(X)[:,1] 
# ROC Curve plotting and finding AUC value
fpr,tpr,thresholds=roc_curve(Y,classifier.predict_proba(X)[:,1])
plt.plot(fpr,tpr,color='red')
auc=roc_auc_score(Y,Y_pred)

plt.plot(fpr,tpr,color='red',label='logit model(area  = %0.2f)'%auc)
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')
plt.show()

print('auc accuracy:',auc)






















