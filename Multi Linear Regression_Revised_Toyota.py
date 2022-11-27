# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 14:19:51 2022

@author: DELL
"""

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import influence_plot
# import dataset
x=pd.read_csv('ToyotaCorolla.csv',encoding='latin1')
x
#EDA
x.info
x1=pd.concat([x.iloc[:,2:4],x.iloc[:,6:7],x.iloc[:,8:9],x.iloc[:,12:14],x.iloc[:,15:18]],axis=1)
x1
x2=x1.rename({'Age_08_04':'Age','cc':'CC','Quarterly_Tax':'QT'},axis=1)
x2[x2.duplicated()]
x3=x2.drop_duplicates().reset_index(drop=True)
x3
x3.describe()
#Correlation Analysis
x3.corr()
sns.set_style(style='darkgrid')
sns.pairplot(x3)
#Model Building
model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=x3).fit()
#Modedl testing
model.params
model.tvalues , np.round(model.pvalues,5)
model.rsquared , model.rsquared_adj
slr_c=smf.ols('Price~CC',data=x3).fit()
slr_c.tvalues , slr_c.pvalues
slr_d=smf.ols('Price~Doors',data=x3).fit()
slr_d.tvalues , slr_d.pvalues
mlr_cd=smf.ols('Price~CC+Doors',data=x3).fit()
mlr_cd.tvalues , mlr_cd.pvalues
#Model Validation Techniques _1. Collinearity Check
#Variance inflation factor
rsq_age=smf.ols('Age~KM+HP+CC+Doors+Gears+QT+Weight',data=x3).fit().rsquared
vif_age=1/(1-rsq_age)

rsq_KM=smf.ols('KM~Age+HP+CC+Doors+Gears+QT+Weight',data=x3).fit().rsquared
vif_KM=1/(1-rsq_KM)

rsq_HP=smf.ols('HP~Age+KM+CC+Doors+Gears+QT+Weight',data=x3).fit().rsquared
vif_HP=1/(1-rsq_HP)

rsq_CC=smf.ols('CC~Age+KM+HP+Doors+Gears+QT+Weight',data=x3).fit().rsquared
vif_CC=1/(1-rsq_CC)

rsq_DR=smf.ols('Doors~Age+KM+HP+CC+Gears+QT+Weight',data=x3).fit().rsquared
vif_DR=1/(1-rsq_DR)

rsq_GR=smf.ols('Gears~Age+KM+HP+CC+Doors+QT+Weight',data=x3).fit().rsquared
vif_GR=1/(1-rsq_GR)

rsq_QT=smf.ols('QT~Age+KM+HP+CC+Doors+Gears+Weight',data=x3).fit().rsquared
vif_QT=1/(1-rsq_QT)

rsq_WT=smf.ols('Weight~Age+KM+HP+CC+Doors+Gears+QT',data=x3).fit().rsquared
vif_WT=1/(1-rsq_WT)

d1={'Variables':['Age','KM','HP','CC','Doors','Gears','QT','Weight'],
    'Vif':[vif_age,vif_KM,vif_HP,vif_CC,vif_DR,vif_GR,vif_QT,vif_WT]}
Vif_df=pd.DataFrame(d1)
Vif_df
# Residual Analysis
sm.qqplot(model.resid,line='q')
plt.title("Normal Q-Q plot of residuals")
plt.show()
list(np.where(model.resid>6000))
list(np.where(model.resid<-6000))          
def standard_values(vals) : return (vals-vals.mean())/vals.std() 

plt.scatter(standard_values(model.fittedvalues),standard_values(model.resid))
plt.title('Residual Plot')
plt.xlabel('standardized fitted values')
plt.ylabel('standardized residual values')
plt.show() 

fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Age',fig=fig)
plt.show()
fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'KM',fig=fig)
plt.show()
fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'HP',fig=fig)
plt.show()
fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'CC',fig=fig)
plt.show()
fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Doors',fig=fig)
plt.show()
fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Gears',fig=fig)
plt.show()
fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'QT',fig=fig)
plt.show()
fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Weight',fig=fig)
plt.show()

# Model Deletion Diagnostics_Cook's Distance
(c,_)=model.get_influence().cooks_distance
c
fig=plt.figure(figsize=(20,7))
plt.stem(np.arange(len(x3)),np.round(c,3))
plt.xlabel('Row Index')
plt.ylabel('Cooks Distance')
plt.show()
np.argmax(c) , np.max(c)
fig,ax=plt.subplots(figsize=(20,20))
fig=influence_plot(model,ax = ax)

#Leverage cut off
k=x3.shape[1]
n=x3.shape[0]
leverage_cutoff = (3*(k+1))/n
leverage_cutoff
x3[x3.index.isin([80])] 
#modedl improvement
x4=x3.copy()
x4
x5=x4.drop(x4.index[[80]],axis=0).reset_index(drop=True)
x5

#Model Deletion Diagnostics and Final Model
while np.max(c)>0.5 :
    model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=x5).fit()
    (c,_)=model.get_influence().cooks_distance
    c
    np.argmax(c) , np.max(c)
    x5=x5.drop(x5.index[[np.argmax(c)]],axis=0).reset_index(drop=True)
    x5
else:
    final_model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=x5).fit()
    final_model.rsquared , final_model.aic
    print("Thus model accuracy is improved to",final_model.rsquared)

if np.max(c)>0.5:
    model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=x5).fit()
    (c,_)=model.get_influence().cooks_distance
    c
    np.argmax(c) , np.max(c)
    x5=x5.drop(toyo5.index[[np.argmax(c)]],axis=0).reset_index(drop=True)
    x5 
elif np.max(c)<0.5:
    final_model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=x5).fit()
    final_model.rsquared , final_model.aic
    print("Thus model accuracy is improved to",final_model.rsquared)

final_model.rsquared
x5

#model preddictions
new_data=pd.DataFrame({'Age':12,"KM":40000,"HP":80,"CC":1300,"Doors":4,"Gears":5,"QT":69,"Weight":1012},index=[0])
new_data
final_model.predict(new_data)
pred_y=final_model.predict(toyo5)
pred_y















































