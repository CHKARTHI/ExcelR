# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 13:24:24 2022

@author: DELL
"""

import numpy as np
import pandas as pd
X=pd.read_csv("Cutlets.csv")

import statsmodels
from statsmodels.stats import weightstats as ztests
Zcal,Pval = ztests.ztest(X['Unit A'],X['Unit B'],alternative='two-sided')
 #other way
from scipy import stats
from scipy.stats import norm
 
statistic , p_value = stats.ttest_ind(X['Unit A'],X['Unit B'], alternative = 'two-sided') 
 
########




Y = pd.read_csv("LabTAT.csv")
Y.insert(0,"none",1)
Y
A=Y.set_index(["none"]).stack()
A
A.index.set_names('name', level=len(A.index.names)-1, inplace=True)
B=A.reset_index().rename(columns={0:'value'})
B["name"]
B["value"]
B["value"] = pd.to_numeric(B["value"])
from statsmodels.formula.api import ols
lm1 = ols('value ~ C(name)',data=B).fit()
import statsmodels.api as sm
table = sm.stats.anova_lm(lm1, type=1)
print(table)
### other methodd 
Y = pd.read_csv("LabTAT.csv")
test_statistic , p_value = stats.f_oneway(Y.iloc[:,0],Y.iloc[:,1],Y.iloc[:,2],Y.iloc[:,3])

####
X = pd.read_csv("BuyerRatio.csv")
Y=X.iloc[:,1:6]
Y
from scipy import stats
val=stats.chi2_contingency(Y)
type(val)
no_of_rows=len(Y.iloc[0:2,0])
no_of_columns=len(Y.iloc[0,0:4])
degree_of_f=(no_of_rows-1)*(no_of_columns-1)
print('Degree of Freedom=',degree_of_f)

Expected_value=val[3]
Expected_value

chi_square=sum([(o-e)**2/e for o,e in zip(Y.values,Expected_value)])
chi_square_statestic=chi_square[0]+chi_square[1]
chi_square_statestic

from scipy.stats import chi2
critical_value=chi2.ppf(0.95,3)
critical_value

if chi_square_statestic >= critical_value:
	print('Dependent (reject H0:male-female buyer rations are not similar across regions)')
else:
	print('Independent (accept H0: male-female buyer rations are similar across regions)')
    
p = 1-chi2.cdf(chi_square_statestic,3)

####


Y = pd.read_csv("Costomer+OrderForm.csv")
Y.insert(0,"none",1)
Y
A=Y.set_index(["none"]).stack()
A.index.set_names('country', level=1, inplace=True)
B=A.reset_index().rename(columns={0:'type'})

pip install researchpy
import researchpy as rp
Table, results = rp.crosstab(B["country"],B["type"],test='chi-square')

print(results)


























































