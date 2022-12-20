# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 11:26:28 2022

@author: DELL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

book = pd.read_csv("book.csv",encoding='latin-1')
book.head()
book.tail()

#EDA
book.shape
book.info()
book.isnull().sum()
book.nunique()
book.columns = ["","UserID","BookTitle","BookRating"]
book
book =book.sort_values(by=['UserID'])
#number of unique users in the dataset
len(book.UserID.unique())
#Unique movies
len(book.BookTitle.unique())

book.loc[book["BookRating"] == 'small', 'BookRating'] = 0
book.loc[book["BookRating"] == 'large', 'BookRating'] = 1
book.BookRating.value_counts()
plt.figure(figsize=(20,6))
sns.distplot(book.BookRating)
book_df = book.pivot_table(index='UserID',
                   columns='BookTitle',
                   values='BookRating').reset_index(drop=True)
book_df.fillna(0,inplace=True)
book_df

#AVERAGE RATING OF BOOKS
AVG = book['BookRating'].mean()
print(AVG)
# Calculate the minimum number of votes required to be in the chart, 
minimum = book['BookRating'].quantile(0.90)
print(minimum)

# Filter out all qualified Books into a new DataFrame
q_Books = book.copy().loc[book['BookRating'] >= minimum]
q_Books.shape
#Cosine similarity
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine,correlation
user_sim=1-pairwise_distances(book_df.values,metric='cosine')
user_sim

user_sim_df=pd.DataFrame(user_sim)
user_sim_df
#Set the index and column names to user ids 
user_sim_df.index = book.UserID.unique()
user_sim_df.columns = book.UserID.unique()
user_sim_df

np.fill_diagonal(user_sim,0)
user_sim_df
#Most Similar Users
print(user_sim_df.idxmax(axis=1)[1348])
print(user_sim_df.max(axis=1).sort_values(ascending=False).head(50))
reader = book[(book['UserID']==1348) | (book['UserID']==2576)]
reader
reader1=book[(book['UserID']==1348)] 
reader1
reader2=book[(book['UserID']==2576)] 
reader2


















