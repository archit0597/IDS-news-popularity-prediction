#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 14:34:26 2023

@author: saikalyantaruntiruchirapally
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def perform_eda(data):
    # Basic idea of the data
    print("\nBasic idea of the data:\n",data.info())    
    
    # Basic summary of the data
    print("\nBasic summary of the data:\n",data.describe())
    
    #Exploring different features in the data
    data.columns = data.columns.str.strip()
    print("\nDifferent features in the dataset:\n",data.columns)
    
    # Exploring our target column shares.
    share_data = data['shares']
    print("\n Exploring our target column shares:\n",data['shares'].describe())
    
    # Missing values in the data
    null_data = data.isnull()
    print('\nNull percentage in data:\n', null_data.sum())
    
    # Calculate correlation matrix
    corr_matrix = data.corr()

    # Print correlation coefficients of each feature with target variable
    print("\ncorrelation coefficients of each feature with target variable:\n",corr_matrix['shares'].sort_values(ascending=False))

    # Visualize correlation matrix with heatmap
    sns.heatmap(corr_matrix, cmap="YlGnBu", annot=True)
    plt.show()    
    
    # Check the distribution of the target variable
    plt.figure(figsize=(8,6))
    sns.histplot(data['shares'], bins=50)
    
    # Plot the share v/s words
    data = data[data['n_tokens_content'] != 0]
    plt.figure(figsize=(10,5))
    ax = sns.scatterplot(y='shares', x='n_tokens_content', data=data)
    
    # Plot the popularity for every day of the week.
    a = data['shares'].mean()
    Wday = data.columns.values[31:38]
    unpopular=data[data['shares']<a]
    popular=data[data['shares']>=a]
    Unpop_day = unpopular[Wday].sum().values
    Pop_day = popular[Wday].sum().values
    fig = plt.figure(figsize = (13,5))
    plt.title("Count of popular/unpopular news over different day of week (Mean)", fontsize = 16)
    plt.bar(np.arange(len(Wday)),Pop_day,width=0.3,align='center',color='b',label='Popular')
    plt.bar(np.arange(len(Wday))-0.3,Unpop_day,width=0.3,align='center',color='r',label='Unpopular')
    plt.xticks(np.arange(len(Wday)), Wday)
    plt.ylabel('COUNT',fontsize=15)
    plt.xlabel('Day of Week',fontsize=17)
    plt.legend(loc = 'upper right')
    plt.tight_layout()
    plt.show()
    