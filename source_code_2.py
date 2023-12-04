#!/usr/bin/env python
# coding: utf-8

# In[168]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# In[169]:


def sort_dataset(dataset_df):
    sorted_df = dataset_df.sort_values(by='year')
    return sorted_df


# In[170]:


def split_dataset(dataset_df):
    dataset_df['label'] = dataset_df['salary'] * 0.001
    
    train = dataset_df.loc[:1718]
    test = dataset_df.loc[1718:]
    
    x_train = train.drop(['salary', 'label'], axis=1)
    y_train = train['label']
    
    x_test = test.drop(['salary', 'label'], axis=1)
    y_test = test['label']
    
    return x_train, x_test, y_train, y_test


# In[171]:


def extract_numerical_cols(dataset_df):
    result = dataset_df[['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war']]
    return result


# In[172]:


def train_predict_decision_tree(X_train, Y_train, X_test):
    dtr = DecisionTreeRegressor()
    dtr.fit(X_train, Y_train)
    
    result = dtr.predict(X_test)
    
    return result


# In[173]:


def train_predict_random_forest(X_train, Y_train, X_test):
    rf = RandomForestRegressor()
    rf.fit(X_train, Y_train)
    
    result = rf.predict(X_test)
    
    return result


# In[174]:


def train_predict_svm(X_train, Y_train, X_test):
    svm_pipe = make_pipeline(StandardScaler(), SVR())
    svm_pipe.fit(X_train, Y_train)
    
    result = svm_pipe.predict(X_test)
    
    return result


# In[175]:


def calculate_RMSE(labels, predictions):
    mse = np.mean((predictions-labels)**2)
    rmse = np.sqrt(mse)
    
    return rmse


# In[176]:


if __name__=='__main__':
	#DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
	data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')
	
	sorted_df = sort_dataset(data_df)	
	X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)
	
	X_train = extract_numerical_cols(X_train)
	X_test = extract_numerical_cols(X_test)

	dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
	rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
	svm_predictions = train_predict_svm(X_train, Y_train, X_test)
	
	print ("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))	
	print ("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))	
	print ("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))


# In[ ]:




