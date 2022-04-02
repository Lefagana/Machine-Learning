# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 23:42:58 2022

@author: MALAH-LG
"""
#importing lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv("Data.csv")
#dataset without purchase column
x = dataset.iloc[:,:-1].values
#dataset  purchase column
y = dataset.iloc[:,3].values 

#dataset missing values
#the reasons we used the NaN as a missing value is bcos our data has a NaN value
#axis = 0 is for column and 1 is for rows
#for age
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean') 
# the upper bound is included thats why it dosnt start with index 0
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])


# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#for country
le = LabelEncoder()
x[:,0] = le.fit_transform(x[:,0])
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
#for purchase
le = LabelEncoder()
y = le.fit_transform(y) 

#Splitting dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))