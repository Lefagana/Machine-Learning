# -*- coding: utf-8 -*-
"""
Created on Sun Mar 04 23:42:58 2022

@author: MALAH-LG

Formular for multiple linear regression 
y = b0 + b1 * x1 + b2 * x2 + ....bn * xn  
 where b0 is the constant
 and the remaining (b) are coefficient
x is the independent viariable
and y is the dependant variable

"""

#importing lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv("50_Startups.csv")
#dataset without purchase column
x = dataset.iloc[:,:-1].values
#dataset  purchase column
y = dataset.iloc[:,4].values 


# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#for country independent variable
le = LabelEncoder()
x[:,3] = le.fit_transform(x[:,3])
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

#Avoiding the Dummy variable trap
x = x[:, 1:];

#Splitting dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""


# Fitting  multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#Predict the test set rule
# vector of depndant varialble  predicted value 
y_predict = regressor.predict(x_test)

# Building the Optimal model using Backword Elimination
import statsmodels.formula.api as sm

 