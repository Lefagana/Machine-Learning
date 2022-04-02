# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 15:05:05 2022

@author: MALAH-LG
"""

#importing lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing dataset
dataset = pd.read_csv("Salary_Data.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values 

#Splitting dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
# object of the class LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#Predict the test set rule
# vector of depndant varialble  predicted value that's the salary
y_predict = regressor.predict(x_test)

#Visualising the training set results
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train,regressor.predict(x_train), color = 'blue')
plt.title("Salary vs Experiance (Test Set)")
plt.xlabel("Year of Experiance")
plt.ylabel("Salary")
plt.show()